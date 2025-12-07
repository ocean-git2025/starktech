import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
import os
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, \
    classification_report, confusion_matrix, roc_curve, auc
import xgboost as xgb
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import re
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV, cross_val_score
import traceback
from tqdm import tqdm
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import time
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import optuna
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import shap
import joblib
import json
from collections import defaultdict

# ==================== 参数设置 ====================
FUTURE_DAYS = 20
LOOKBACK_DAYS = 30
USE_PKL_CACHE = False

# 数据路径,需改为本地电脑文件存储路径
PRICE_DATA_PATH = 'taiwan_stock_price_202511122027.csv'
REPORTS_DATA_PATH = 'reports_202511122033.csv'
PRE_MERGED_FILE = 'taiwan_stock_data_optimized.pkl'  # 预合并文件名
# 模型参数
RANDOM_STATE = 42
TEST_RATIO = 0.2
VAL_RATIO = 0.1
N_JOBS = -1  # 使用所有CPU核心

# 性能优化参数
MAX_SAMPLES = 200000
CHUNK_SIZE = 1000
FEATURE_SELECTION_THRESHOLD = 0.001

QUICK_MODE = False  # 启用快速模式
MAX_FEATURES = 50  # 限制特征数量
HYPERPARAM_TRIALS = 10  # 减少超参数搜索次数
SAMPLE_SIZE_TUNING = 5000  # 调优时的样本大小
MERGE_OPTIMIZATION = True  # 启用合并优化
QUICK_TUNING = True  # 快速调优模式
FORCE_REMERGE = False      # 是否强制重新合并（设为True可重新生成预合并文件）

# 扩展参数网格
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# ==================== 辅助函数 ====================
def timer_decorator(func):
    """计时装饰器"""

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f" {func.__name__} 执行时间: {end_time - start_time:.2f}秒")
        return result

    return wrapper

def get_conservative_params():
    """返回保守的模型参数，防止过拟合"""
    return {
        'rf': {
            'n_estimators': 50,        # 树数量
            'max_depth': 6,           # 深度
            'min_samples_split': 20,   # 分裂样本数
            'min_samples_leaf': 10,   # 叶节点样本）
            'max_features': 0.3,      # 特征采样比例
            'class_weight': 'balanced',
            'random_state': RANDOM_STATE,
            'n_jobs': N_JOBS
        },
        'xgb': {
            'n_estimators': 50,        # 树数量
            'max_depth': 3,           # 深度
            'learning_rate': 0.01,    # 学习率
            'subsample': 0.6,         # 采样比例
            'colsample_bytree': 0.6,  # 特征采样
            'reg_alpha': 1.0,         # L1正则
            'reg_lambda': 1.0,        # L2正则
            'scale_pos_weight': 1,    # 手动控制类别权重
            'random_state': RANDOM_STATE,
            'n_jobs': 1,
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        }
    }

def print_section(title):
    """打印章节标题"""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def reduce_memory_usage(df, verbose=True):
    """减少数据内存使用 - 修复了datetime64[ns, UTC+08:00]类型的问题"""
    start_mem = df.memory_usage().sum() / 1024 ** 2

    for col in df.columns:
        col_type = str(df[col].dtype)

        # 跳过日期列和非数值列
        if 'datetime' in col_type or col_type in ['object', 'category', 'bool', 'string']:
            continue

        if np.issubdtype(df[col].dtype, np.number):
            c_min = df[col].min()
            c_max = df[col].max()

            if np.issubdtype(df[col].dtype, np.integer):
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(f"内存使用减少: {100 * (start_mem - end_mem) / start_mem:.1f}%")
        print(f"从 {start_mem:.2f} MB 到 {end_mem:.2f} MB")

    return df


# ==================== 数据加载和处理 ====================
@timer_decorator
def load_and_preprocess_data():
    """加载和预处理数据 - 确保正确调用修复后的技术指标计算"""
    print_section("数据加载和预处理")

    # ==================== 1. 预合并文件检查 ====================

    if not FORCE_REMERGE and os.path.exists(PRE_MERGED_FILE):
        print(f"加载预合并文件: {PRE_MERGED_FILE}")
        try:
            with open(PRE_MERGED_FILE, 'rb') as f:
                data = pickle.load(f)

            # 适配两种数据格式
            if isinstance(data, pd.DataFrame):
                # 格式1: 只有DataFrame
                df = data
                # 自动提取特征列
                base_cols = ['date', 'stock_code', 'close', 'volume', 'open', 'high', 'low',
                             'future_return', 'market_avg_return', 'label']
                feature_cols = [col for col in df.columns
                                if col not in base_cols and pd.api.types.is_numeric_dtype(df[col])]
                print("检测到DataFrame格式，自动提取特征列")
                print(f"预合并数据加载成功: {df.shape}")
                print(f"时间范围: {df['date'].min()} 到 {df['date'].max()}")
                print(f"股票数量: {df['stock_code'].nunique()}")
                return df, feature_cols
            elif isinstance(data, tuple) and len(data) == 2:
                # 格式2: (df, feature_cols)
                df, feature_cols = data
                print("检测到元组格式: (DataFrame, feature_cols)")
                print(f"预合并数据加载成功: {df.shape}")
                print(f"时间范围: {df['date'].min()} 到 {df['date'].max()}")
                print(f"股票数量: {df['stock_code'].nunique()}")
                return df, feature_cols
            else:
                print(f"未知数据格式: {type(data)}")
                # 继续执行完整处理流程
        except Exception as e:
            print(f"预合并文件加载失败: {e}，重新处理...")

    # ==================== 2. 完整的数据处理流程 ====================
    print("执行完整的数据处理流程（这可能需要一些时间）...")

    try:
        # 1. 加载股价数据
        print(f"加载股价数据: {PRICE_DATA_PATH}")
        if PRICE_DATA_PATH.endswith('.csv'):
            price_df = pd.read_csv(PRICE_DATA_PATH, encoding='utf-8')
        else:
            price_df = pd.read_excel(PRICE_DATA_PATH)

        print(f"股价数据加载成功: {price_df.shape}")
        print(f"列名: {list(price_df.columns)}")

        # 查看前几行数据
        print("\n股价数据样例（前3行）:")
        print(price_df.head(3))

    except Exception as e:
        print(f"股价数据加载失败: {e}")
        return None

    # 2. 标准化列名
    print("标准化列名...")
    column_mapping = {
        'stock_id': 'stock_code', 'stock_code': 'stock_code', 'symbol': 'stock_code', 'number': 'stock_code',
        'date': 'date', 'Date': 'date', '交易日': 'date',
        'close': 'close', 'Close': 'close', '收盘价': 'close',
        'open': 'open', 'Open': 'open', '开盘价': 'open',
        'high': 'high', 'High': 'high', '最高价': 'high',
        'low': 'low', 'Low': 'low', '最低价': 'low',
        'max': 'high', 'min': 'low',
        'volume': 'volume', 'Volume': 'volume', '成交量': 'volume', 'trading_volume': 'volume',
        'trading_money': 'amount', '成交金额': 'amount',
        'spread': 'change', 'change': 'change', '涨跌': 'change',
        'turnover_rate': 'turnover_rate', 'trading_turnover': 'turnover_rate',
    }

    # 应用列名映射
    for old_col, new_col in column_mapping.items():
        if old_col in price_df.columns and new_col not in price_df.columns:
            price_df = price_df.rename(columns={old_col: new_col})
            print(f"   重命名: {old_col} -> {new_col}")

    # 检查必要的列
    required_cols = ['stock_code', 'date', 'close']
    missing_cols = [col for col in required_cols if col not in price_df.columns]
    if missing_cols:
        print(f"错误: 缺少必要列 {missing_cols}")
        print(f"可用列: {list(price_df.columns)}")
        return None

    # 3. 数据清洗
    print("数据清洗...")

    # 转换数据类型
    price_df['stock_code'] = price_df['stock_code'].astype(str).str.strip()

    # 修复日期转换问题 - 统一为无时区的datetime
    try:
        # 尝试不同的日期格式
        price_df['date'] = pd.to_datetime(price_df['date'], errors='coerce')
        # 移除时区信息
        if hasattr(price_df['date'].dtype, 'tz') and price_df['date'].dtype.tz is not None:
            price_df['date'] = price_df['date'].dt.tz_convert(None)
    except Exception as e:
        print(f"日期转换失败: {e}")
        return None

    # 移除无效日期
    initial_rows = len(price_df)
    price_df = price_df.dropna(subset=['date'])
    print(f"移除无效日期: {initial_rows - len(price_df):,} 行")

    # 按股票和日期排序
    price_df = price_df.sort_values(['stock_code', 'date'])

    # 移除重复行
    initial_rows = len(price_df)
    price_df = price_df.drop_duplicates(subset=['stock_code', 'date'])
    print(f"移除重复行: {initial_rows - len(price_df):,} 行")

    # 处理数值列
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 'change', 'turnover_rate']
    numeric_cols = [col for col in numeric_cols if col in price_df.columns]

    for col in numeric_cols:
        price_df[col] = pd.to_numeric(price_df[col], errors='coerce')

    # 按股票分组填充缺失值
    print("按股票填充缺失值...")
    for stock_code in tqdm(price_df['stock_code'].unique(), desc="填充缺失值"):
        stock_mask = price_df['stock_code'] == stock_code
        for col in numeric_cols:
            if col in price_df.columns:
                # 前向填充然后后向填充
                price_df.loc[stock_mask, col] = price_df.loc[stock_mask, col].ffill().bfill()

    # 移除仍有缺失值的行
    initial_size = len(price_df)
    price_df = price_df.dropna(subset=numeric_cols)
    print(f"移除缺失值行: {initial_size - len(price_df):,} 行")

    # 减少内存使用
    price_df = reduce_memory_usage(price_df)

    print(f"股价数据处理完成!")
    print(f"处理后的数据形状: {price_df.shape}")
    print(f"时间范围: {price_df['date'].min()} 到 {price_df['date'].max()}")
    print(f"股票数量: {price_df['stock_code'].nunique()}")

    # 4. 加载财报数据
    financial_df = None
    if os.path.exists(REPORTS_DATA_PATH):
        print(f"\n加载财报数据: {REPORTS_DATA_PATH}")
        try:
            if REPORTS_DATA_PATH.endswith('.csv'):
                financial_df = pd.read_csv(REPORTS_DATA_PATH, encoding='utf-8')
            else:
                financial_df = pd.read_excel(REPORTS_DATA_PATH)

            print(f"财报数据加载成功: {financial_df.shape}")

            # 优化财报数据处理
            if not financial_df.empty:
                print("使用优化版处理财报数据...")
                financial_wide = process_financial_data(financial_df)

                if financial_wide is not None and not financial_wide.empty:
                    print("使用优化版合并财报数据...")
                    price_df = merge_financial_data_optimized(price_df, financial_wide)

        except Exception as e:
            print(f"财报数据加载失败: {e}")
            financial_df = None

    # ==================== 5. 调用技术指标计算 ====================
    print("\n计算技术指标（使用修复版函数）...")

    # 验证价格数据质量
    print("🔍 验证价格数据质量...")
    if not validate_price_data(price_df):
        print("价格数据验证失败")
        return None

    # 调用技术指标计算函数
    try:
        price_df = calculate_technical_indicators(price_df)
        print(f"技术指标计算完成!")

        # 验证技术特征生成情况
        tech_cols = [col for col in price_df.columns
                     if any(pattern in col for pattern in
                            ['ma_', 'ema_', 'volatility_', 'momentum_', 'rsi_',
                             'macd_', 'bb_', 'atr_', 'obv_', 'volume_ratio_',
                             'price_vs_', 'return_', 'log_return', 'price_change'])]

        print(f"生成技术特征: {len(tech_cols)} 个")
        if tech_cols:
            print(f"技术特征示例: {tech_cols[:10]}...")

    except Exception as e:
        print(f"技术指标计算失败: {e}")
        import traceback
        traceback.print_exc()
        return None

    # ==================== 6. 计算未来收益率和标签 ====================
    print("\n计算未来收益率和标签（适合20天预测）...")

    # 在计算未来收益率之前添加验证
    print("验证价格数据质量...")
    if not validate_price_data(price_df):
        print("价格数据验证失败")
        return None

    # 未来收益率计算函数
    try:
        price_df = calculate_future_returns_and_labels(price_df, days=FUTURE_DAYS)

        if price_df.empty:
            print("计算未来收益率后数据为空")
            return None

        # 验证收益率计算
        if 'future_return' in price_df.columns:
            future_returns = price_df['future_return'].dropna()
            print(f"未来收益率计算完成!")
            print(f"有效收益率样本: {len(future_returns):,}")
            print(f"收益率范围: {future_returns.min():.4f} 到 {future_returns.max():.4f}")
            print(f"平均收益率: {future_returns.mean():.4f}")

    except Exception as e:
        print(f"❌ 未来收益率计算失败: {e}")
        import traceback
        traceback.print_exc()
        return None

    # ==================== 7. 特征工程 ====================
    print("\n特征工程...")
    try:
        price_df, feature_cols = create_features(price_df)

        if price_df is None or len(feature_cols) < 5:
            print("特征数量不足")
            return None

        print(f"特征工程完成!")
        print(f"总特征数量: {len(feature_cols)} 个")

        # 统计特征类型
        tech_features = [col for col in feature_cols if not col.startswith('fin_')]
        fin_features = [col for col in feature_cols if col.startswith('fin_')]
        other_features = [col for col in feature_cols if col not in tech_features and col not in fin_features]

        print(f"技术特征: {len(tech_features)} 个")
        print(f"财务特征: {len(fin_features)} 个")
        print(f"其他特征: {len(other_features)} 个")
        print(f"特征平衡比例: {len(tech_features)}:{len(fin_features)}")

    except Exception as e:
        print(f"特征工程失败: {e}")
        import traceback
        traceback.print_exc()
        return None

    # ==================== 8. 保存预合并文件 ====================
    print("\n保存预合并文件供后续快速加载...")
    try:
        with open(PRE_MERGED_FILE, 'wb') as f:
            pickle.dump((price_df, feature_cols), f, protocol=4)
        print(f"预合并数据已保存: {PRE_MERGED_FILE}")
        print("下次运行将直接加载此文件，速度提升10-100倍！")
    except Exception as e:
        print(f"预合并保存失败: {e}")

    return price_df, feature_cols


def emergency_fix_returns_simple(df, days=FUTURE_DAYS):
    """修复收益率计算 """
    print_section("修复收益率计算")

    # 创建数据副本
    df_fixed = df.copy()

    # 1. 移除零价格和无效数据
    print("1. 清理无效数据...")
    zero_mask = df_fixed['close'] <= 0
    print(f"   移除零价格: {zero_mask.sum()} 行")
    df_fixed = df_fixed[~zero_mask]

    # 2. 按股票和日期排序
    df_fixed = df_fixed.sort_values(['stock_code', 'date'])

    # 3. 重新计算未来收益率
    print("2. 重新计算未来收益率...")

    def simple_recalculate(group):
        group = group.sort_values('date')
        # 使用shift计算未来价格
        group['future_price'] = group['close'].shift(-days)
        # 计算收益率（添加安全性检查）
        valid_mask = (group['close'] > 0) & (group['future_price'] > 0)
        group['future_return_new'] = np.nan
        group.loc[valid_mask, 'future_return_new'] = (
                group.loc[valid_mask, 'future_price'] / group.loc[valid_mask, 'close'] - 1
        )
        return group

    try:
        df_fixed = df_fixed.groupby('stock_code', group_keys=False).apply(simple_recalculate)
        # 使用新计算的收益率
        df_fixed['future_return'] = df_fixed['future_return_new']
        print("收益率重新计算完成")
    except Exception as e:
        print(f"分组计算失败: {e}")
        return df  # 失败时返回原数据

    # 4. 处理特殊值
    print("3. 处理特殊值...")
    inf_mask = np.isinf(df_fixed['future_return'])
    if inf_mask.any():
        print(f"   修复 {inf_mask.sum()} 个inf值...")
        df_fixed.loc[inf_mask, 'future_return'] = np.nan

    # 5. 移除无效行
    initial_size = len(df_fixed)
    df_fixed = df_fixed.dropna(subset=['future_return'])
    final_size = len(df_fixed)
    print(f"有效数据: {final_size:,}/{initial_size:,} ({final_size / initial_size:.1%})")

    # 6. 验证修复结果
    future_returns = df_fixed['future_return'].dropna()
    if len(future_returns) > 0:
        print(f"紧急修复完成!")
        print(f"有效收益率: {len(future_returns):,}")
        print(f"范围: {future_returns.min():.6f} 到 {future_returns.max():.6f}")
        print(f"均值: {future_returns.mean():.6f}")
        print(f"inf值: {np.isinf(future_returns).sum()}")
    else:
        print("紧急修复后没有有效收益率!")

    return df_fixed

@timer_decorator
def merge_financial_data_optimized(price_df, financial_df):
    """优化财报数据合并-使用向量化操作提升性能"""
    if financial_df is None or financial_df.empty:
        return price_df

    print_section("优化合并财报数据")
    start_time = time.time()

    try:
        # 创建副本避免修改原数据
        price_df = price_df.copy()
        financial_df = financial_df.copy()

        # 确保股票代码格式一致
        price_df['stock_code'] = price_df['stock_code'].astype(str).str.strip()
        financial_df['stock_code'] = financial_df['stock_code'].astype(str).str.strip()

        # 处理日期
        if 'report_date' in financial_df.columns:
            financial_df['report_date'] = pd.to_datetime(financial_df['report_date'])
            # 移除时区信息
            if hasattr(financial_df['report_date'].dtype, 'tz') and financial_df['report_date'].dtype.tz is not None:
                financial_df['report_date'] = financial_df['report_date'].dt.tz_convert(None)

        # 找出共同股票
        common_stocks = set(price_df['stock_code'].unique()) & set(financial_df['stock_code'].unique())
        print(f"共同股票数量: {len(common_stocks)}")

        if len(common_stocks) == 0:
            print("没有共同股票,仅使用股价数据")
            return price_df

        # 方法1: 使用merge_asof进行快速合并(性能最佳)
        try:
            print("使用merge_asof进行快速合并...")
            # 只处理共同股票的数据
            price_common = price_df[price_df['stock_code'].isin(common_stocks)].copy()
            financial_common = financial_df[financial_df['stock_code'].isin(common_stocks)].copy()

            # 修复: 确保数据排序
            price_common = price_common.sort_values(['stock_code', 'date'])
            financial_common = financial_common.sort_values(['stock_code', 'report_date'])

            # 检查排序是否成功
            print(
                f"价格数据排序检查: 股票{price_common['stock_code'].iloc[0]}, 日期范围{price_common['date'].min()}到{price_common['date'].max()}")
            print(
                f"财报数据排序检查: 股票{financial_common['stock_code'].iloc[0]}, 日期范围{financial_common['report_date'].min()}到{financial_common['report_date'].max()}")

            # 使用merge_asof进行快速近似合并
            merged_df = pd.merge_asof(
                price_common,
                financial_common,
                left_on='date',
                right_on='report_date',
                by='stock_code',
                direction='backward'  # 找最近的小于等于当前日期的财报
            )

            # 处理没有财报数据的股票
            price_other = price_df[~price_df['stock_code'].isin(common_stocks)].copy()

            # 合并所有数据
            final_merged = pd.concat([merged_df, price_other], ignore_index=True)

            end_time = time.time()
            print(f"merge_asof合并完成! 形状: {final_merged.shape}")
            print(f"合并时间: {end_time - start_time:.2f}秒 (比原方法快10倍以上)")
            return final_merged

        except Exception as e:
            print(f"merge_asof失败，使用分组优化方法: {e}")
            # 回退到分组优化方法
            return merge_financial_data_grouped(price_df, financial_df, common_stocks)

    except Exception as e:
        print(f"优化合并失败: {e}")
        return price_df


def merge_financial_data_grouped(price_df, financial_df, common_stocks):
    """优化版分组合并方法 - 替代原有的grouped函数"""
    print("使用优化版分组合并方法...")
    start_time = time.time()

    # 使用列表推导式加速
    merged_chunks = []

    for stock_code in tqdm(common_stocks, desc="优化合并财报"):
        try:
            # 获取股票数据
            stock_prices = price_df[price_df['stock_code'] == stock_code].copy().sort_values('date')
            stock_financials = financial_df[financial_df['stock_code'] == stock_code].sort_values('report_date')

            if stock_financials.empty:
                merged_chunks.append(stock_prices)
                continue

            # 使用向量化操作加速
            price_dates = stock_prices['date'].values
            financial_dates = stock_financials['report_date'].values

            # 使用searchsorted进行快速查找
            indices = np.searchsorted(financial_dates, price_dates, side='right') - 1

            # 批量处理
            valid_indices = indices >= 0
            valid_price_indices = np.where(valid_indices)[0]

            if len(valid_price_indices) > 0:
                # 批量处理有效索引
                for i in valid_price_indices:
                    idx = indices[i]
                    latest_financial = stock_financials.iloc[idx]
                    price_row = stock_prices.iloc[i:i + 1].copy()

                    # 添加财务指标（只添加数值型指标）
                    for col, value in latest_financial.items():
                        if col not in ['stock_code', 'report_date'] and pd.api.types.is_numeric_dtype(
                                type(value)) and pd.notna(value):
                            price_row[f'fin_{col}'] = value

                    merged_chunks.append(price_row)

                # 处理没有财报数据的日期
                invalid_indices = np.where(~valid_indices)[0]
                if len(invalid_indices) > 0:
                    for i in invalid_indices:
                        merged_chunks.append(stock_prices.iloc[i:i + 1])
            else:
                # 所有日期都没有财报数据
                merged_chunks.append(stock_prices)

        except Exception as e:
            print(f"股票 {stock_code} 合并失败: {e}")
            # 即使失败也添加基础数据
            merged_chunks.append(price_df[price_df['stock_code'] == stock_code])

    # 合并所有块
    if merged_chunks:
        result_df = pd.concat(merged_chunks, ignore_index=True)
        end_time = time.time()
        print(f"优化分组合并完成! 形状: {result_df.shape}")
        print(f"合并时间: {end_time - start_time:.2f}秒")
        return result_df

    return price_df

@timer_decorator
def process_financial_data(financial_df):
    """处理财报数据"""
    if financial_df.empty:
        return pd.DataFrame()

    print_section("处理财报数据")

    print(f"财报数据形状: {financial_df.shape}")
    print(f"财报列名: {list(financial_df.columns)}")

    # 查看前几行数据
    print("\n财报数据样例（前5行）:")
    print(financial_df.head())

    # 创建副本
    df = financial_df.copy()

    # 去重
    df = df.drop_duplicates()
    print(f"去重后形状: {df.shape}")

    # 处理股票代码
    if 'number' in df.columns:
        df['stock_code'] = df['number'].astype(str).str.strip()
    elif 'symbol' in df.columns:
        df['stock_code'] = df['symbol'].astype(str).str.strip()
    else:
        print("使用第一列作为股票代码")
        df['stock_code'] = df.iloc[:, 0].astype(str).str.strip()

    print(f"股票数量: {df['stock_code'].nunique()}")

    # 财务指标映射
    financial_mapping = {
        '現金及約當現金': 'cash',
        'Cash and cash equivalents': 'cash',
        '流動資產合計': 'current_assets',
        'Total current assets': 'current_assets',
        '資產總計': 'total_assets',
        'Total assets': 'total_assets',
        '流動負債合計': 'current_liabilities',
        'Total current liabilities': 'current_liabilities',
        '負債合計': 'total_liabilities',
        'Total liabilities': 'total_liabilities',
        '股東權益合計': 'equity',
        'Total equity': 'equity',
        '應收帳款淨額': 'accounts_receivable',
        'Accounts receivable, net': 'accounts_receivable',
        '存貨': 'inventory',
        'Current inventories': 'inventory',
        '營業收入合計': 'revenue',
        'Total operating revenue': 'revenue',
        '營業成本合計': 'operating_costs',
        'Total operating costs': 'operating_costs',
        '營業毛利（毛損）': 'gross_profit',
        'Gross profit (loss)': 'gross_profit',
        '營業利益（損失）': 'operating_profit',
        'Operating profit (loss)': 'operating_profit',
        '本期稅後淨利（淨損）': 'net_profit',
        'Profit (loss)': 'net_profit',
        '基本每股盈餘合計': 'eps',
        'Total basic earnings per share': 'eps',
        '營業活動之淨現金流入（流出）': 'operating_cash_flow',
        'Net cash flows from (used in) operating activities': 'operating_cash_flow',
        '投資活動之淨現金流入（流出）': 'investing_cash_flow',
        'Net cash flows from (used in) investing activities': 'investing_cash_flow',
        '籌資活動之淨現金流入（流出）': 'financing_cash_flow',
        'Net cash flows from (used in) financing activities': 'financing_cash_flow'
    }

    def map_financial_indicator(key, key_en):
        if pd.isna(key) and pd.isna(key_en):
            return None

        key_str = str(key) if pd.notna(key) else ''
        key_en_str = str(key_en) if pd.notna(key_en) else ''

        # 先尝试中文匹配
        for chinese_name, std_name in financial_mapping.items():
            if chinese_name in key_str:
                return std_name

        # 再尝试英文匹配
        for english_name, std_name in financial_mapping.items():
            if english_name.lower() in key_en_str.lower():
                return std_name

        return None

    # 查找指标名称列
    indicator_col = None
    for col in ['key', 'key_en', 'indicator', 'account', 'item']:
        if col in df.columns:
            indicator_col = col
            break

    if indicator_col is None:
        print("使用第一列非股票代码列作为指标")
        indicator_col = df.columns[1] if len(df.columns) > 1 else None

    if indicator_col:
        # 处理数值
        if 'value' in df.columns:
            df['value'] = pd.to_numeric(df['value'], errors='coerce')

        # 处理日期
        if 'year' in df.columns and 'period' in df.columns:
            # 台湾财报日期通常: Q1(5/15), Q2(8/14), Q3(11/14), Q4(次年3/31)
            def get_report_date(row):
                try:
                    year = int(row['year'])
                    period = int(row['period'])

                    if period == 1:  # 第一季度
                        return pd.Timestamp(f"{year}-05-15")
                    elif period == 2:  # 第二季度
                        return pd.Timestamp(f"{year}-08-14")
                    elif period == 3:  # 第三季度
                        return pd.Timestamp(f"{year}-11-14")
                    elif period == 4:  # 第四季度
                        return pd.Timestamp(f"{year + 1}-03-31")
                    else:
                        return pd.NaT
                except:
                    return pd.NaT

            df['report_date'] = df.apply(get_report_date, axis=1)
        elif 'date' in df.columns:
            df['report_date'] = pd.to_datetime(df['date'], errors='coerce')
        else:
            print("无法确定财报日期，使用当前日期")
            df['report_date'] = datetime.now()

        # 移除无效日期
        df = df[df['report_date'].notna()]

        # 获取映射列
        key_col = indicator_col
        key_en_col = None
        for col in ['key_en', 'account_en', 'item_en']:
            if col in df.columns:
                key_en_col = col
                break

        if key_en_col:
            df['mapped_indicator'] = df.apply(
                lambda x: map_financial_indicator(x[key_col], x[key_en_col]), axis=1
            )
        else:
            df['mapped_indicator'] = df[key_col].apply(
                lambda x: map_financial_indicator(x, None)
            )

        # 统计映射结果
        mapped_count = df['mapped_indicator'].notna().sum()
        print(f"财务指标映射成功率: {mapped_count / len(df):.2%} ({mapped_count}/{len(df)})")

        if mapped_count > 0:
            # 转换为宽表格式
            financial_wide = df.pivot_table(
                index=['stock_code', 'report_date'],
                columns='mapped_indicator',
                values='value',
                aggfunc='first'
            ).reset_index()

            financial_wide.columns.name = None

            # 移除时区信息
            if hasattr(financial_wide['report_date'].dtype, 'tz') and financial_wide[
                'report_date'].dtype.tz is not None:
                financial_wide['report_date'] = financial_wide['report_date'].dt.tz_convert(None)

            # 计算财务比率
            print("计算财务比率...")

            if all(col in financial_wide.columns for col in ['revenue', 'operating_costs']):
                financial_wide['gross_margin'] = (financial_wide['revenue'] - financial_wide['operating_costs']) / \
                                                 financial_wide['revenue']
                print("  ✓ 计算毛利率")

            if all(col in financial_wide.columns for col in ['revenue', 'operating_profit']):
                financial_wide['operating_margin'] = financial_wide['operating_profit'] / financial_wide['revenue']
                print("  ✓ 计算营业利润率")

            if all(col in financial_wide.columns for col in ['revenue', 'net_profit']):
                financial_wide['net_margin'] = financial_wide['net_profit'] / financial_wide['revenue']
                print("  ✓ 计算净利率")

            if all(col in financial_wide.columns for col in ['current_assets', 'current_liabilities']):
                financial_wide['current_ratio'] = financial_wide['current_assets'] / financial_wide[
                    'current_liabilities']
                print("  ✓ 计算流动比率")

            if all(col in financial_wide.columns for col in ['total_assets', 'total_liabilities']):
                financial_wide['debt_to_assets'] = financial_wide['total_liabilities'] / financial_wide['total_assets']
                financial_wide['equity_ratio'] = 1 - financial_wide['debt_to_assets']
                print("  ✓ 计算资产负债率和权益比率")

            if all(col in financial_wide.columns for col in ['equity', 'net_profit']):
                financial_wide['roe'] = financial_wide['net_profit'] / financial_wide['equity']
                print("  ✓ 计算ROE")

            if all(col in financial_wide.columns for col in ['total_assets', 'net_profit']):
                financial_wide['roa'] = financial_wide['net_profit'] / financial_wide['total_assets']
                print("  ✓ 计算ROA")

            if all(col in financial_wide.columns for col in ['operating_cash_flow', 'total_liabilities']):
                financial_wide['ocf_to_debt'] = financial_wide['operating_cash_flow'] / financial_wide[
                    'total_liabilities']
                print("  ✓ 计算经营活动现金流/负债比率")

            if all(col in financial_wide.columns for col in ['operating_cash_flow', 'revenue']):
                financial_wide['ocf_margin'] = financial_wide['operating_cash_flow'] / financial_wide['revenue']
                print("  ✓ 计算经营活动现金流/收入比率")

            # 处理缺失值
            numeric_cols = [col for col in financial_wide.columns
                            if col not in ['stock_code', 'report_date'] and pd.api.types.is_numeric_dtype(
                    financial_wide[col])]

            for col in numeric_cols:
                if col in financial_wide.columns:
                    financial_wide[col] = financial_wide.groupby('stock_code')[col].transform(
                        lambda x: x.ffill().bfill().fillna(x.median())
                    )

            print(f"财报处理完成: {financial_wide.shape}")
            print(f"时间范围: {financial_wide['report_date'].min()} 到 {financial_wide['report_date'].max()}")

            return financial_wide

    print("财报数据处理失败，返回空DataFrame")
    return pd.DataFrame()


@timer_decorator
def calculate_basic_technical_features(stock_data):
    """技术特征 - 生成更多技术指标"""
    if stock_data.empty or 'close' not in stock_data.columns:
        return stock_data

    stock_data = stock_data.copy()
    close_prices = stock_data['close']

    try:
        # 1. 基础价格变化
        close_shifted = close_prices.shift(1)
        valid_mask = (close_shifted != 0) & close_shifted.notna()

        stock_data['price_change'] = 0.0
        stock_data.loc[valid_mask, 'price_change'] = (close_prices[valid_mask] - close_shifted[valid_mask]) / \
                                                     close_shifted[valid_mask]

        stock_data['log_return'] = 0.0
        stock_data.loc[valid_mask, 'log_return'] = np.log(close_prices[valid_mask] / close_shifted[valid_mask])

        # 2. 价格范围特征
        if all(col in stock_data.columns for col in ['high', 'low']):
            stock_data['high_low_range'] = stock_data['high'] - stock_data['low']
            stock_data['price_strength'] = (stock_data['close'] - stock_data['low']) / (
                        stock_data['high'] - stock_data['low']).replace(0, 1)

        # 3. 生成简单移动平均线
        for window in [3, 5, 10, 20]:
            ma_col = f'ma_{window}'
            stock_data[ma_col] = close_prices.rolling(window=window, min_periods=1).mean()
            stock_data[f'price_vs_ma{window}'] = close_prices / stock_data[ma_col] - 1

        # 4. 生成简单动量指标
        for period in [1, 5, 10]:
            momentum_col = f'momentum_{period}d'
            return_col = f'return_{period}d'
            shifted = close_prices.shift(period)
            valid_mask = (shifted != 0) & shifted.notna()
            stock_data[momentum_col] = 0.0
            stock_data[return_col] = 0.0
            stock_data.loc[valid_mask, momentum_col] = (close_prices[valid_mask] - shifted[valid_mask]) / shifted[
                valid_mask]
            stock_data.loc[valid_mask, return_col] = (close_prices[valid_mask] / shifted[valid_mask] - 1)

        # 5. 成交量相关指标
        if 'volume' in stock_data.columns:
            volume = stock_data['volume']
            for window in [5, 10, 20]:
                volume_ma = volume.rolling(window=window, min_periods=1).mean()
                valid_volume_mask = volume_ma != 0
                stock_data[f'volume_ratio_{window}'] = 1.0
                stock_data.loc[valid_volume_mask, f'volume_ratio_{window}'] = volume[valid_volume_mask] / volume_ma[
                    valid_volume_mask]

        print(
            f" 基础技术特征计算完成，生成特征: {len([col for col in stock_data.columns if col not in ['date', 'stock_code', 'close', 'volume']])}个")
        return stock_data

    except Exception as e:
        print(f"基础技术特征计算失败: {e}")
        return stock_data


# ==================== 修复：将函数移出嵌套 ====================
@timer_decorator
def calculate_technical_indicators(df):
    """修复版技术指标计算 - 确保生成20+个有效技术指标"""
    print_section("修复版技术指标计算")

    if df.empty or 'close' not in df.columns:
        print("数据为空或缺少close列")
        return df

    df_tech = df.copy()
    close_prices = df_tech['close']

    technical_features_generated = 0
    feature_categories = {}

    try:
        # 1. 基础价格变化特征 (确保这部分一定能生成)
        print("1. 计算基础价格变化特征...")
        close_shifted = close_prices.shift(1)
        valid_mask = (close_shifted > 0) & close_shifted.notna()

        # 价格变化率
        df_tech['price_change'] = 0.0
        df_tech.loc[valid_mask, 'price_change'] = (
                (close_prices[valid_mask] - close_shifted[valid_mask]) / close_shifted[valid_mask]
        )

        # 对数收益率
        df_tech['log_return'] = 0.0
        df_tech.loc[valid_mask, 'log_return'] = np.log(
            close_prices[valid_mask] / close_shifted[valid_mask]
        )

        technical_features_generated += 2
        feature_categories['price_change'] = 2
        print("生成2个基础价格变化特征")

    except Exception as e:
        print(f"基础价格变化特征失败: {e}")

    # 2. 移动平均线系列 (核心指标)
    try:
        print("2. 计算移动平均线系列...")
        ma_windows = [3, 5, 8, 10, 13, 20, 30, 50]

        for window in ma_windows:
            try:
                # 简单移动平均
                ma_col = f'ma_{window}'
                df_tech[ma_col] = close_prices.rolling(
                    window=window, min_periods=max(1, window // 2)
                ).mean()

                # 价格相对于移动平均的位置
                df_tech[f'price_vs_ma{window}'] = close_prices / df_tech[ma_col] - 1

                # 指数移动平均
                ema_col = f'ema_{window}'
                df_tech[ema_col] = close_prices.ewm(
                    span=window, min_periods=max(1, window // 2)
                ).mean()

                df_tech[f'price_vs_ema{window}'] = close_prices / df_tech[ema_col] - 1

                technical_features_generated += 4
                print(f"生成窗口{window}的4个移动平均特征")
            except Exception as e:
                print(f"窗口{window}移动平均计算失败: {e}")
                continue

        feature_categories['moving_averages'] = len(ma_windows) * 4

    except Exception as e:
        print(f"移动平均线计算失败: {e}")

    # 3. 动量指标系列
    try:
        print("3. 计算动量指标系列...")
        momentum_periods = [1, 2, 3, 5, 10, 20]

        for period in momentum_periods:
            try:
                # 简单动量
                shifted = close_prices.shift(period)
                valid_mask = (shifted > 0) & shifted.notna()

                momentum_col = f'momentum_{period}d'
                return_col = f'return_{period}d'

                df_tech[momentum_col] = 0.0
                df_tech[return_col] = 0.0

                df_tech.loc[valid_mask, momentum_col] = (
                        (close_prices[valid_mask] - shifted[valid_mask]) / shifted[valid_mask]
                )

                df_tech.loc[valid_mask, return_col] = (
                        close_prices[valid_mask] / shifted[valid_mask] - 1
                )

                technical_features_generated += 2
                print(f"生成周期{period}的2个动量特征")
            except Exception as e:
                print(f"周期{period}动量计算失败: {e}")
                continue

        feature_categories['momentum'] = len(momentum_periods) * 2

    except Exception as e:
        print(f"动量指标计算失败: {e}")

    # 4. 波动率指标
    try:
        print("4. 计算波动率指标...")
        volatility_windows = [5, 10, 20, 30]

        # 日收益率
        daily_returns = close_prices.pct_change()

        for window in volatility_windows:
            try:
                vol_col = f'volatility_{window}d'
                df_tech[vol_col] = daily_returns.rolling(
                    window=window, min_periods=max(1, window // 2)
                ).std()

                technical_features_generated += 1
                print(f"生成窗口{window}的波动率特征")
            except Exception as e:
                print(f"窗口{window}波动率计算失败: {e}")
                continue

        feature_categories['volatility'] = len(volatility_windows)

    except Exception as e:
        print(f"波动率指标计算失败: {e}")

    # 5. 成交量相关指标 (如果有成交量数据)
    if 'volume' in df_tech.columns:
        try:
            print("5. 计算成交量指标...")
            volume = df_tech['volume']
            volume_windows = [5, 10, 20]

            for window in volume_windows:
                try:
                    # 成交量移动平均
                    vol_ma_col = f'volume_ma_{window}'
                    df_tech[vol_ma_col] = volume.rolling(
                        window=window, min_periods=max(1, window // 2)
                    ).mean()

                    # 成交量比率
                    df_tech[f'volume_ratio_{window}'] = volume / df_tech[vol_ma_col]

                    technical_features_generated += 2
                    print(f"生成窗口{window}的2个成交量特征")
                except Exception as e:
                    print(f"窗口{window}成交量计算失败: {e}")
                    continue

            feature_categories['volume'] = len(volume_windows) * 2

        except Exception as e:
            print(f"成交量指标计算失败: {e}")

    # 6. RSI指标
    try:
        print("6. 计算RSI指标...")
        rsi_periods = [6, 14, 24]

        for period in rsi_periods:
            try:
                delta = close_prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()

                rs = gain / (loss + 1e-10)  # 避免除零
                rsi = 100 - (100 / (1 + rs))

                df_tech[f'rsi_{period}'] = rsi
                technical_features_generated += 1
                print(f"生成周期{period}的RSI特征")
            except Exception as e:
                print(f"周期{period}RSI计算失败: {e}")
                continue

        feature_categories['rsi'] = len(rsi_periods)

    except Exception as e:
        print(f"RSI计算失败: {e}")

    # 7. 价格位置特征 (需要high, low)
    if all(col in df_tech.columns for col in ['high', 'low']):
        try:
            print("7. 计算价格位置特征...")
            high = df_tech['high']
            low = df_tech['low']

            # 1. 当日价格强度
            print("   a. 计算当日价格强度...")
            try:
                range_mask = (high != low)
                df_tech['price_strength'] = 0.5
                df_tech.loc[range_mask, 'price_strength'] = (
                        (close_prices[range_mask] - low[range_mask]) /
                        (high[range_mask] - low[range_mask])
                )
                technical_features_generated += 1
                print("生成当日价格强度特征")
            except Exception as e:
                print(f"当日价格强度计算失败: {e}")

            # 2. 价格区间位置（3个时间窗口）
            print("   b. 计算价格区间位置...")
            windows = [5, 10, 20]
            for window in windows:
                try:
                    # 2.1 计算滚动窗口的最高价和最低价
                    high_roll = high.rolling(window=window, min_periods=1).max()
                    low_roll = low.rolling(window=window, min_periods=1).min()

                    # 2.2 创建有效掩码（避免除零）
                    range_mask = (high_roll != low_roll)

                    # 2.3 设置默认值
                    df_tech[f'price_position_{window}'] = 0.5

                    # 2.4 计算价格位置
                    df_tech.loc[range_mask, f'price_position_{window}'] = (
                            (close_prices[range_mask] - low_roll[range_mask]) /
                            (high_roll[range_mask] - low_roll[range_mask])
                    )

                    # 2.5 更新计数器
                    technical_features_generated += 1
                    feature_categories.setdefault('price_position', 0)
                    feature_categories['price_position'] += 1

                    print(f" 生成窗口{window}的价格位置特征")
                except Exception as e:
                    print(f"窗口{window}价格位置计算失败: {e}")
                    continue

            print(f"价格位置特征计算完成: 总计{feature_categories.get('price_position', 0)}个特征")

        except Exception as e:
            print(f"价格位置特征计算失败: {e}")

    # 最终统计
    print_section("技术指标生成统计")
    print(f"总生成技术特征: {technical_features_generated}个")
    for category, count in feature_categories.items():
        print(f"  {category}: {count}个")

    # 验证生成的特征
    tech_cols = [col for col in df_tech.columns
                 if any(pattern in col for pattern in
                        ['ma_', 'ema_', 'momentum_', 'return_', 'volatility_',
                         'volume_', 'rsi_', 'price_', 'change_'])]

    print(f"实际技术特征列: {len(tech_cols)}个")

    # 检查特征有效性
    valid_tech_cols = []
    for col in tech_cols:
        if col in df_tech.columns:
            non_na_ratio = df_tech[col].notna().mean()
            unique_vals = df_tech[col].nunique()
            if non_na_ratio > 0.5 and unique_vals > 1:
                valid_tech_cols.append(col)

    print(f"有效技术特征(非空>50%, 唯一值>1): {len(valid_tech_cols)}个")

    if len(valid_tech_cols) < 15:
        print("技术特征不足，执行紧急增强...")
        df_tech = emergency_enhance_technical_features(df_tech)

    return df_tech


def emergency_enhance_technical_features(df):
    """紧急增强技术特征 - 当常规方法失败时使用"""
    print("执行紧急技术特征增强...")

    if 'close' not in df.columns:
        return df

    close_prices = df['close']
    enhanced_features = []

    # 1. 基础比率特征
    try:
        if 'open' in df.columns:
            df['open_close_ratio'] = df['close'] / df['open'] - 1
            enhanced_features.append('open_close_ratio')

        if all(col in df.columns for col in ['high', 'low']):
            df['price_intensity'] = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, 1)
            df['daily_range_pct'] = (df['high'] - df['low']) / df['close']
            enhanced_features.extend(['price_intensity', 'daily_range_pct'])
    except Exception as e:
        print(f"基础比率特征计算失败: {e}")

    # 2. 简单移动平均线
    try:
        for window in [3, 5, 8, 13, 21, 34, 55]:  # 斐波那契数列窗口
            ma_col = f'emergency_ma_{window}'
            df[ma_col] = close_prices.rolling(window=window, min_periods=1).mean()
            enhanced_features.append(ma_col)
    except Exception as e:
        print(f"简单移动平均线计算失败: {e}")

    # 3. 简单动量指标
    try:
        for period in [1, 2, 3, 5, 8, 13]:
            mom_col = f'emergency_mom_{period}'
            df[mom_col] = close_prices.pct_change(period)
            enhanced_features.append(mom_col)
    except Exception as e:
        print(f"简单动量指标计算失败: {e}")

    # 4. 价格位置特征
    try:
        for window in [5, 10, 20]:
            high_col = f'emergency_high_{window}'
            low_col = f'emergency_low_{window}'
            df[high_col] = close_prices.rolling(window=window).max()
            df[low_col] = close_prices.rolling(window=window).min()
            df[f'emergency_position_{window}'] = (close_prices - df[low_col]) / (df[high_col] - df[low_col]).replace(0,
                                                                                                                     1)
            enhanced_features.extend([high_col, low_col, f'emergency_position_{window}'])
    except Exception as e:
        print(f"价格位置特征计算失败: {e}")

    print(f"紧急增强完成: {len(enhanced_features)}个特征")
    return df
# ==================== 辅助函数保持不变 ====================
def calculate_enhanced_moving_averages(stock_data):
    """移动平均线计算"""
    if 'close' not in stock_data.columns:
        return stock_data

    close_prices = stock_data['close']

    # 扩展窗口范围
    key_windows = [3, 5, 10, 20, 30, 60]
    for window in key_windows:
        # 简单移动平均
        ma_col = f'ma_{window}'
        stock_data[ma_col] = close_prices.rolling(window=window, min_periods=1).mean()
        stock_data[f'price_vs_ma{window}'] = close_prices / stock_data[ma_col] - 1

        # 指数移动平均
        ema_col = f'ema_{window}'
        stock_data[ema_col] = close_prices.ewm(span=window, min_periods=1).mean()
        stock_data[f'price_vs_ema{window}'] = close_prices / stock_data[ema_col] - 1

    return stock_data


def calculate_enhanced_momentum_indicators(stock_data):
    """动量指标计算"""
    if 'close' not in stock_data.columns:
        return stock_data

    close_prices = stock_data['close']

    try:
        # 价格动量
        for period in [1, 5, 10, 20]:
            momentum_col = f'momentum_{period}d'
            return_col = f'return_{period}d'
            shifted = close_prices.shift(period)
            valid_mask = (shifted != 0) & shifted.notna()
            stock_data[momentum_col] = 0.0
            stock_data[return_col] = 0.0
            stock_data.loc[valid_mask, momentum_col] = (close_prices[valid_mask] - shifted[valid_mask]) / shifted[
                valid_mask]
            stock_data.loc[valid_mask, return_col] = (close_prices[valid_mask] / shifted[valid_mask] - 1)

        # RSI指标
        period = 14
        delta = close_prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        rs = avg_gain / avg_loss.replace(0, 1)
        rs = rs.replace([np.inf, -np.inf], 1).fillna(1)
        stock_data[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        return stock_data
    except Exception as e:
        print(f"动量指标计算失败: {e}")
        return stock_data

def calculate_enhanced_volatility_indicators(stock_data):
    """波动率指标计算"""
    if 'close' not in stock_data.columns:
        return stock_data

    close_prices = stock_data['close']
    high_prices = stock_data['high'] if 'high' in stock_data.columns else stock_data['close']
    low_prices = stock_data['low'] if 'low' in stock_data.columns else stock_data['close']

    try:
        # 波动率
        window = 20
        close_shifted = close_prices.shift(1)
        valid_mask = (close_shifted != 0) & close_shifted.notna()
        daily_returns = np.zeros(len(close_prices))
        daily_returns[valid_mask] = (close_prices[valid_mask] - close_shifted[valid_mask]) / close_shifted[valid_mask]
        stock_data[f'volatility_{window}d'] = pd.Series(daily_returns).rolling(window=window, min_periods=1).std()

        # 布林带位置
        if 'ma_20' in stock_data.columns:
            ma_20 = stock_data['ma_20']
            std_20 = close_prices.rolling(window=20, min_periods=1).std()
            bb_upper = ma_20 + 2 * std_20
            bb_lower = ma_20 - 2 * std_20
            bb_width = bb_upper - bb_lower
            valid_bb_mask = bb_width != 0
            stock_data['bb_position_20'] = 0.5
            stock_data.loc[valid_bb_mask, 'bb_position_20'] = (close_prices[valid_bb_mask] - bb_lower[valid_bb_mask]) / \
                                                              bb_width[valid_bb_mask]

        return stock_data
    except Exception as e:
        print(f"波动率指标计算失败: {e}")
        return stock_data


def calculate_enhanced_volume_indicators(stock_data):
    """成交量指标计算"""
    if 'volume' not in stock_data.columns:
        return stock_data

    volume = stock_data['volume']
    close_prices = stock_data['close']

    try:
        # 成交量比率
        window = 20
        volume_ma = volume.rolling(window=window, min_periods=1).mean()
        valid_volume_mask = volume_ma != 0
        stock_data['volume_ratio_20'] = 1.0
        stock_data.loc[valid_volume_mask, 'volume_ratio_20'] = volume[valid_volume_mask] / volume_ma[valid_volume_mask]

        return stock_data
    except Exception as e:
        print(f"成交量指标计算失败: {e}")
        return stock_data


@timer_decorator
def calculate_future_returns_and_labels(df, days=FUTURE_DAYS):
    """未来收益率计算 """
    print_section("收益率计算")

    if df.empty or 'close' not in df.columns:
        print("数据为空或缺少close列")
        return df

    # 创建数据副本
    df_fixed = df.copy()
    df_fixed = df_fixed.sort_values(['stock_code', 'date'])

    print(f"使用收益率计算，预期间隔: {days}个交易日")
    print(f"原始数据形状: {df_fixed.shape}")
    print(f"股票数量: {df_fixed['stock_code'].nunique()}")

    # ==================== 关键修复开始 ====================
    def safe_calculate_returns(group):
        """安全计算收益率 - 避免inf和除零错误"""
        group = group.sort_values('date')

        if len(group) < days + 1:
            group['future_return'] = np.nan
            return group

        close_prices = group['close'].values
        returns = np.full(len(group), np.nan)

        for i in range(len(group)):
            if i + days < len(group):
                current_price = close_prices[i]
                future_price = close_prices[i + days]

                #关键修复：严格的价格有效性检查
                if (current_price > 0 and future_price > 0 and
                        not np.isnan(current_price) and not np.isnan(future_price) and
                        not np.isinf(current_price) and not np.isinf(future_price)):

                    # 计算收益率
                    return_val = (future_price / current_price) - 1

                    # 修复：限制收益率范围，避免极端值
                    if return_val < -0.9:  # 限制最大亏损90%
                        return_val = -0.9
                    elif return_val > 10.0:  # 限制最大收益1000%
                        return_val = 10.0

                    # 修复：检查是否为有限值
                    if np.isfinite(return_val):
                        returns[i] = return_val

        group['future_return'] = returns
        return group

    # 应用修复计算
    print("应用分组计算收益率...")
    df_fixed = df_fixed.groupby('stock_code', group_keys=False).apply(safe_calculate_returns)

    # 移除无效行
    initial_size = len(df_fixed)
    df_fixed = df_fixed.dropna(subset=['future_return'])
    removed_count = initial_size - len(df_fixed)
    print(f"收益率计算完成，移除无效数据: {removed_count:,}行")

    # 修复：额外检查并处理无穷值
    if 'future_return' in df_fixed.columns:
        future_returns = df_fixed['future_return']

        # 检查无穷大值
        inf_mask = np.isinf(future_returns)
        inf_count = inf_mask.sum()

        if inf_count > 0:
            print(f"发现无穷大收益率: {inf_count}个，将其设置为NaN")
            df_fixed.loc[inf_mask, 'future_return'] = np.nan

        # 检查NaN值
        nan_count = future_returns.isna().sum()
        if nan_count > 0:
            print(f"移除NaN收益率: {nan_count}个")
            df_fixed = df_fixed.dropna(subset=['future_return'])

    # 验证修复结果
    if 'future_return' in df_fixed.columns and len(df_fixed) > 0:
        future_returns = df_fixed['future_return'].dropna()

        if len(future_returns) > 0:
            print(f"彻底修复后收益率统计:")
            print(f"有效样本: {len(future_returns):,}")
            print(f"范围: {future_returns.min():.6f} 到 {future_returns.max():.6f}")
            print(f"均值: {future_returns.mean():.6f}")

            # 检查是否还有无效值
            if np.isinf(future_returns).any() or np.isnan(future_returns).any():
                print("仍然存在无效收益率，进行紧急处理...")
                median_return = future_returns.replace([np.inf, -np.inf], np.nan).median()
                df_fixed['future_return'] = df_fixed['future_return'].replace(
                    [np.inf, -np.inf], median_return
                )
        else:
            print("修复后没有有效收益率！")

    # ==================== 标签计算部分 ====================
    print("计算市场平均收益率和标签...")

    # 计算市场平均收益率
    daily_avg_return = df_fixed.groupby('date')['future_return'].mean().reset_index()
    daily_avg_return.columns = ['date', 'market_avg_return']
    df_fixed = pd.merge(df_fixed, daily_avg_return, on='date', how='left')

    # 使用分位数方法定义标签（更稳健）
    def calculate_smart_labels(group):
        if len(group) < 10:
            group['label'] = 0
            return group

        future_returns = group['future_return']

        # 方法1：使用分位数
        try:
            quantile_threshold = future_returns.quantile(0.6)  # 前40%为正样本
            group['label'] = (future_returns > quantile_threshold).astype(int)
        except:
            # 回退方法：使用市场平均
            market_avg = group['market_avg_return'].mean()
            group['label'] = (future_returns > market_avg).astype(int)

        return group

    df_fixed = df_fixed.groupby('date', group_keys=False).apply(calculate_smart_labels)

    # 验证标签有效性
    print("验证标签有效性...")
    if 'label' in df_fixed.columns and 'future_return' in df_fixed.columns:
        positive_mask = df_fixed['label'] == 1
        negative_mask = df_fixed['label'] == 0

        if positive_mask.any() and negative_mask.any():
            positive_return = df_fixed[positive_mask]['future_return'].mean()
            negative_return = df_fixed[negative_mask]['future_return'].mean()
            return_diff = positive_return - negative_return

            print("✅ 标签有效性验证:")
            print(f"  正样本平均收益: {positive_return:.6f} ({positive_return:.4%})")
            print(f"  负样本平均收益: {negative_return:.6f} ({negative_return:.4%})")
            print(f"  收益差异: {return_diff:.6f} ({return_diff:.4%})")
            print(f"  正样本比例: {df_fixed['label'].mean():.2%}")

            if return_diff < 0.01:
                print("❌ 标签区分度不足，尝试调整...")
                # 使用更严格的分位数
                try:
                    df_fixed = df_fixed.groupby('date', group_keys=False).apply(
                        lambda x: x.assign(label=(x['future_return'] > x['future_return'].quantile(0.7)).astype(int))
                    )
                    # 重新验证
                    positive_return = df_fixed[df_fixed['label'] == 1]['future_return'].mean()
                    negative_return = df_fixed[df_fixed['label'] == 0]['future_return'].mean()
                    return_diff = positive_return - negative_return
                    print(f"调整后收益差异: {return_diff:.4f} ({return_diff:.2%})")
                except Exception as e:
                    print(f"调整失败: {e}")
        else:
            print("❌ 无法验证标签有效性：缺少正样本或负样本")

    print(f"标签计算完成! 正样本比例: {df_fixed['label'].mean():.2%}")
    return df_fixed


def filter_financial_features_by_importance(df, financial_features, target_count):
    """筛选财务特征"""
    if len(financial_features) <= target_count:
        return financial_features

    print(f"筛选财务特征: {len(financial_features)} -> {target_count}个")

    financial_features_filtered = []

    # 方法1：使用与label的相关性进行筛选
    if 'label' in df.columns:
        financial_correlations = []
        for col in financial_features:
            try:
                if df[col].notna().sum() > 100:
                    corr = abs(df[col].corr(df['label']))
                    if not np.isnan(corr):
                        financial_correlations.append((col, corr))
            except:
                continue

        if financial_correlations:
            financial_correlations.sort(key=lambda x: x[1], reverse=True)
            selected_financial = [col for col, corr in financial_correlations[:target_count]]
            financial_features_filtered = selected_financial
            print(f"基于相关性筛选: {len(selected_financial)}个财务特征")
        else:
            financial_features_filtered = financial_features[:target_count]
            print(f"使用简单截取: {len(financial_features_filtered)}个财务特征")
    else:
        # 如果没有label，使用方差筛选
        financial_variances = []
        for col in financial_features:
            try:
                variance = df[col].var()
                if not np.isnan(variance):
                    financial_variances.append((col, variance))
            except:
                continue

        if financial_variances:
            financial_variances.sort(key=lambda x: x[1], reverse=True)
            financial_features_filtered = [col for col, var in financial_variances[:target_count]]
            print(f"基于方差筛选: {len(financial_features_filtered)}个财务特征")
        else:
            financial_features_filtered = financial_features[:target_count]
            print(f"使用简单截取: {len(financial_features_filtered)}个财务特征")

    return financial_features_filtered


@timer_decorator
def create_features(df):
    """特征工程 - 确保技术特征和财务特征平衡"""
    print_section("特征平衡优化")

    if df.empty:
        return df, []

    # 基础列（不包含在特征中）
    base_cols = ['date', 'stock_code', 'close', 'volume', 'open', 'high', 'low',
                 'future_return', 'market_avg_return', 'label']

    # 1. 收集所有数值型特征
    all_numeric_cols = []
    for col in df.columns:
        if (col not in base_cols and
                pd.api.types.is_numeric_dtype(df[col]) and
                df[col].nunique() > 1 and
                df[col].notna().mean() > 0.3):  # 降低非空阈值到30%
            all_numeric_cols.append(col)

    print(f"所有数值型特征: {len(all_numeric_cols)}个")

    if len(all_numeric_cols) == 0:
        print("没有找到数值型特征")
        return df, []

    # 2. 重新定义特征分类模式 - 更全面的匹配
    tech_patterns = [
        'ma_', 'ema_', 'volatility_', 'momentum_', 'rsi_', 'macd_', 'bb_', 'atr_', 'obv_',
        'volume_ratio_', 'price_vs_', 'return_', 'log_return', 'price_change', 'change_',
        'breakout_', 'strength_', 'position_', 'ratio_', 'signal_', 'index_', 'oscillator_'
    ]

    # 3. 分类特征
    tech_features = []
    financial_features = []
    other_features = []

    for col in all_numeric_cols:
        # 优先识别财务特征
        if any(col.startswith(pattern) for pattern in ['fin_', 'financial_']):
            financial_features.append(col)
        # 识别技术特征
        elif any(pattern in col for pattern in tech_patterns):
            tech_features.append(col)
        # 识别其他财务特征（基于关键词）
        elif any(keyword in col.lower() for keyword in
                 ['cash', 'asset', 'liability', 'equity', 'revenue', 'profit',
                  'margin', 'debt', 'flow', 'eps', 'roe', 'roa']):
            financial_features.append(col)
        else:
            other_features.append(col)

    print(f"初始特征统计:")
    print(f"技术特征: {len(tech_features)}个")
    print(f"财务特征: {len(financial_features)}个")
    print(f"其他特征: {len(other_features)}个")

    # 4. 目标平衡比例
    target_tech = 25  # 技术特征目标
    target_fin = 35  # 财务特征目标

    # 5. 如果技术特征不足，执行紧急增强
    if len(tech_features) < target_tech:
        print(f"技术特征不足({len(tech_features)}个)，执行紧急增强...")
        df = emergency_enhance_technical_features(df)

        # 重新收集特征
        all_numeric_cols = []
        for col in df.columns:
            if (col not in base_cols and
                    pd.api.types.is_numeric_dtype(df[col]) and
                    df[col].nunique() > 1 and
                    df[col].notna().mean() > 0.3):
                all_numeric_cols.append(col)

        # 重新分类
        tech_features = []
        financial_features = []
        other_features = []

        for col in all_numeric_cols:
            if any(col.startswith(pattern) for pattern in ['fin_', 'financial_']):
                financial_features.append(col)
            elif any(pattern in col for pattern in tech_patterns):
                tech_features.append(col)
            elif any(keyword in col.lower() for keyword in
                     ['cash', 'asset', 'liability', 'equity', 'revenue', 'profit',
                      'margin', 'debt', 'flow', 'eps', 'roe', 'roa']):
                financial_features.append(col)
            else:
                other_features.append(col)

        print(f"增强后特征统计:")
        print(f" 技术特征: {len(tech_features)}个")
        print(f" 财务特征: {len(financial_features)}个")
        print(f" 其他特征: {len(other_features)}个")

    # 6. 简化平衡策略
    print("执行简化平衡策略...")

    # 6.1 如果技术特征仍然不足，从其他特征中借用
    if len(tech_features) < target_tech and len(other_features) > 0:
        print(f"技术特征仍不足({len(tech_features)}个)，从其他特征中借用...")

        # 计算其他特征与标签的相关性（如果可用）
        correlations = []
        if 'label' in df.columns:
            for col in other_features:
                try:
                    if df[col].notna().sum() > 50:  # 降低样本数量要求
                        corr = abs(df[col].corr(df['label']))
                        if not np.isnan(corr):
                            correlations.append((col, corr))
                except:
                    continue

            if correlations:
                correlations.sort(key=lambda x: x[1], reverse=True)
                # 借用相关性最高的特征
                borrow_count = min(target_tech - len(tech_features), len(correlations), 10)  # 最多借10个
                borrowed_features = [col for col, corr in correlations[:borrow_count]]
                tech_features.extend(borrowed_features)
                # 从其他特征中移除
                other_features = [col for col in other_features if col not in borrowed_features]
                print(f"  借用 {len(borrowed_features)} 个高相关性特征给技术特征")

    # 6.2 如果财务特征过多，进行筛选
    if len(financial_features) > target_fin:
        print(f"财务特征过多({len(financial_features)}个)，进行筛选...")

        # 使用相关性筛选
        fin_correlations = []
        if 'label' in df.columns:
            for col in financial_features:
                try:
                    if df[col].notna().sum() > 50:  # 降低样本数量要求
                        corr = abs(df[col].corr(df['label']))
                        if not np.isnan(corr):
                            fin_correlations.append((col, corr))
                except:
                    continue

            if fin_correlations:
                fin_correlations.sort(key=lambda x: x[1], reverse=True)
                financial_features = [col for col, corr in fin_correlations[:target_fin]]
                print(f"  基于相关性筛选到 {len(financial_features)} 个财务特征")
            else:
                # 使用方差筛选
                variances = []
                for col in financial_features:
                    try:
                        variance = df[col].var()
                        if not np.isnan(variance):
                            variances.append((col, variance))
                    except:
                        continue

                if variances:
                    variances.sort(key=lambda x: x[1], reverse=True)
                    financial_features = [col for col, var in variances[:target_fin]]
                    print(f"  基于方差筛选到 {len(financial_features)} 个财务特征")
                else:
                    # 简单截取
                    financial_features = financial_features[:target_fin]
                    print(f"  简单截取到 {len(financial_features)} 个财务特征")

    # 6.3 最终特征合并
    selected_features = tech_features + financial_features

    # 确保特征数量在合理范围内
    total_target = target_tech + target_fin
    if len(selected_features) > total_target * 1.5:
        print(f"特征数量超额({len(selected_features)}个)，进行最终精简...")
        # 优先保留技术特征
        tech_keep = min(len(tech_features), int(total_target * 0.4))
        fin_keep = min(len(financial_features), total_target - tech_keep)
        selected_features = tech_features[:tech_keep] + financial_features[:fin_keep]
        print(f"精简到: {len(selected_features)}个特征")

    # 7. 最终统计
    tech_selected = [col for col in selected_features if col in tech_features]
    fin_selected = [col for col in selected_features if col in financial_features]
    other_selected = [col for col in selected_features if col in other_features]

    print(f"特征平衡完成!")
    print(f"最终技术特征: {len(tech_selected)}个")
    print(f"最终财务特征: {len(fin_selected)}个")
    print(f"其他特征: {len(other_selected)}个")
    print(f"平衡比例: {len(tech_selected)}:{len(fin_selected)} (目标: {target_tech}:{target_fin})")
    print(f"总特征数量: {len(selected_features)}个")

    # 显示特征示例
    if len(selected_features) > 0:
        print(f"技术特征示例: {tech_selected[:5] if tech_selected else '无'}")
        print(f"财务特征示例: {fin_selected[:5] if fin_selected else '无'}")

    return df, selected_features


def emergency_enhance_technical_features(df):
    """紧急增强技术特征"""
    print("执行紧急技术特征增强...")

    if 'close' not in df.columns:
        return df

    df_enhanced = df.copy()
    close_prices = df_enhanced['close']
    enhanced_features = []

    try:
        # 1. 基础比率特征
        print("1. 计算基础比率特征...")
        if 'open' in df_enhanced.columns:
            df_enhanced['emergency_open_close_ratio'] = df_enhanced['close'] / df_enhanced['open'] - 1
            enhanced_features.append('emergency_open_close_ratio')

        if all(col in df_enhanced.columns for col in ['high', 'low']):
            df_enhanced['emergency_price_intensity'] = (df_enhanced['close'] - df_enhanced['low']) / (
                        df_enhanced['high'] - df_enhanced['low']).replace(0, 1)
            df_enhanced['emergency_daily_range_pct'] = (df_enhanced['high'] - df_enhanced['low']) / df_enhanced['close']
            enhanced_features.extend(['emergency_price_intensity', 'emergency_daily_range_pct'])

        print(f"生成{len([f for f in enhanced_features if 'emergency' in f])}个基础比率特征")

    except Exception as e:
        print(f"基础比率特征计算失败: {e}")

    try:
        # 2. 简单移动平均线 (使用斐波那契数列窗口)
        print("2. 计算简单移动平均线...")
        fib_windows = [3, 5, 8, 13, 21, 34, 55]  # 斐波那契数列窗口

        for window in fib_windows:
            try:
                ma_col = f'emergency_ma_{window}'
                df_enhanced[ma_col] = close_prices.rolling(window=window, min_periods=1).mean()
                enhanced_features.append(ma_col)

                # 价格相对于移动平均的位置
                df_enhanced[f'emergency_price_vs_ma{window}'] = close_prices / df_enhanced[ma_col] - 1
                enhanced_features.append(f'emergency_price_vs_ma{window}')
            except Exception as e:
                print(f"窗口{window}移动平均计算失败: {e}")
                continue

        print(f"生成{len(fib_windows) * 2}个移动平均相关特征")

    except Exception as e:
        print(f"移动平均线计算失败: {e}")

    try:
        # 3. 简单动量指标
        print("3. 计算简单动量指标...")
        mom_periods = [1, 2, 3, 5, 8, 13]  # 斐波那契数列周期

        for period in mom_periods:
            try:
                mom_col = f'emergency_mom_{period}d'
                df_enhanced[mom_col] = close_prices.pct_change(period)
                enhanced_features.append(mom_col)
            except Exception as e:
                print(f"周期{period}动量计算失败: {e}")
                continue

        print(f"生成{len(mom_periods)}个动量特征")

    except Exception as e:
        print(f"动量指标计算失败: {e}")

    try:
        # 4. 价格位置特征
        print("4. 计算价格位置特征...")
        position_windows = [5, 10, 20]

        for window in position_windows:
            try:
                high_col = f'emergency_high_{window}'
                low_col = f'emergency_low_{window}'

                df_enhanced[high_col] = close_prices.rolling(window=window, min_periods=1).max()
                df_enhanced[low_col] = close_prices.rolling(window=window, min_periods=1).min()

                position_col = f'emergency_position_{window}'
                df_enhanced[position_col] = (close_prices - df_enhanced[low_col]) / (
                            df_enhanced[high_col] - df_enhanced[low_col]).replace(0, 1)

                enhanced_features.extend([high_col, low_col, position_col])
            except Exception as e:
                print(f"窗口{window}价格位置计算失败: {e}")
                continue

        print(f"生成{len(position_windows) * 3}个价格位置特征")

    except Exception as e:
        print(f"价格位置特征计算失败: {e}")

    try:
        # 5. 波动率特征
        print("5. 计算波动率特征...")
        vol_windows = [5, 10, 20]

        for window in vol_windows:
            try:
                vol_col = f'emergency_volatility_{window}d'
                df_enhanced[vol_col] = close_prices.pct_change().rolling(window=window, min_periods=1).std()
                enhanced_features.append(vol_col)
            except Exception as e:
                print(f"窗口{window}波动率计算失败: {e}")
                continue

        print(f"生成{len(vol_windows)}个波动率特征")

    except Exception as e:
        print(f"波动率特征计算失败: {e}")

    try:
        # 6. 成交量特征 (如果有成交量数据)
        if 'volume' in df_enhanced.columns:
            print("6. 计算成交量特征...")
            volume = df_enhanced['volume']
            volume_windows = [5, 10, 20]

            for window in volume_windows:
                try:
                    # 成交量移动平均
                    vol_ma_col = f'emergency_volume_ma_{window}'
                    df_enhanced[vol_ma_col] = volume.rolling(window=window, min_periods=1).mean()
                    enhanced_features.append(vol_ma_col)

                    # 成交量比率
                    vol_ratio_col = f'emergency_volume_ratio_{window}'
                    # 避免除零
                    valid_mask = df_enhanced[vol_ma_col] != 0
                    df_enhanced[vol_ratio_col] = 1.0
                    df_enhanced.loc[valid_mask, vol_ratio_col] = volume[valid_mask] / df_enhanced.loc[
                        valid_mask, vol_ma_col]
                    enhanced_features.append(vol_ratio_col)
                except Exception as e:
                    print(f"窗口{window}成交量计算失败: {e}")
                    continue

            print(f"生成{len(volume_windows) * 2}个成交量特征")

    except Exception as e:
        print(f"成交量特征计算失败: {e}")

    try:
        # 7. RSI指标
        print("7. 计算RSI指标...")
        rsi_periods = [6, 14]

        for period in rsi_periods:
            try:
                delta = close_prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()

                rs = gain / (loss + 1e-10)  # 避免除零
                rsi = 100 - (100 / (1 + rs))

                rsi_col = f'emergency_rsi_{period}'
                df_enhanced[rsi_col] = rsi
                enhanced_features.append(rsi_col)
            except Exception as e:
                print(f"周期{period}RSI计算失败: {e}")
                continue

        print(f"生成{len(rsi_periods)}个RSI特征")

    except Exception as e:
        print(f"RSI指标计算失败: {e}")

    try:
        # 8. 价格变化特征
        print("8. 计算价格变化特征...")
        change_periods = [1, 2, 3, 5]

        for period in change_periods:
            try:
                # 绝对价格变化
                change_col = f'emergency_price_change_{period}d'
                df_enhanced[change_col] = close_prices.diff(period)
                enhanced_features.append(change_col)

                # 百分比价格变化
                pct_change_col = f'emergency_pct_change_{period}d'
                df_enhanced[pct_change_col] = close_prices.pct_change(period)
                enhanced_features.append(pct_change_col)
            except Exception as e:
                print(f"周期{period}价格变化计算失败: {e}")
                continue

        print(f"生成{len(change_periods) * 2}个价格变化特征")

    except Exception as e:
        print(f"价格变化特征计算失败: {e}")

    try:
        # 9. 价格加速度特征
        print("9. 计算价格加速度特征...")
        try:
            # 一阶差分（速度）
            velocity = close_prices.diff()
            # 二阶差分（加速度）
            acceleration = velocity.diff()

            df_enhanced['emergency_price_velocity'] = velocity
            df_enhanced['emergency_price_acceleration'] = acceleration
            enhanced_features.extend(['emergency_price_velocity', 'emergency_price_acceleration'])

            print("生成2个价格加速度特征")
        except Exception as e:
            print(f"价格加速度计算失败: {e}")

    except Exception as e:
        print(f"价格加速度特征计算失败: {e}")

    try:
        # 10. 价格波动特征
        print("10. 计算价格波动特征...")
        volatility_windows = [5, 10, 20]

        for window in volatility_windows:
            try:
                # 真实波动幅度（True Range）
                if all(col in df_enhanced.columns for col in ['high', 'low']):
                    tr1 = df_enhanced['high'] - df_enhanced['low']
                    tr2 = abs(df_enhanced['high'] - df_enhanced['close'].shift(1))
                    tr3 = abs(df_enhanced['low'] - df_enhanced['close'].shift(1))
                    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

                    atr_col = f'emergency_atr_{window}'
                    df_enhanced[atr_col] = true_range.rolling(window=window, min_periods=1).mean()
                    enhanced_features.append(atr_col)

                    # 标准化真实波动幅度
                    atr_pct_col = f'emergency_atr_pct_{window}'
                    df_enhanced[atr_pct_col] = df_enhanced[atr_col] / close_prices
                    enhanced_features.append(atr_pct_col)
            except Exception as e:
                print(f"窗口{window}波动特征计算失败: {e}")
                continue

        print(f"生成{len(volatility_windows) * 2}个波动特征")

    except Exception as e:
        print(f"价格波动特征计算失败: {e}")

    # 最终统计
    actual_enhanced_features = [col for col in enhanced_features if col in df_enhanced.columns]
    print(f"紧急增强完成! 生成{len(actual_enhanced_features)}个技术特征")

    # 显示生成的特征类型统计
    feature_types = {
        '移动平均': len([f for f in actual_enhanced_features if 'ma_' in f or 'price_vs_ma' in f]),
        '动量': len([f for f in actual_enhanced_features if 'mom_' in f]),
        '波动率': len([f for f in actual_enhanced_features if 'volatility_' in f or 'atr_' in f]),
        '价格位置': len([f for f in actual_enhanced_features if 'position_' in f or 'intensity' in f]),
        '成交量': len([f for f in actual_enhanced_features if 'volume_' in f]),
        'RSI': len([f for f in actual_enhanced_features if 'rsi_' in f]),
        '价格变化': len([f for f in actual_enhanced_features if 'change_' in f or 'pct_change' in f])
    }

    print("生成特征类型统计:")
    for feature_type, count in feature_types.items():
        if count > 0:
            print(f"  {feature_type}: {count}个")

    return df_enhanced

@timer_decorator
def prepare_modeling_data(df, feature_cols):
    """准备建模数据"""
    print_section("准备建模数据")

    if df.empty or len(feature_cols) == 0:
        print("数据为空或无特征")
        return pd.DataFrame()

    # 基础列
    base_cols = ['date', 'stock_code', 'close', 'volume', 'future_return', 'market_avg_return', 'label']

    # 添加open, high, low如果存在
    for col in ['open', 'high', 'low', 'spread', 'turnover_rate', 'change', 'amount']:
        if col in df.columns and col not in base_cols:
            base_cols.append(col)

    # 合并所有需要的列
    all_cols = base_cols + feature_cols
    all_cols = [col for col in all_cols if col in df.columns]

    modeling_df = df[all_cols].copy()

    # 处理缺失值
    print(f"处理前数据形状: {modeling_df.shape}")

    # 移除标签缺失的行
    initial_size = len(modeling_df)
    modeling_df = modeling_df.dropna(subset=['future_return', 'market_avg_return', 'label'])
    print(f"移除标签缺失行: {initial_size - len(modeling_df):,} 行")

    # 处理特征缺失值
    for col in feature_cols:
        if col in modeling_df.columns:
            modeling_df[col] = modeling_df[col].fillna(modeling_df[col].median())

    # 处理无穷值
    numeric_cols = modeling_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in modeling_df.columns:
            modeling_df[col] = modeling_df[col].replace([np.inf, -np.inf], np.nan)
            modeling_df[col] = modeling_df[col].fillna(modeling_df[col].median())

    # 移除仍有缺失值的行
    modeling_df = modeling_df.dropna()
    print(f"处理后数据形状: {modeling_df.shape}")

    print(f"建模数据准备完成!")
    print(f"特征数量: {len(feature_cols)}")
    print(f"正样本比例: {modeling_df['label'].mean():.2%}")
    print(f"时间范围: {modeling_df['date'].min()} 到 {modeling_df['date'].max()}")
    print(f"股票数量: {modeling_df['stock_code'].nunique()}")

    return modeling_df


@timer_decorator
def split_train_val_test_data(df, feature_cols, test_ratio=0.2, val_ratio=0.1):
    """时间序列数据集划分"""
    print_section("数据集划分")

    if df.empty or len(feature_cols) == 0:
        print("数据为空或无特征")
        return None, None, None, None, None, None, None, None, None

    # 确保按日期排序
    df = df.sort_values('date')

    # 获取唯一日期
    dates = np.sort(df['date'].unique())
    n_dates = len(dates)

    # 计算分割点
    train_end_idx = int(n_dates * (1 - test_ratio - val_ratio))
    val_end_idx = int(n_dates * (1 - test_ratio))

    train_dates = dates[:train_end_idx]
    val_dates = dates[train_end_idx:val_end_idx]
    test_dates = dates[val_end_idx:]

    # 划分数据集
    train_df = df[df['date'].isin(train_dates)]
    val_df = df[df['date'].isin(val_dates)]
    test_df = df[df['date'].isin(test_dates)]

    # 检查数据集是否为空
    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        print("数据集划分失败，使用简单划分")
        # 回退到简单划分
        train_idx = int(len(df) * (1 - test_ratio - val_ratio))
        val_idx = int(len(df) * (1 - test_ratio))

        train_df = df.iloc[:train_idx]
        val_df = df.iloc[train_idx:val_idx]
        test_df = df.iloc[val_idx:]

    print(f"训练集: {train_df['date'].min().date()} 到 {train_df['date'].max().date()}, 大小: {len(train_df):,}")
    print(f"验证集: {val_df['date'].min().date()} 到 {val_df['date'].max().date()}, 大小: {len(val_df):,}")
    print(f"测试集: {test_df['date'].min().date()} 到 {test_df['date'].max().date()}, 大小: {len(test_df):,}")
    print(f"训练集正样本比例: {train_df['label'].mean():.2%}")
    print(f"验证集正样本比例: {val_df['label'].mean():.2%}")
    print(f"测试集正样本比例: {test_df['label'].mean():.2%}")

    # 准备特征和标签
    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]
    X_test = test_df[feature_cols]

    y_train = train_df['label']
    y_val = val_df['label']
    y_test = test_df['label']

    print(f"特征形状: X_train{X_train.shape}, X_val{X_val.shape}, X_test{X_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test, train_df, val_df, test_df


@timer_decorator
def hyperparameter_tuning(X_train, y_train, X_val, y_val, n_trials=5):
    """验证集超参数调优"""
    # 快速模式：减少调优次数
    if QUICK_MODE:
        n_trials = HYPERPARAM_TRIALS
        print_section(f"快速超参数调优 (n_trials={n_trials})")
    else:
        print_section("验证集超参数调优")

    # 如果数据量大，进行采样以加速调优
    if len(X_train) > SAMPLE_SIZE_TUNING:
        from sklearn.model_selection import train_test_split
        X_train_sample, _, y_train_sample, _ = train_test_split(
            X_train, y_train,
            train_size=SAMPLE_SIZE_TUNING,
            stratify=y_train,  # 保持正负样本比例
            random_state=RANDOM_STATE
        )

    best_params = {}

    # 1. 随机森林调参（简化参数网格）
    print("1. 随机森林超参数调优...")
    rf_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 5],
        'max_features': ['sqrt']
    }

    rf_model = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS
    )

    # 使用随机搜索
    rf_search = RandomizedSearchCV(
        rf_model,
        rf_param_grid,
        n_iter=n_trials,
        cv=2,
        scoring='f1',
        n_jobs=-1,
        verbose=1,
        random_state=RANDOM_STATE
    )

    rf_search.fit(X_train_sample, y_train_sample)
    best_params['rf'] = rf_search.best_params_
    print(f" 最佳参数: {rf_search.best_params_}")
    print(f" 最佳验证分数: {rf_search.best_score_:.4f}")

    # 2. XGBoost调参（简化参数网格）- 修复：转换为numpy数组
    print("\n2. XGBoost超参数调优...")

    if hasattr(X_train, 'values'):
        X_train_sample = X_train.values
    else:
        X_train_sample = X_train

    if hasattr(y_train, 'values'):
        y_train_sample = y_train.values
    else:
        y_train_sample = y_train

    # 修复：转换数据类型
    X_train_sample = X_train_sample.astype(np.float32)
    y_train_sample = y_train_sample.astype(np.int32)

    xgb_param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.05, 0.1, 0.15],
        'subsample': [0.6, 0.7, 0.8],
        'colsample_bytree': [0.6, 0.7, 0.8]
    }

    # 修复：使用更兼容的XGBoost参数
    xgb_model = xgb.XGBClassifier(
        random_state=RANDOM_STATE,
        n_jobs=1,  # 避免并行问题
        use_label_encoder=False,
        eval_metric='logloss',
        verbosity=0  # 减少输出
    )

    try:
        xgb_search = RandomizedSearchCV(
            xgb_model, xgb_param_grid,
            n_iter=n_trials, cv=2, scoring='f1', n_jobs=1,
            verbose=1, random_state=RANDOM_STATE, error_score='raise'
        )
        xgb_search.fit(X_train_sample, y_train_sample)  # 直接使用原始数据
        best_params['xgb'] = xgb_search.best_params_
        print(f"   最佳参数: {xgb_search.best_params_}")
        print(f"   最佳验证分数: {xgb_search.best_score_:.4f}")

    except Exception as e:
        print(f"  XGBoost调优失败: {e}")
        print("   使用默认XGBoost参数")
        best_params['xgb'] = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8
        }

    return best_params


@timer_decorator
def train_models(X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, best_params=None):
    """修复版模型训练 - 解决XGBoost dtype错误和收益率inf问题"""
    print_section("修复版模型训练")

    # ==================== 1. 数据验证和特征数量检查 ====================
    print("数据验证和特征数量检查...")

    if X_train.empty or X_val.empty or X_test.empty:
        print("输入数据为空")
        return {}, None, {}, {}, {}

    # 验证特征数量一致性
    print(f"特征数量验证:")
    print(f"  特征列表: {len(feature_cols)} 个特征")
    print(f"  X_train 形状: {X_train.shape} -> {X_train.shape[1]} 个特征")
    print(f"  X_val 形状: {X_val.shape} -> {X_val.shape[1]} 个特征")
    print(f"  X_test 形状: {X_test.shape} -> {X_test.shape[1]} 个特征")

    # 检查特征数量是否匹配
    if len(feature_cols) != X_train.shape[1]:
        print(f"特征数量不匹配: 特征列表{len(feature_cols)} vs 训练数据{X_train.shape[1]}")
        if hasattr(X_train, 'columns'):
            actual_features = list(X_train.columns)
            print(f"  使用实际特征名称: {len(actual_features)} 个")
            feature_cols = actual_features
        else:
            feature_cols = [f'feature_{i}' for i in range(X_train.shape[1])]
            print(f"  创建新特征名称: {len(feature_cols)} 个")

    # ==================== 2. 标准化特征 ====================
    print("特征标准化...")
    scaler = StandardScaler()

    try:
        # 确保数据是numpy数组格式
        X_train_array = X_train.values if hasattr(X_train, 'values') else X_train
        X_val_array = X_val.values if hasattr(X_val, 'values') else X_val
        X_test_array = X_test.values if hasattr(X_test, 'values') else X_test

        X_train_scaled = scaler.fit_transform(X_train_array)
        X_val_scaled = scaler.transform(X_val_array)
        X_test_scaled = scaler.transform(X_test_array)

        print(f"特征标准化完成")
        print(f"标准化后形状: X_train{X_train_scaled.shape}, X_val{X_val_scaled.shape}, X_test{X_test_scaled.shape}")
    except Exception as e:
        print(f"特征标准化失败: {e}")
        X_train_scaled = X_train_array
        X_val_scaled = X_val_array
        X_test_scaled = X_test_array
        print("使用未标准化数据继续训练")

    # ==================== 3. 处理类别不平衡 ====================
    print("处理类别不平衡...")
    try:
        smote = SMOTE(random_state=RANDOM_STATE, sampling_strategy=0.8)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
        print(f"平衡后训练集: {X_train_balanced.shape}")

        # 确保y_train_balanced是正确格式
        if hasattr(y_train_balanced, 'values'):
            y_train_balanced = y_train_balanced.values
        elif hasattr(y_train_balanced, 'to_numpy'):
            y_train_balanced = y_train_balanced.to_numpy()

        # 修复：确保数据类型一致
        if hasattr(X_train_balanced, 'dtype') and X_train_balanced.dtype != np.float32:
            X_train_balanced = X_train_balanced.astype(np.float32)
        if hasattr(y_train_balanced, 'dtype') and y_train_balanced.dtype != np.int32:
            y_train_balanced = y_train_balanced.astype(np.int32)

    except Exception as e:
        print(f"SMOTE处理失败: {e}")
        print("使用原始不平衡数据")
        X_train_balanced, y_train_balanced = X_train_scaled, y_train
        if hasattr(y_train_balanced, 'values'):
            y_train_balanced = y_train_balanced.values
        elif hasattr(y_train_balanced, 'to_numpy'):
            y_train_balanced = y_train_balanced.to_numpy()

    # ==================== 4. 初始化结果字典 ====================
    models = {}
    results = {}
    predictions = {}
    probabilities = {}

    # ==================== 5. 模型默认参数 ====================
    if best_params is None:
        print("使用保守模型参数防止过拟合...")
        best_params = {
            'rf': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 10,
                'min_samples_leaf': 4,
                'max_features': 'sqrt',
                'random_state': RANDOM_STATE,
                'n_jobs': N_JOBS
            },
            'xgb': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': RANDOM_STATE,
                'n_jobs': 1,
                'use_label_encoder': False,
                'eval_metric': 'logloss'
            }
        }

    # ==================== 6. 训练随机森林模型 ====================
    print("\n1. 训练随机森林模型...")
    try:
        rf_params = best_params.get('rf', {})
        rf_model = RandomForestClassifier(**rf_params)

        print(f"训练数据格式检查:")
        print(f"  X_train_balanced: {type(X_train_balanced)}, shape: {X_train_balanced.shape}")
        print(f"  y_train_balanced: {type(y_train_balanced)}, shape: {y_train_balanced.shape}")

        # 添加样本权重
        if len(X_train_balanced) > 1000:
            try:
                print("计算样本权重...")
                time_decay = np.linspace(0.8, 1.2, len(X_train_balanced))
                class_balance = 1 + (y_train_balanced * 0.2)
                sample_weights = time_decay * class_balance

                print(f"使用样本权重训练")
                print(f"权重范围: {sample_weights.min():.2f} - {sample_weights.max():.2f}")
                rf_model.fit(X_train_balanced, y_train_balanced, sample_weight=sample_weights)
            except Exception as e:
                print(f"样本权重训练失败: {e}")
                print("回退到无权重训练")
                rf_model.fit(X_train_balanced, y_train_balanced)
        else:
            rf_model.fit(X_train_balanced, y_train_balanced)

        models['rf'] = rf_model
        print("随机森林模型训练完成")

        # 在验证集和测试集上评估
        y_val_pred_rf = rf_model.predict(X_val_scaled)
        y_val_proba_rf = rf_model.predict_proba(X_val_scaled)[:, 1]
        y_test_pred_rf = rf_model.predict(X_test_scaled)
        y_test_proba_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

        # 确保y_true是numpy数组格式
        y_val_array = y_val.values if hasattr(y_val, 'values') else y_val
        y_test_array = y_test.values if hasattr(y_test, 'values') else y_test

        results['rf'] = {
            'val_accuracy': accuracy_score(y_val_array, y_val_pred_rf),
            'val_precision': precision_score(y_val_array, y_val_pred_rf, zero_division=0),
            'val_recall': recall_score(y_val_array, y_val_pred_rf, zero_division=0),
            'val_f1': f1_score(y_val_array, y_val_pred_rf, zero_division=0),
            'val_roc_auc': roc_auc_score(y_val_array, y_val_proba_rf),
            'test_accuracy': accuracy_score(y_test_array, y_test_pred_rf),
            'test_precision': precision_score(y_test_array, y_test_pred_rf, zero_division=0),
            'test_recall': recall_score(y_test_array, y_test_pred_rf, zero_division=0),
            'test_f1': f1_score(y_test_array, y_test_pred_rf, zero_division=0),
            'test_roc_auc': roc_auc_score(y_test_array, y_test_proba_rf)
        }

        predictions['rf'] = y_test_pred_rf
        probabilities['rf'] = y_test_proba_rf

        print("随机森林模型验证集结果:")
        print(f"  准确率: {results['rf']['val_accuracy']:.4f}")
        print(f"  精确率: {results['rf']['val_precision']:.4f}")
        print(f"  召回率: {results['rf']['val_recall']:.4f}")
        print(f"  F1分数: {results['rf']['val_f1']:.4f}")
        print(f"  ROC-AUC: {results['rf']['val_roc_auc']:.4f}")

        print("随机森林模型测试集结果:")
        print(f"  准确率: {results['rf']['test_accuracy']:.4f}")
        print(f"  精确率: {results['rf']['test_precision']:.4f}")
        print(f"  召回率: {results['rf']['test_recall']:.4f}")
        print(f"  F1分数: {results['rf']['test_f1']:.4f}")
        print(f"  ROC-AUC: {results['rf']['test_roc_auc']:.4f}")

    except Exception as e:
        print(f"随机森林模型训练失败: {e}")
        import traceback
        traceback.print_exc()

        # 设置默认结果
        y_test_array = y_test.values if hasattr(y_test, 'values') else y_test
        results['rf'] = {
            'val_accuracy': 0.5, 'val_precision': 0.5, 'val_recall': 0.5, 'val_f1': 0.5, 'val_roc_auc': 0.5,
            'test_accuracy': 0.5, 'test_precision': 0.5, 'test_recall': 0.5, 'test_f1': 0.5, 'test_roc_auc': 0.5
        }
        predictions['rf'] = np.zeros(len(y_test_array))
        probabilities['rf'] = np.ones(len(y_test_array)) * 0.5
        models['rf'] = None

    # ==================== 7. 训练XGBoost模型（关键修复部分） ====================
    print("\n2. 训练XGBoost模型...")
    try:
        xgb_params = best_params.get('xgb', {})

        if not xgb_params:
            # 计算正负样本比例
            scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1]) if len(
                y_train[y_train == 1]) > 0 else 1
            xgb_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'scale_pos_weight': scale_pos_weight,
                'random_state': RANDOM_STATE,
                'n_jobs': 1,
                'use_label_encoder': False,
                'eval_metric': 'logloss',
                'verbosity': 0
            }

        print(f"使用XGBoost参数:")
        for key, value in xgb_params.items():
            if key in ['n_estimators', 'max_depth', 'learning_rate', 'subsample',
                       'colsample_bytree', 'scale_pos_weight']:
                print(f"   {key}: {value}")

        # 关键修复：XGBoost数据格式兼容性
        print("准备XGBoost训练数据...")

        def safe_convert_to_float32(data):
            """安全转换为float32，兼容DataFrame和numpy数组"""
            if hasattr(data, 'values'):
                # 如果是DataFrame或Series，获取values
                array_data = data.values
            else:
                array_data = data

            # 确保是numpy数组
            if not isinstance(array_data, np.ndarray):
                array_data = np.array(array_data)

            # 检查数据类型并安全转换
            try:
                if hasattr(array_data, 'dtype'):
                    if array_data.dtype != np.float32:
                        return array_data.astype(np.float32)
                return array_data
            except Exception as e:
                print(f"数据类型转换失败: {e}，使用原数据类型")
                return array_data

        # 应用安全转换
        X_train_balanced_float32 = safe_convert_to_float32(X_train_balanced)
        y_train_balanced_int32 = y_train_balanced.astype(np.int32) if hasattr(y_train_balanced, 'astype') else np.array(
            y_train_balanced, dtype=np.int32)
        X_val_float32 = safe_convert_to_float32(X_val_scaled)
        X_test_float32 = safe_convert_to_float32(X_test_scaled)

        print(f"数据格式检查:")
        print(f"  X_train_balanced: {type(X_train_balanced_float32)}")
        if hasattr(X_train_balanced_float32, 'dtype'):
            print(f"    dtype: {X_train_balanced_float32.dtype}")
        print(f"  y_train_balanced: {type(y_train_balanced_int32)}")
        if hasattr(y_train_balanced_int32, 'dtype'):
            print(f"    dtype: {y_train_balanced_int32.dtype}")
        print(f"  X_val: {type(X_val_float32)}")
        if hasattr(X_val_float32, 'dtype'):
            print(f"    dtype: {X_val_float32.dtype}")
        print(f"  X_test: {type(X_test_float32)}")
        if hasattr(X_test_float32, 'dtype'):
            print(f"    dtype: {X_test_float32.dtype}")

        # 创建XGBoost模型
        xgb_model = xgb.XGBClassifier(**xgb_params)

        # 训练模型
        print("训练XGBoost模型...")
        xgb_model.fit(X_train_balanced_float32, y_train_balanced_int32)
        models['xgb'] = xgb_model
        print(" XGBoost模型训练完成")

        # 在验证集和测试集上评估
        y_val_pred_xgb = xgb_model.predict(X_val_float32)
        y_val_proba_xgb = xgb_model.predict_proba(X_val_float32)[:, 1]
        y_test_pred_xgb = xgb_model.predict(X_test_float32)
        y_test_proba_xgb = xgb_model.predict_proba(X_test_float32)[:, 1]

        # 确保y_true格式正确
        y_val_array = y_val.values if hasattr(y_val, 'values') else y_val
        y_test_array = y_test.values if hasattr(y_test, 'values') else y_test

        results['xgb'] = {
            'val_accuracy': accuracy_score(y_val_array, y_val_pred_xgb),
            'val_precision': precision_score(y_val_array, y_val_pred_xgb, zero_division=0),
            'val_recall': recall_score(y_val_array, y_val_pred_xgb, zero_division=0),
            'val_f1': f1_score(y_val_array, y_val_pred_xgb, zero_division=0),
            'val_roc_auc': roc_auc_score(y_val_array, y_val_proba_xgb),
            'test_accuracy': accuracy_score(y_test_array, y_test_pred_xgb),
            'test_precision': precision_score(y_test_array, y_test_pred_xgb, zero_division=0),
            'test_recall': recall_score(y_test_array, y_test_pred_xgb, zero_division=0),
            'test_f1': f1_score(y_test_array, y_test_pred_xgb, zero_division=0),
            'test_roc_auc': roc_auc_score(y_test_array, y_test_proba_xgb)
        }

        predictions['xgb'] = y_test_pred_xgb
        probabilities['xgb'] = y_test_proba_xgb

        print("XGBoost模型验证集结果:")
        print(f"  准确率: {results['xgb']['val_accuracy']:.4f}")
        print(f"  精确率: {results['xgb']['val_precision']:.4f}")
        print(f"  召回率: {results['xgb']['val_recall']:.4f}")
        print(f"  F1分数: {results['xgb']['val_f1']:.4f}")
        print(f"  ROC-AUC: {results['xgb']['val_roc_auc']:.4f}")

        print("XGBoost模型测试集结果:")
        print(f"  准确率: {results['xgb']['test_accuracy']:.4f}")
        print(f"  精确率: {results['xgb']['test_precision']:.4f}")
        print(f"  召回率: {results['xgb']['test_recall']:.4f}")
        print(f"  F1分数: {results['xgb']['test_f1']:.4f}")
        print(f"  ROC-AUC: {results['xgb']['test_roc_auc']:.4f}")

    except Exception as e:
        print(f" XGBoost模型训练失败: {e}")
        import traceback
        traceback.print_exc()

        print("跳过XGBoost模型，仅使用随机森林")
        # 设置默认结果
        y_test_array = y_test.values if hasattr(y_test, 'values') else y_test
        results['xgb'] = {
            'val_accuracy': 0.5, 'val_precision': 0.5, 'val_recall': 0.5, 'val_f1': 0.5, 'val_roc_auc': 0.5,
            'test_accuracy': 0.5, 'test_precision': 0.5, 'test_recall': 0.5, 'test_f1': 0.5, 'test_roc_auc': 0.5
        }
        predictions['xgb'] = np.zeros(len(y_test_array))
        probabilities['xgb'] = np.ones(len(y_test_array)) * 0.5
        models['xgb'] = None

    # ==================== 8. 最终结果统计 ====================
    print_section("模型训练完成")

    # 统计成功训练的模型
    successful_models = [name for name, model in models.items() if model is not None]
    print(f"成功训练的模型: {len(successful_models)}/{len(models)}")

    for model_name in successful_models:
        test_f1 = results[model_name]['test_f1']
        test_auc = results[model_name]['test_roc_auc']
        print(f"  {model_name.upper()}: F1={test_f1:.4f}, AUC={test_auc:.4f}")

    # 检查是否有可用的模型
    if not any(models.values()):
        print("所有模型训练失败!")
        return {}, None, {}, {}, {}

    return models, scaler, results, predictions, probabilities

@timer_decorator
def analyze_feature_importance(models, feature_cols, n_top=20):
    """分析特征重要性 - 修复版本"""
    print_section("特征重要性分析")

    # 创建特征重要性DataFrame
    feature_importance = pd.DataFrame({'feature': feature_cols})

    for model_name, model in models.items():
        if model is not None and hasattr(model, 'feature_importances_'):
            try:
                importances = model.feature_importances_

                # 修复：获取模型实际使用的特征名称
                if hasattr(model, 'feature_names_in_'):
                    # 使用模型训练时的特征名称
                    model_features = list(model.feature_names_in_)
                else:
                    # 回退到传入的特征列表
                    model_features = feature_cols

                # 确保特征数量匹配
                if len(importances) == len(model_features):
                    # 创建临时DataFrame来匹配特征
                    temp_importance = pd.DataFrame({
                        'feature': model_features,
                        f'importance_{model_name}': importances
                    })
                    # 合并到主DataFrame
                    feature_importance = feature_importance.merge(
                        temp_importance, on='feature', how='left'
                    )
                else:
                    print(f"特征数量不匹配: 模型{model_name}")
                    # 使用对齐的逻辑
                    min_len = min(len(importances), len(feature_cols))
                    importance_series = np.zeros(len(feature_cols))
                    importance_series[:min_len] = importances[:min_len]
                    feature_importance[f'importance_{model_name}'] = importance_series

            except Exception as e:
                print(f"模型 {model_name} 特征重要性计算失败: {e}")
                feature_importance[f'importance_{model_name}'] = 0.0

    # 计算平均重要性
    importance_cols = [col for col in feature_importance.columns if col.startswith('importance_')]
    if importance_cols:
        feature_importance['importance_mean'] = feature_importance[importance_cols].mean(axis=1)
        feature_importance = feature_importance.sort_values('importance_mean', ascending=False)

    print(f"Top {n_top} 重要特征:")
    if len(feature_importance) > 0:
        print(feature_importance.head(min(n_top, len(feature_importance))).to_string(index=False))

        # 显示特征类型统计
        tech_features = len([col for col in feature_cols if not col.startswith('fin_')])
        fin_features = len([col for col in feature_cols if col.startswith('fin_')])
        print(f"特征类型统计: 技术特征={tech_features}, 财务特征={fin_features}")
    else:
        print("没有特征重要性数据")

    return feature_importance


def generate_daily_selected_stocks(test_df, predictions, probabilities, top_n=10):
    """生成每日选股列表 - 修复版本（删除收益率计算）"""
    print_section("生成每日选股列表")

    if test_df.empty or not predictions:
        print("测试数据或预测结果为空")
        return pd.DataFrame()

    try:
        # ==================== 1. 数据准备和验证 ====================
        print("数据准备和验证...")

        # 复制测试集数据
        required_cols = ['date', 'stock_code', 'close', 'future_return']
        missing_cols = [col for col in required_cols if col not in test_df.columns]
        if missing_cols:
            print(f"缺少必要列: {missing_cols}")
            return pd.DataFrame()

        selected_stocks = test_df[required_cols].copy()

        # 验证数据完整性
        initial_count = len(selected_stocks)
        selected_stocks = selected_stocks.dropna(subset=['future_return'])
        print(f"移除未来收益率缺失的数据: {initial_count - len(selected_stocks):,} 行")

        if selected_stocks.empty:
            print("选股数据为空")
            return pd.DataFrame()

        # 添加模型预测概率
        for model_name in predictions.keys():
            if model_name in predictions and len(predictions[model_name]) == len(selected_stocks):
                selected_stocks[f'{model_name}_prediction'] = predictions[model_name]
                selected_stocks[f'{model_name}_probability'] = probabilities[model_name]
            else:
                print(f"模型 {model_name} 预测结果长度不匹配，跳过")

        # 使用第一个可用的模型进行选股
        available_models = [m for m in predictions.keys() if f'{m}_probability' in selected_stocks.columns]
        if available_models:
            best_model = available_models[0]
        else:
            best_model = 'rf'
            # 如果没有模型概率，使用随机分数
            selected_stocks['selection_score'] = np.random.random(len(selected_stocks))
            print("无可用模型概率，使用随机选股")

        print(f"使用模型进行选股: {best_model.upper()}")
        selected_stocks['selection_score'] = selected_stocks[f'{best_model}_probability']

        # ==================== 2. 修复选股逻辑 ====================
        print("生成每日选股列表...")
        daily_top_stocks = []
        valid_dates = 0

        # 获取唯一日期并排序
        unique_dates = sorted(selected_stocks['date'].unique())
        print(f"处理 {len(unique_dates)} 个交易日的选股...")

        for date in tqdm(unique_dates, desc="生成每日选股"):
            date_data = selected_stocks[selected_stocks['date'] == date].copy()

            if len(date_data) == 0:
                continue

            # 按预测概率排序
            date_data = date_data.sort_values('selection_score', ascending=False)
            date_data = date_data.drop_duplicates(subset=['stock_code'], keep='first')

            # 修复：确保有足够的股票可选
            if len(date_data) < top_n:
                if len(date_data) > 0:
                    # 使用所有可用股票
                    top_n_stocks = date_data.copy()
                    print(f"日期 {date.date()} 只有 {len(date_data)} 只股票，使用全部可用股票")
                else:
                    print(f"日期 {date.date()} 没有可用股票，跳过")
                    continue
            else:
                # 选择Top N
                top_n_stocks = date_data.head(top_n).copy()

            # 确保有选股结果
            if len(top_n_stocks) == 0:
                print(f"日期 {date.date()} 选股结果为空，使用随机选择")
                # 回退：随机选择top_n只股票
                if len(date_data) > 0:
                    top_n_stocks = date_data.sample(n=min(top_n, len(date_data)),
                                                    random_state=RANDOM_STATE)
                else:
                    continue

            top_n_stocks['rank'] = range(1, len(top_n_stocks) + 1)
            daily_top_stocks.append(top_n_stocks)
            valid_dates += 1

        print(f"成功处理 {valid_dates}/{len(unique_dates)} 个交易日的选股")

        if not daily_top_stocks:
            print("没有生成任何选股列表")
            return pd.DataFrame()

        # ==================== 3. 合并结果 ====================
        result_df = pd.concat(daily_top_stocks, ignore_index=True)

        # 添加选股理由
        result_df['selection_reason'] = result_df.apply(
            lambda
                x: f"模型预测概率:{x['selection_score']:.3f}, 排名:{x['rank']}/{min(top_n, len(result_df[result_df['date'] == x['date']]))}",
            axis=1
        )

        # 重命名列
        result_df = result_df.rename(columns={
            'date': '交易日',
            'stock_code': '股票代码',
            'close': '收盘价',
            'future_return': '未来15天绝对收益率',
            'selection_score': '模型预测概率',
            'rank': '当日排名',
            'selection_reason': '选股理由'
        })

        # 选择需要的列
        final_columns = ['交易日', '股票代码', '收盘价', '未来15天绝对收益率',
                         '模型预测概率', '当日排名', '选股理由']
        final_columns = [col for col in final_columns if col in result_df.columns]
        result_df = result_df[final_columns]

        print(f"生成每日选股列表: {result_df.shape}")

        # ==================== 4. 简单的选股统计（删除收益率计算） ====================
        print_section("选股结果统计")

        # 简单的统计（不涉及复杂收益率计算）
        total_stocks = len(result_df)
        unique_stocks = result_df['股票代码'].nunique()
        avg_daily_stocks = result_df.groupby('交易日').size().mean()
        avg_prob_all = result_df['模型预测概率'].mean()

        print(f"选股统计:")
        print(f"   总选股记录: {total_stocks:,} 条")
        print(f"   唯一股票数量: {unique_stocks} 只")
        print(f"   平均每日选股: {avg_daily_stocks:.1f} 只")
        print(f"   平均预测概率: {avg_prob_all:.3f}")

        # ==================== 5. 验证选股结果 ====================
        print_section("选股结果验证")

        # 检查最近几个交易日的选股结果
        recent_dates = result_df['交易日'].unique()[-3:]  # 最近3个交易日
        for test_date in recent_dates:
            daily_selection = result_df[result_df['交易日'] == test_date]
            print(f"验证 {test_date.date()} 的选股结果:")
            print(f"   选股数量: {len(daily_selection)} 只")
            print(f"   唯一股票: {len(daily_selection['股票代码'].unique())} 只")
            if len(daily_selection) > 0:
                top_stocks = daily_selection['股票代码'].head(3).tolist()
                avg_prob = daily_selection['模型预测概率'].mean()
                print(f"   前3只股票: {top_stocks}")
                print(f"   平均预测概率: {avg_prob:.3f}")
                # 删除收益率计算，只显示基本信息
            else:
                print(" 该日无选股结果")

        return result_df

    except Exception as e:
        print(f"生成每日选股列表失败: {e}")
        traceback.print_exc()
        return pd.DataFrame()



def emergency_recalculate_returns(df, days=FUTURE_DAYS):
    """紧急重新计算收益率 - 简化版本"""
    print("执行紧急收益率重新计算...")

    df = df.copy().sort_values(['stock_code', 'date'])
    returns = np.full(len(df), np.nan)

    # 按股票分组计算
    for stock_code in df['stock_code'].unique():
        stock_data = df[df['stock_code'] == stock_code].sort_values('date')
        close_prices = stock_data['close'].values

        for i in range(len(stock_data)):
            if i + days < len(stock_data):
                current_price = close_prices[i]
                future_price = close_prices[i + days]

                # 检查价格有效性
                if current_price > 0 and future_price > 0 and not np.isnan(current_price) and not np.isnan(
                        future_price):
                    return_val = (future_price / current_price) - 1
                    # 找到在原始df中的索引
                    original_idx = stock_data.index[i]
                    returns[df.index.get_loc(original_idx)] = return_val

    df['future_return'] = returns

    # 统计结果
    valid_returns = returns[~np.isnan(returns)]
    if len(valid_returns) > 0:
        print(f"紧急计算完成: {len(valid_returns):,} 个有效收益率")
        print(f"  收益率范围: {valid_returns.min():.4f} 到 {valid_returns.max():.4f}")

        # 显示几个样本
        sample_count = min(3, len(valid_returns))
        sample_indices = np.random.choice(len(valid_returns), sample_count, replace=False)
        for i, idx in enumerate(sample_indices):
            print(f"  样本{i + 1}: {valid_returns[idx]:.4f} ({valid_returns[idx]:.2%})")
    else:
        print("紧急计算失败：没有生成有效收益率")

    return df
# ==================== 主程序 ====================
def main():
    """主程序 - 添加紧急收益率修复版本"""
    print_section("台湾股票选股预测模型")
    print(f"预测未来天数: {FUTURE_DAYS}天")
    print(f"回看天数: {LOOKBACK_DAYS}天")
    print(f"随机种子: {RANDOM_STATE}")
    print(f"快速模式: {'启用' if QUICK_MODE else '关闭'}")

    # 时间预估
    print("\n预计执行时间:")
    if QUICK_MODE:
        print("  财报数据合并: 1-2分钟 (原10-20分钟)")
        print("  超参数调优: 2-3分钟 (原20-30分钟)")
        print("  总时间: 10-15分钟 (原60-90分钟)")
    else:
        print("  总时间: 30-45分钟")
    print("=" * 50)

    start_time = time.time()

    try:
        # 1. 加载和预处理数据
        data = load_and_preprocess_data()
        if data is None:
            print("数据加载失败")
            return None

        df, feature_cols = data
        if df is None or df.empty:
            print("数据为空")
            return None

        # ====================  紧急收益率修复 ====================
        print_section("执行紧急收益率修复")

        # 检查当前收益率状态
        if 'future_return' in df.columns:
            current_returns = df['future_return'].dropna()
            inf_count = np.isinf(current_returns).sum()
            print(f"当前收益率状态: 有效样本{len(current_returns):,}, inf值{inf_count}个")

            if inf_count > 0 or current_returns.mean() == float('inf'):
                print("检测到收益率问题，执行紧急修复...")
                df = emergency_fix_returns_simple(df)
            else:
                print("收益率数据正常，跳过修复")
        else:
            print("数据中没有future_return列，需要重新计算收益率")
            # 调用修复版的收益率计算函数
            df = calculate_future_returns_and_labels(df)

        # 验证修复结果
        if 'future_return' in df.columns:
            fixed_returns = df['future_return'].dropna()
            inf_count_fixed = np.isinf(fixed_returns).sum()
            print(f"修复后收益率状态: 有效样本{len(fixed_returns):,}, inf值{inf_count_fixed}个")

            if inf_count_fixed > 0:
                print("紧急修复后仍然存在inf值，进行二次修复...")
                # 强制重新计算
                df = calculate_future_returns_and_labels(df)

        # ==================== 后续原有代码 ====================
        # 2. 准备建模数据
        modeling_df = prepare_modeling_data(df, feature_cols)
        if modeling_df.empty or len(feature_cols) < 5:
            print("建模数据为空或特征数量不足")
            return None

        # 3. 数据集划分
        data_split = split_train_val_test_data(
            modeling_df, feature_cols, test_ratio=TEST_RATIO, val_ratio=VAL_RATIO
        )
        if data_split[0] is None:
            print("数据集划分失败")
            return None

        X_train, X_val, X_test, y_train, y_val, y_test, train_df, val_df, test_df = data_split

        if X_train.empty or X_test.empty or X_val.empty:
            print("数据集划分失败")
            return None

        # 4. 验证集超参数调优
        best_params = hyperparameter_tuning(X_train, y_train, X_val, y_val, n_trials=5)

        # 5. 使用调优参数训练模型
        models, scaler, results, predictions, probabilities = train_models(
            X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, best_params
        )

        # 6. 特征重要性分析
        try:
            feature_importance = analyze_feature_importance(models, feature_cols)
        except Exception as e:
            print(f"特征重要性分析失败: {e}")
            print("跳过特征重要性分析...")
            feature_importance = None

        # 7. 创建结果DataFrame
        results_df = test_df[['date', 'stock_code', 'close']].copy()
        results_df = results_df.rename(columns={
            'date': '交易日',
            'stock_code': '股票代码',
            'close': '收盘价'
        })

        # 添加预测结果
        for model_name in models.keys():
            results_df[f'{model_name}_预测'] = predictions[model_name]
            results_df[f'{model_name}_概率'] = probabilities[model_name]

        results_df['说明'] = '基于历史数据和财务指标的机器学习选股预测'

        print("输出表格字段说明:")
        print("1. 交易日 - 预测基准日（如 2025-01-05）")
        print("2. 股票代码 - 个股唯一标识（如 2344）")
        print("3. 收盘价 - 预测日收盘价格")
        print("4. 模型预测 - 各模型预测结果（1=看涨/0=看跌）")
        print("5. 模型概率 - 模型预测的置信度（0-1）")

        # 保存预测结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f'stock_predictions_{timestamp}.csv'
        results_df.to_csv(results_file, index=False, encoding='utf-8-sig')
        print(f"预测结果已保存: {results_file}")

        # 保存特征重要性
        if feature_importance is not None:
            importance_file = f'feature_importance_{timestamp}.csv'
            feature_importance.to_csv(importance_file, index=False)
            print(f"特征重要性已保存: {importance_file}")

        # 保存模型
        model_file = f'stock_models_{timestamp}.pkl'
        with open(model_file, 'wb') as f:
            pickle.dump({
                'models': models,
                'scaler': scaler,
                'features': feature_cols,
                'best_params': best_params,
                'results': results
            }, f, protocol=4)
        print(f"模型已保存: {model_file}")

        # 保存评估结果
        eval_file = f'model_evaluation_{timestamp}.txt'
        with open(eval_file, 'w', encoding='utf-8') as f:
            f.write("台湾股票选股预测模型评估结果\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"数据时间范围: {df['date'].min().date()} 到 {df['date'].max().date()}\n")
            f.write(f"股票数量: {df['stock_code'].nunique()}\n")
            f.write(f"总样本数: {len(df):,}\n")
            f.write(f"特征数量: {len(feature_cols)}\n")
            f.write(f"训练集样本: {len(train_df):,}\n")
            f.write(f"验证集样本: {len(val_df):,}\n")
            f.write(f"测试集样本: {len(test_df):,}\n\n")

            f.write("模型性能比较:\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'模型':<10} {'准确率':<8} {'精确率':<8} {'召回率':<8} {'F1分数':<8} {'ROC-AUC':<8}\n")
            f.write("-" * 60 + "\n")
            for model_name, result in results.items():
                f.write(f"{model_name.upper():<10} {result['test_accuracy']:.4f}   {result['test_precision']:.4f}   "
                        f"{result['test_recall']:.4f}   {result['test_f1']:.4f}   {result['test_roc_auc']:.4f}\n")
            f.write("=" * 60 + "\n")
        print(f"评估结果已保存: {eval_file}")

        # 8. 生成每日选股列表
        print_section("生成每日选股列表")
        daily_selected_df = generate_daily_selected_stocks(test_df, predictions, probabilities, top_n=10)

        if not daily_selected_df.empty:
            selected_stocks_file = f'daily_selected_stocks_top10_{timestamp}.csv'
            daily_selected_df.to_csv(selected_stocks_file, index=False, encoding='utf-8-sig')
            print(f"每日选股列表已保存: {selected_stocks_file}")
        else:
            print("每日选股列表生成失败")

        # 9. 最终报告
        end_time = time.time()
        execution_time = (end_time - start_time) / 60

        print_section("最终模型评估报告")

        # 找出最佳模型
        best_model_name = max(results.keys(), key=lambda k: results[k]['test_f1'])
        best_f1 = results[best_model_name]['test_f1']
        best_roc_auc = results[best_model_name]['test_roc_auc']

        print(f"最佳模型: {best_model_name.upper()} (F1: {best_f1:.4f}, ROC-AUC: {best_roc_auc:.4f})")
        print(f"数据时间范围: {df['date'].min().date()} 到 {df['date'].max().date()}")
        print(f"股票数量: {df['stock_code'].nunique()}")
        print(f"总样本数: {len(df):,}")
        print(f"特征数量: {len(feature_cols)}")
        print(f"技术特征: {len([col for col in feature_cols if not col.startswith('fin_')])}")
        print(f"财务特征: {len([col for col in feature_cols if col.startswith('fin_')])}")
        print(f"训练集样本: {len(train_df):,}")
        print(f"验证集样本: {len(val_df):,}")
        print(f"测试集样本: {len(test_df):,}")
        print(f"程序执行时间: {execution_time:.1f} 分钟")

        return {
            'models': models,
            'scaler': scaler,
            'features': feature_cols,
            'best_params': best_params,
            'results': results,
            'feature_importance': feature_importance,
            'test_df': test_df,
            'predictions': predictions,
            'probabilities': probabilities
        }

    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        traceback.print_exc()
        return None


# ==================== 运行程序 ====================
if __name__ == "__main__":
    print("开始运行台湾股票超额收益预测模型...")
    result = main()

    if result is not None:
        print_section("程序执行成功!")
        print("已生成以下文件:")
        print("1. stock_predictions_*.csv - 预测结果")
        print("2. feature_importance_*.csv - 特征重要性")
        print("3. stock_models_*.pkl - 保存的模型")
        print("4. model_evaluation_*.txt - 模型评估报告")
    else:
        print_section("程序执行失败!")

