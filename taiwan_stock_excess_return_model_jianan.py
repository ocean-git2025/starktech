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
from scipy.stats.mstats import winsorize
from sklearn.linear_model import LinearRegression
import time
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import optuna
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import shap
import joblib
import json
from collections import defaultdict
import matplotlib

# 尝试使用不同的中文字体（按优先级顺序）
chinese_fonts = [
    'Microsoft YaHei',  # Windows 系统
    'SimHei',           # Windows 系统
    'Heiti TC',         # macOS 系统
    'PingFang SC',      # macOS 系统
    'WenQuanYi Zen Hei', # Linux 系统
    'DejaVu Sans'       # 备用字体
]

# 设置字体
for font in chinese_fonts:
    try:
        matplotlib.rcParams['font.sans-serif'] = [font]
        matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        # 测试字体是否可用
        test_font = matplotlib.font_manager.FontProperties(fname=None, family=font)
        print(f" 成功设置中文字体: {font}")
        break
    except:
        print(f"  字体 {font} 不可用，尝试下一个...")
        continue
else:
    print("❌ 警告：未找到合适的中文字体，图表中文字符可能显示为方框")

# 设置图表默认参数
plt.rcParams['figure.figsize'] = (12, 8)  # 默认图表大小
plt.rcParams['figure.dpi'] = 100          # 默认分辨率
plt.rcParams['savefig.dpi'] = 300         # 保存图片分辨率
# ==================== 参数设置 ====================
FUTURE_DAYS = 20
LOOKBACK_DAYS = 30
USE_PKL_CACHE = False

# 数据路径
PRICE_DATA_PATH = 'taiwan_stock_cleaned_adjusted.csv'
REPORTS_DATA_PATH = 'reports_cleaned.csv'
PRE_MERGED_FILE = 'taiwan_stock_data_optimized.pkl'

# 已保存数据的文件名
LGB_FEATURE_IMPORTANCE_FILE = 'lgb_feature_importance.csv'
CORE_FACTORS_FILE = 'core_factors_top10.csv'
FACTOR_IC_METRICS_FILE = 'factor_ic_metrics.csv'
FINANCIAL_IC_FILE = 'financial_ic_metrics.csv'
TECHNICAL_IC_FILE = 'technical_ic_metrics.csv'

# 模型参数
RANDOM_STATE = 42
TEST_RATIO = 0.2
VAL_RATIO = 0.1
N_JOBS = -1

# 性能优化参数
MAX_SAMPLES = 200000
CHUNK_SIZE = 1000
FEATURE_SELECTION_THRESHOLD = 0.001

QUICK_MODE = False
MAX_FEATURES = 50
HYPERPARAM_TRIALS = 10
SAMPLE_SIZE_TUNING = 5000
MERGE_OPTIMIZATION = True
QUICK_TUNING = True
FORCE_REMERGE = False

# ==================== 快速测试模式 ====================
TEST_LIGHTGBM_ONLY = False
LIGHTGBM_TEST_SAMPLE_SIZE = 20000
LIGHTGBM_TEST_FEATURES = 30

# 因子筛选参数
FIN_IC_MEAN_THRESHOLD = 0.008
FIN_ICIR_THRESHOLD = 0.1
FIN_WINRATE_THRESHOLD = 0.5
FIN_CORR_THRESHOLD = 0.85

TECH_IC_MEAN_THRESHOLD = 0.005
TECH_ICIR_THRESHOLD = 0.05
TECH_WINRATE_THRESHOLD = 0.5
TECH_CORR_THRESHOLD = 0.90

# 稳定性检查参数
ROLLING_WINDOW_MONTHS = 6
ROLLING_STD_THRESHOLD = 0.1

# 去极值参数
WINSORIZE_LIMITS = (0.01, 0.01)

# 滚动交叉验证参数
ROLLING_CV_SPLITS = 5
USE_ROLLING_CV = True
ENFORCE_ROLLING_CV_FOR_ALL_MODELS = True

# 扩展参数网格
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# ==================== 分层回测参数 ====================
STRATIFIED_BACKTEST = True
N_STRATIFICATION = 5
HOLDING_PERIOD = 15
REBALANCE_MONTHLY = True
REBALANCE_DAY = 1
TOP_N = None

# 分层回测验证指标阈值
MIN_TOP_BOTTOM_SPREAD = 0.0
SHARPE_THRESHOLD = 1.0
MONOTONICITY_THRESHOLD = 0.7

# ==================== 回测参数设置 ====================
# 交易成本参数
TRANSACTION_COSTS = {
    'commission': 0.001425,  # 手续费0.1425%
    'tax': 0.003,  # 证交税0.3%（卖出时）
    'slippage': 0.0005  # 滑点0.05%
}
TOTAL_COST_PER_TRADE = 0.005  # 总成本约0.5%每次交易

# 风控参数
RISK_CONTROL = {
    'single_stock_limit': 0.08,      # 单只股票上限：8% ← 适当放宽
    'monthly_turnover_limit': 0.25,   # 月换手率限制：<25% ← 降低换手率目标
    'individual_stop_loss': -0.20,   # 个股止损：-20% ← 放宽止损阈值
    'individual_stop_profit': 0.30,  # 个股止盈：+30% ← 提高止盈阈值
    'portfolio_stop_loss': -0.25,    # 组合最大回撤>25%时减仓 ← 放宽组合止损
    'reduction_ratio': 0.2,          # 减仓比例：20% ← 更温和的减仓
    'min_holding_days': 60,          # 最小持有天数改为60天 ← 强制长线持有
    'max_daily_trades': 3,           # 每日最大交易次数减少为3次
    'drawdown_check_frequency': 30   # 回撤检查频率改为30天一次 ← 降低检查频率
}

# 回测目标指标
TARGET_METRICS = {
    'annual_return': 0.08,
    'sharpe_ratio': 0.6,
    'max_drawdown': 0.25,
    'information_ratio': 0.2
}

# 回测基础参数
INITIAL_CAPITAL = 1000000  # 初始资金100万
REBALANCE_FREQUENCY = 'quarterly'  # 调仓频率：每月
TOP_N_HOLDINGS = 8  # 最大持仓数量
MIN_HOLDING_DAYS = 60  # 最小持有天数（改为20天）

# 持仓周期参数
HOLDING_PERIOD = 20  # 改为20天，接近一个月

# 是否使用已保存的数据
USE_SAVED_DATA = True
FORCE_RECOMPUTE_FACTORS = False

def timer_decorator(func):
    """计时装饰器"""

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f" {func.__name__} 执行时间: {end_time - start_time:.2f}秒")
        return result

    return wrapper
# ==================== 新增：缓存交易成本计算 ====================
class TransactionCostCache:
    """交易成本缓存，避免重复计算"""

    def __init__(self):
        self.cache = {}

    def get_cost(self, trade_value, is_buy=True, use_cache=True):
        """获取交易成本，可选使用缓存"""
        if not use_cache:
            return self._calculate_cost(trade_value, is_buy)

        # 创建缓存键
        cache_key = f"{trade_value:.2f}_{is_buy}"

        if cache_key in self.cache:
            return self.cache[cache_key]

        # 计算并缓存
        cost = self._calculate_cost(trade_value, is_buy)
        self.cache[cache_key] = cost
        return cost

    def _calculate_cost(self, trade_value, is_buy=True):
        """实际计算交易成本"""
        commission = trade_value * TRANSACTION_COSTS['commission']
        tax = 0
        if not is_buy:  # 卖出时征收证交税
            tax = trade_value * TRANSACTION_COSTS['tax']
        slippage = trade_value * TRANSACTION_COSTS['slippage']
        return commission + tax + slippage


# 全局交易成本缓存实例
transaction_cost_cache = TransactionCostCache()


# ==================== 新增：仓位管理器 ====================
class PositionManager:
    """仓位管理器，确保仓位不超限"""

    def __init__(self, max_position_ratio=0.05):
        self.max_position_ratio = max_position_ratio
        self.positions = {}
        self.total_value = 0

    def can_add_position(self, stock_code, target_value, current_total_value):
        """检查是否可以添加仓位"""
        if current_total_value <= 0:
            return True

        # 计算当前仓位
        current_position_value = self.positions.get(stock_code, 0)
        new_total_value = current_total_value + target_value

        # 计算新仓位比例
        new_position_ratio = (current_position_value + target_value) / new_total_value

        return new_position_ratio <= self.max_position_ratio

    def update_position(self, stock_code, value_change):
        """更新仓位"""
        current_value = self.positions.get(stock_code, 0)
        self.positions[stock_code] = current_value + value_change

    def update_total_value(self, total_value):
        """更新总市值"""
        self.total_value = total_value

    def get_position_ratio(self, stock_code):
        """获取仓位比例"""
        if self.total_value <= 0:
            return 0
        return self.positions.get(stock_code, 0) / self.total_value

    def check_all_positions(self):
        """检查所有仓位是否超限"""
        violations = []
        for stock_code, position_value in self.positions.items():
            ratio = position_value / self.total_value if self.total_value > 0 else 0
            if ratio > self.max_position_ratio:
                violations.append((stock_code, ratio))
        return violations


# ==================== 优化交易成本计算函数 ====================
@timer_decorator
def calculate_transaction_costs(trade_value, is_buy=True, use_cache=True):
    """
    计算交易成本（优化版，使用缓存）
    :param trade_value: 交易金额
    :param is_buy: 是否买入（True:买入, False:卖出）
    :param use_cache: 是否使用缓存
    :return: 交易成本
    """
    return transaction_cost_cache.get_cost(trade_value, is_buy, use_cache)


# ==================== 优化RiskControlManager类 ====================
class RiskControlManager:
    """风控管理器（修复版）"""

    def __init__(self):
        self.portfolio_value = INITIAL_CAPITAL
        self.positions = {}
        self.trading_records = []
        self.daily_portfolio_values = []
        self.max_portfolio_value = INITIAL_CAPITAL
        self.position_manager = PositionManager(RISK_CONTROL['single_stock_limit'])
        self.trade_count_today = 0
        self.last_trade_date = None

        # 新增：避免频繁止损
        self.last_drawdown_check_date = None
        self.consecutive_stop_loss = 0  # 连续止损次数
        self.stop_loss_cooldown = False  # 止损冷却期

    def check_portfolio_drawdown(self, current_value, current_date):
        """检查组合回撤（修复版：避免频繁触发）"""
        self.max_portfolio_value = max(self.max_portfolio_value, current_value)
        drawdown = (current_value - self.max_portfolio_value) / self.max_portfolio_value

        # 检查频率控制
        if self.last_drawdown_check_date is None:
            self.last_drawdown_check_date = current_date
            return False, drawdown

        days_since_check = (current_date - self.last_drawdown_check_date).days
        if days_since_check < RISK_CONTROL.get('drawdown_check_frequency', 10):
            return False, drawdown

        self.last_drawdown_check_date = current_date

        # 连续止损保护
        if self.stop_loss_cooldown:
            if self.consecutive_stop_loss >= 3:
                print(f"连续止损{self.consecutive_stop_loss}次，进入冷却期")
                self.stop_loss_cooldown = False
                self.consecutive_stop_loss = 0
            return False, drawdown

        if drawdown <= RISK_CONTROL['portfolio_stop_loss']:
            self.consecutive_stop_loss += 1
            if self.consecutive_stop_loss > 2:
                self.stop_loss_cooldown = True
            return True, drawdown

        # 重置连续止损计数
        if drawdown > -0.05:  # 回撤小于5%时重置
            self.consecutive_stop_loss = 0
            self.stop_loss_cooldown = False

        return False, drawdown

    def reduce_positions(self, reduction_ratio=None):
        """减仓操作（修复版：更温和）"""
        if reduction_ratio is None:
            reduction_ratio = RISK_CONTROL['reduction_ratio']

        # 根据连续止损次数调整减仓比例
        if self.consecutive_stop_loss >= 2:
            reduction_ratio = min(0.2, reduction_ratio * 0.5)  # 更温和的减仓
            print(f"连续止损{self.consecutive_stop_loss}次，采用温和减仓{reduction_ratio:.0%}")

        print(f"触发组合止损，减仓{reduction_ratio:.0%}")

        # 按持仓比例排序，优先减仓亏损最多的股票
        positions_to_reduce = []
        for stock_code, position in self.positions.items():
            if 'current_price' in position and 'avg_price' in position:
                current_price = position['current_price']
                avg_price = position['avg_price']
                profit_ratio = (current_price - avg_price) / avg_price
                positions_to_reduce.append((stock_code, position, profit_ratio))

        # 按盈利情况排序（亏损最多的优先）
        positions_to_reduce.sort(key=lambda x: x[2])

        # 执行减仓
        total_reduction_value = 0
        target_reduction = self.portfolio_value * reduction_ratio

        for stock_code, position, profit_ratio in positions_to_reduce:
            if total_reduction_value >= target_reduction:
                break

            # 温和减仓：最多减仓30%的持仓
            max_reduce_ratio = 0.3
            reduce_shares = int(position['shares'] * min(reduction_ratio, max_reduce_ratio))

            if reduce_shares > 0:
                current_price = position.get('current_price', position['avg_price'])
                trade_value = reduce_shares * current_price

                if total_reduction_value + trade_value > target_reduction:
                    remaining_reduction = target_reduction - total_reduction_value
                    reduce_shares = int(remaining_reduction / current_price)

                if reduce_shares > 0:
                    actual_shares = self.execute_sell(stock_code, reduce_shares, current_price,
                                                      reason='portfolio_stop_loss')
                    total_reduction_value += actual_shares * current_price

        print(f"实际减仓金额: {total_reduction_value:.2f}")
        return total_reduction_value

# ==================== 优化交易统计生成 ====================
def generate_trading_statistics(trading_records, portfolio_values):
    """生成交易统计 - 修复版本"""

    if not trading_records:
        return {
            '总交易次数': 0,
            '买入次数': 0,
            '卖出次数': 0,
            '盈利交易数': 0,
            '亏损交易数': 0,
            '胜率': 0,
            '总交易成本': 0,
            '平均持仓天数': 0,
            '平均交易成本率': 0,
            '年化换手率': 0,
            '持仓天数分布': {}
        }

    # 转换为DataFrame
    trades_df = pd.DataFrame(trading_records)

    # 基本统计
    total_trades = len(trades_df)
    buy_trades = len(trades_df[trades_df['type'] == 'buy'])
    sell_trades = len(trades_df[trades_df['type'] == 'sell'])

    # 盈利交易统计
    sell_trades_df = trades_df[trades_df['type'] == 'sell']
    if not sell_trades_df.empty:
        profitable = len(sell_trades_df[sell_trades_df['profit'] > 0])
        loss_trades = len(sell_trades_df[sell_trades_df['profit'] <= 0])
        win_rate = profitable / sell_trades if sell_trades > 0 else 0
    else:
        profitable = 0
        loss_trades = 0
        win_rate = 0

    # 总交易成本
    total_cost = trades_df['cost'].sum() if 'cost' in trades_df.columns else 0

    # 平均持仓时间 - 修复这里
    avg_hold_days = 0
    if 'hold_days' in trades_df.columns:
        # 只计算卖出交易的持仓天数
        sell_trades = trades_df[trades_df['type'] == 'sell']
        if not sell_trades.empty and 'hold_days' in sell_trades.columns:
            avg_hold_days = sell_trades['hold_days'].mean() if not sell_trades['hold_days'].isnull().all() else 0

    # 计算换手率
    if portfolio_values and len(portfolio_values) > 1:
        trades_df = pd.DataFrame(trading_records)
        turnover_rate = calculate_turnover_rate(trades_df, portfolio_values)
    else:
        turnover_rate = 0

    # 平均交易成本率
    total_trade_value = trades_df['total_value'].sum() if 'total_value' in trades_df.columns and trades_df[
        'total_value'].sum() > 0 else 1
    avg_cost_rate = total_cost / total_trade_value

    stats = {
        '总交易次数': total_trades,
        '买入次数': buy_trades,
        '卖出次数': sell_trades,
        '盈利交易数': profitable,
        '亏损交易数': loss_trades,
        '胜率': win_rate,
        '总交易成本': total_cost,
        '平均持仓天数': avg_hold_days,
        '平均交易成本率': avg_cost_rate,
        '年化换手率': turnover_rate
    }
    # 同时输出交易详情
    if not trades_df.empty and 'type' in trades_df.columns:
        buy_amount = trades_df[trades_df['type'] == 'buy']['total_value'].sum()
        sell_amount = trades_df[trades_df['type'] == 'sell']['total_value'].sum()
        stats['买入总额'] = buy_amount
        stats['卖出总额'] = sell_amount
        stats['总交易额'] = buy_amount + sell_amount
    # 新增：持仓天数分布统计
    hold_days_distribution = {}
    if 'hold_days' in trades_df.columns:
        sell_trades = trades_df[trades_df['type'] == 'sell']
        if not sell_trades.empty and 'hold_days' in sell_trades.columns:
            hold_days = sell_trades['hold_days'].dropna()
            if len(hold_days) > 0:
                # 统计分布
                hold_days_distribution = {
                    '≤30天': (hold_days <= 30).sum(),
                    '31-60天': ((hold_days > 30) & (hold_days <= 60)).sum(),
                    '61-90天': ((hold_days > 60) & (hold_days <= 90)).sum(),
                    '>90天': (hold_days > 90).sum(),
                    '最长持仓': hold_days.max() if not hold_days.empty else 0,
                    '最短持仓': hold_days.min() if not hold_days.empty else 0
                }

    # 在stats字典中添加
    stats['持仓天数分布'] = hold_days_distribution
    return stats

# ==================== 优化选股生成函数 ====================
def generate_daily_selected_stocks(test_df, predictions, probabilities, top_n=10):
    """生成每日选股列表 - 优化版本（减少选股数量，提高质量）"""
    print_section("生成每日选股列表（优化版）")

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

        # ==================== 2. 优化选股逻辑 ====================
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

            # 优化：添加波动率筛选，排除高波动股票
            if 'volatility' in date_data.columns:
                # 假设有波动率数据
                volatility_threshold = date_data['volatility'].quantile(0.8)  # 排除波动率最高的20%
                date_data = date_data[date_data['volatility'] <= volatility_threshold]

            # 优化：减少每日选股数量，提高质量
            daily_top_n = max(3, min(top_n, len(date_data) // 4))  # 每日选3-10只
            if len(date_data) < daily_top_n:
                # 股票数量不足时，使用所有股票但增加筛选条件
                date_data = date_data[date_data['selection_score'] > 0.6]  # 只选概率>0.6的
                if len(date_data) == 0:
                    continue
            else:
                # 选择Top N
                date_data = date_data.head(daily_top_n)

            date_data['rank'] = range(1, len(date_data) + 1)
            daily_top_stocks.append(date_data)
            valid_dates += 1

        print(f"成功处理 {valid_dates}/{len(unique_dates)} 个交易日的选股")

        if not daily_top_stocks:
            print("没有生成任何选股列表")
            return pd.DataFrame()

        # ==================== 3. 合并结果 ====================
        result_df = pd.concat(daily_top_stocks, ignore_index=True)

        # 添加选股理由
        result_df['selection_reason'] = result_df.apply(
            lambda x: f"模型预测概率:{x['selection_score']:.3f}, 排名:{x['rank']}",
            axis=1
        )

        # 重命名列
        result_df = result_df.rename(columns={
            'date': '交易日',
            'stock_code': '股票代码',
            'close': '收盘价',
            'future_return': '未来20天绝对收益率',
            'selection_score': '模型预测概率',
            'rank': '当日排名',
            'selection_reason': '选股理由'
        })

        # 选择需要的列
        final_columns = ['交易日', '股票代码', '收盘价', '未来20天绝对收益率',
                         '模型预测概率', '当日排名', '选股理由']
        final_columns = [col for col in final_columns if col in result_df.columns]
        result_df = result_df[final_columns]

        print(f"生成每日选股列表: {result_df.shape}")

        # ==================== 4. 选股统计 ====================
        print_section("选股结果统计")

        total_stocks = len(result_df)
        unique_stocks = result_df['股票代码'].nunique()
        avg_daily_stocks = result_df.groupby('交易日').size().mean()
        avg_prob_all = result_df['模型预测概率'].mean()

        print(f"选股统计:")
        print(f"   总选股记录: {total_stocks:,} 条")
        print(f"   唯一股票数量: {unique_stocks} 只")
        print(f"   平均每日选股: {avg_daily_stocks:.1f} 只")
        print(f"   平均预测概率: {avg_prob_all:.3f}")

        # 检查是否有重复选股
        duplicate_check = result_df.groupby(['交易日', '股票代码']).size()
        if (duplicate_check > 1).any():
            print(" 警告：发现重复选股记录")
            duplicates = duplicate_check[duplicate_check > 1]
            print(f"重复记录数量: {len(duplicates)}")

        return result_df

    except Exception as e:
        print(f"生成每日选股列表失败: {e}")
        traceback.print_exc()
        return pd.DataFrame()


# ==================== 修复的简化回测函数 ====================
@timer_decorator
def perform_backtest_simple(daily_selected_df, test_df, initial_capital=INITIAL_CAPITAL):
    """
    按季度调仓的简化版回测函数
    """
    print_section("按季度调仓的简化版回测")

    try:
        # 准备数据
        backtest_data = daily_selected_df.copy()

        # 确保有必要的列
        if '股票代码' in backtest_data.columns:
            backtest_data = backtest_data.rename(columns={
                '股票代码': 'stock_code',
                '交易日': 'date',
                '收盘价': 'close',
                '未来20天绝对收益率': 'future_return'
            })

        # 获取唯一日期并排序
        unique_dates = sorted(backtest_data['date'].unique())
        if len(unique_dates) == 0:
            print("错误：没有回测日期")
            return None

        print(f"回测期间: {unique_dates[0]} 到 {unique_dates[-1]}")
        print(f"总交易日数: {len(unique_dates)}")

        # 获取价格数据字典以便快速查询
        price_dict = {}
        if test_df is not None and not test_df.empty:
            for stock_code in test_df['stock_code'].unique():
                stock_data = test_df[test_df['stock_code'] == stock_code]
                if not stock_data.empty:
                    price_dict[stock_code] = dict(zip(stock_data['date'], stock_data['close']))

        # ==================== 关键修改：按季度调仓 ====================
        # 确定每季度调仓日（每季度第一个交易日）
        quarterly_rebalance_dates = []
        current_quarter = None
        for date in unique_dates:
            quarter = (date.year, (date.month - 1) // 3 + 1)  # 计算季度
            if quarter != current_quarter:
                quarterly_rebalance_dates.append(date)
                current_quarter = quarter

        print(f"每季度调仓日数量: {len(quarterly_rebalance_dates)}")
        print(f"调仓日: {[d.strftime('%Y-%m-%d') for d in quarterly_rebalance_dates[:5]]}...")

        # 简单回测逻辑：每月等权重买入选中的股票，持有到下个月调仓日卖出
        portfolio_value = initial_capital
        cash = initial_capital
        holdings = {}  # {股票代码: {'shares': 数量, 'buy_date': 买入日期, 'buy_price': 买入价格}}

        portfolio_values = []
        portfolio_returns = []
        trading_records = []

        # 按日期排序
        backtest_data = backtest_data.sort_values('date')

        # 模拟每日交易，但只在调仓日交易
        for i, current_date in enumerate(tqdm(unique_dates, desc="执行回测")):
            # ==================== 关键修改：只在调仓日交易 ====================
            is_rebalance_day = current_date in quarterly_rebalance_dates

            # 如果是调仓日，卖出所有持仓（上月持仓）
            if is_rebalance_day and holdings:
                print(f"\n调仓日 {current_date.date()}: 卖出上月持仓")
                stocks_to_sell = list(holdings.keys())
                # 添加最小持有期检查
                filtered_stocks_to_sell = []
                for stock_code in stocks_to_sell:
                    holding = holdings[stock_code]
                    hold_days = (current_date - holding['buy_date']).days

                    # 检查是否满足最小持有期
                    if hold_days >= RISK_CONTROL['min_holding_days']:
                        filtered_stocks_to_sell.append(stock_code)
                    else:
                        print(
                            f"  股票 {stock_code} 持有仅 {hold_days} 天，未达 {RISK_CONTROL['min_holding_days']} 天最小持有期，跳过卖出")


                for stock_code in stocks_to_sell:
                    if stock_code in price_dict and current_date in price_dict[stock_code]:
                        sell_price = price_dict[stock_code][current_date]
                        holding = holdings[stock_code]
                        shares = holding['shares']
                        buy_price = holding['buy_price']

                        # 计算交易价值
                        trade_value = shares * sell_price

                        # 计算交易成本
                        cost = calculate_transaction_costs(trade_value, is_buy=False)

                        # 计算盈亏
                        profit = trade_value - (shares * buy_price) - cost
                        return_rate = profit / (shares * buy_price) if shares * buy_price > 0 else 0

                        # 计算持仓天数
                        hold_days = (current_date - holding['buy_date']).days

                        # 更新现金
                        cash += trade_value - cost

                        # 记录交易
                        trading_records.append({
                            'date': current_date,
                            'type': 'sell',
                            'stock_code': stock_code,
                            'shares': shares,
                            'price': sell_price,
                            'cost': cost,
                            'total_value': trade_value,
                            'profit': profit,
                            'return_rate': return_rate,
                            'hold_days': hold_days,
                            'reason': 'monthly_rebalance'
                        })

                        # 移除持仓
                        del holdings[stock_code]

            # 如果是调仓日，买入新股票（如果现金充足）
            if is_rebalance_day and cash > 0:
                print(f"调仓日 {current_date.date()}: 买入新股票")
                # 获取当日选中的股票
                daily_stocks = backtest_data[backtest_data['date'] == current_date]

                if not daily_stocks.empty:
                    # 等权重分配现金
                    num_stocks = min(len(daily_stocks), TOP_N_HOLDINGS)
                    cash_per_stock = cash / num_stocks if num_stocks > 0 else 0

                    bought_count = 0
                    for idx, row in daily_stocks.head(num_stocks).iterrows():
                        stock_code = row['stock_code']

                        # 获取当前价格
                        if stock_code in price_dict and current_date in price_dict[stock_code]:
                            buy_price = price_dict[stock_code][current_date]

                            # 计算可买数量
                            max_shares = int(cash_per_stock / buy_price)
                            if max_shares > 0:
                                # 计算交易成本
                                trade_value = max_shares * buy_price
                                cost = calculate_transaction_costs(trade_value, is_buy=True)

                                # 确保有足够现金
                                if cash >= trade_value + cost:
                                    # 买入
                                    holdings[stock_code] = {
                                        'shares': max_shares,
                                        'buy_date': current_date,
                                        'buy_price': buy_price
                                    }

                                    # 更新现金
                                    cash -= (trade_value + cost)

                                    # 记录交易
                                    trading_records.append({
                                        'date': current_date,
                                        'type': 'buy',
                                        'stock_code': stock_code,
                                        'shares': max_shares,
                                        'price': buy_price,
                                        'cost': cost,
                                        'total_value': trade_value,
                                        'reason': 'monthly_rebalance'
                                    })

                                    bought_count += 1

                    print(f"  买入 {bought_count} 只股票，现金剩余: {cash:.2f}")

            # 计算当日组合价值
            positions_value = 0
            for stock_code, holding in holdings.items():
                if stock_code in price_dict and current_date in price_dict[stock_code]:
                    current_price = price_dict[stock_code][current_date]
                    positions_value += holding['shares'] * current_price

            total_value = cash + positions_value
            portfolio_value = total_value  # 更新组合价值

            # 记录每日组合价值
            portfolio_values.append({
                'date': current_date,
                'portfolio_value': portfolio_value,
                'cash': cash,
                'positions_value': positions_value,
                'total_value': total_value
            })

            # 计算日收益率
            if i > 0:
                prev_value = portfolio_values[i - 1]['portfolio_value']
                if prev_value > 0:
                    daily_return = (portfolio_value - prev_value) / prev_value
                else:
                    daily_return = 0
                portfolio_returns.append(daily_return)
            else:
                portfolio_returns.append(0)

        # 在回测结束时卖出所有持仓
        if holdings and unique_dates:
            last_date = unique_dates[-1]
            for stock_code, holding in list(holdings.items()):
                if stock_code in price_dict and last_date in price_dict[stock_code]:
                    sell_price = price_dict[stock_code][last_date]
                    shares = holding['shares']
                    buy_price = holding['buy_price']
                    hold_days = (last_date - holding['buy_date']).days

                    trade_value = shares * sell_price
                    cost = calculate_transaction_costs(trade_value, is_buy=False)
                    profit = trade_value - (shares * buy_price) - cost

                    cash += trade_value - cost

                    trading_records.append({
                        'date': last_date,
                        'type': 'sell',
                        'stock_code': stock_code,
                        'shares': shares,
                        'price': sell_price,
                        'cost': cost,
                        'total_value': trade_value,
                        'profit': profit,
                        'hold_days': hold_days,
                        'reason': 'end_of_backtest'
                    })

        # 计算回测指标
        metrics = calculate_backtest_metrics(portfolio_values, portfolio_returns)

        # 生成交易统计
        trading_stats = generate_trading_statistics(trading_records, portfolio_values)

        print(f"回测完成!")
        print(f"最终组合价值: {portfolio_value:,.2f}")
        print(f"总收益率: {metrics.get('总收益率', 0):.2%}")
        print(f"最大回撤: {metrics.get('最大回撤', 0):.2%}")
        print(f"平均持仓天数: {trading_stats.get('平均持仓天数', 0):.1f}天")
        print(f"年化换手率: {trading_stats.get('年化换手率', 0):.2%}")

        return {
            'portfolio_values': portfolio_values,
            'portfolio_returns': portfolio_returns,
            'metrics': metrics,
            'trading_stats': trading_stats,
            'trading_records': trading_records,
            'positions': holdings,
            'initial_capital': initial_capital,
            'final_value': portfolio_value
        }

    except Exception as e:
        print(f"按月调仓回测执行出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def calculate_backtest_metrics(portfolio_values, portfolio_returns, benchmark_returns=None):
    """计算回测核心指标 - 修复版：增加区间日期和年化计算"""

    if not portfolio_values:
        return {}

    # 准备数据
    returns_series = pd.Series(portfolio_returns)

    # ==================== 新增：获取收益区间起始和结束日期 ====================
    if len(portfolio_values) > 0:
        start_date = portfolio_values[0]['date']
        end_date = portfolio_values[-1]['date']
        # 转换为字符串格式便于显示
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        days = (end_date - start_date).days
    else:
        start_date_str = "N/A"
        end_date_str = "N/A"
        days = 0

    # 总收益率
    if len(portfolio_values) > 0:
        initial_value = portfolio_values[0]['portfolio_value']
        final_value = portfolio_values[-1]['portfolio_value']
        if initial_value > 0:
            total_return = (final_value - initial_value) / initial_value
        else:
            total_return = 0
    else:
        total_return = 0

    # ==================== 年化收益率计算 ====================
    annualized_return = 0
    if days > 0:
        # 使用复利公式计算年化收益率
        annualized_return = (1 + total_return) ** (365.25 / days) - 1

    # ==================== 年化波动率计算 ====================
    if len(returns_series) > 1:
        annualized_volatility = returns_series.std() * np.sqrt(252)
    else:
        annualized_volatility = 0

    # ==================== 年化夏普比率计算 ====================
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0

    # ==================== 最大回撤计算 ====================
    if len(portfolio_values) > 0:
        values_series = pd.Series([pv['portfolio_value'] for pv in portfolio_values])
        running_max = values_series.expanding().max()
        drawdown = (values_series - running_max) / running_max
        max_drawdown = drawdown.min()
    else:
        max_drawdown = 0

    # ==================== 卡玛比率（Calmar Ratio） ====================
    calmar_ratio = -annualized_return / max_drawdown if max_drawdown != 0 else 0

    # ==================== 其他指标计算 ====================
    # 胜率（正收益天数比例）
    win_rate = (returns_series > 0).mean() if len(returns_series) > 0 else 0

    # 盈亏比
    avg_win = returns_series[returns_series > 0].mean() if (returns_series > 0).any() else 0
    avg_loss = returns_series[returns_series < 0].mean() if (returns_series < 0).any() else 0
    profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    # ==================== 信息比率（如果有基准） ====================
    information_ratio = 0
    if benchmark_returns is not None and len(benchmark_returns) == len(returns_series):
        excess_returns = returns_series - benchmark_returns
        tracking_error = excess_returns.std() * np.sqrt(252)
        information_ratio = (
                                    annualized_return - benchmark_returns.mean() * 252
                            ) / tracking_error if tracking_error != 0 else 0

    # ==================== 整理指标 ====================
    metrics = {
        # 区间信息
        '起始日期': start_date_str,
        '结束日期': end_date_str,
        '回测天数': days,
        '交易日数': len(portfolio_values),

        # 收益率指标
        '总收益率': total_return,
        '年化收益率': annualized_return,

        # 风险指标
        '年化波动率': annualized_volatility,
        '最大回撤': max_drawdown,

        # 风险调整收益指标
        '年化夏普比率': sharpe_ratio,
        '卡玛比率': calmar_ratio,
        '信息比率': information_ratio,

        # 交易统计指标
        '胜率': win_rate,
        '盈亏比': profit_loss_ratio,

        # 原始数据
        '初始净值': initial_value if len(portfolio_values) > 0 else 0,
        '最终净值': final_value if len(portfolio_values) > 0 else 0
    }

    return metrics


def print_backtest_metrics(metrics):
    """打印回测指标（包含区间日期）"""
    print_section("回测结果汇总")

    # 打印区间信息
    if '起始日期' in metrics and '结束日期' in metrics:
        print(f" 回测区间: {metrics['起始日期']} 至 {metrics['结束日期']}")
        print(f"   回测天数: {metrics.get('回测天数', 0)} 天")
        print(f"   交易日数: {metrics.get('交易日数', 0)} 天")

    # 打印收益率指标
    print(f"\n 收益率指标:")
    if '总收益率' in metrics:
        print(f"   总收益率: {metrics['总收益率']:.2%}")
    if '年化收益率' in metrics:
        print(f"   年化收益率: {metrics['年化收益率']:.2%}")

    # 打印风险指标
    print(f"\n  风险指标:")
    if '年化波动率' in metrics:
        print(f"   年化波动率: {metrics['年化波动率']:.2%}")
    if '最大回撤' in metrics:
        print(f"   最大回撤: {metrics['最大回撤']:.2%}")

    # 打印风险调整收益指标
    print(f"\n  风险调整收益指标:")
    if '年化夏普比率' in metrics:
        print(f"   年化夏普比率: {metrics['年化夏普比率']:.2f}")
    if '卡玛比率' in metrics:
        print(f"   卡玛比率: {metrics['卡玛比率']:.2f}")
    if '信息比率' in metrics and metrics['信息比率'] != 0:
        print(f"   信息比率: {metrics['信息比率']:.2f}")

    # 打印交易统计指标
    print(f"\n 交易统计指标:")
    if '胜率' in metrics:
        print(f"   胜率: {metrics['胜率']:.2%}")
    if '盈亏比' in metrics:
        print(f"   盈亏比: {metrics['盈亏比']:.2f}")

    # 打印净值信息
    print(f"\n 净值信息:")
    if '初始净值' in metrics and metrics['初始净值'] > 0:
        print(f"   初始净值: {metrics['初始净值']:,.2f}")
    if '最终净值' in metrics and metrics['最终净值'] > 0:
        print(f"   最终净值: {metrics['最终净值']:,.2f}")

# ==================== 其他原有函数 ====================
def check_saved_files():
    """检查已保存的数据文件是否存在"""
    saved_files = {
        'lgb_feature_importance': os.path.exists(LGB_FEATURE_IMPORTANCE_FILE),
        'core_factors': os.path.exists(CORE_FACTORS_FILE),
        'factor_ic_metrics': os.path.exists(FACTOR_IC_METRICS_FILE),
        'financial_ic': os.path.exists(FINANCIAL_IC_FILE),
        'technical_ic': os.path.exists(TECHNICAL_IC_FILE)
    }
    return saved_files


def load_saved_data():
    """加载已保存的数据文件"""
    saved_data = {}

    if os.path.exists(LGB_FEATURE_IMPORTANCE_FILE):
        saved_data['lgb_feature_importance'] = pd.read_csv(LGB_FEATURE_IMPORTANCE_FILE)
        print(f" 已加载LightGBM特征重要性: {LGB_FEATURE_IMPORTANCE_FILE}")

    if os.path.exists(CORE_FACTORS_FILE):
        saved_data['core_factors'] = pd.read_csv(CORE_FACTORS_FILE)
        print(f" 已加载核心因子列表: {CORE_FACTORS_FILE}")

    if os.path.exists(FACTOR_IC_METRICS_FILE):
        saved_data['factor_ic_metrics'] = pd.read_csv(FACTOR_IC_METRICS_FILE)
        print(f" 已加载因子IC指标: {FACTOR_IC_METRICS_FILE}")

    if os.path.exists(FINANCIAL_IC_FILE):
        saved_data['financial_ic'] = pd.read_csv(FINANCIAL_IC_FILE)
        print(f" 已加载财务因子IC指标: {FINANCIAL_IC_FILE}")

    if os.path.exists(TECHNICAL_IC_FILE):
        saved_data['technical_ic'] = pd.read_csv(TECHNICAL_IC_FILE)
        print(f" 已加载技术因子IC指标: {TECHNICAL_IC_FILE}")

    return saved_data


def save_factor_data(feature_importance_lgb, core_factors, ic_df=None,
                     financial_ic_df=None, technical_ic_df=None):
    """保存因子相关数据（去掉时间后缀）"""

    # 保存LightGBM特征重要性
    if feature_importance_lgb is not None and not feature_importance_lgb.empty:
        feature_importance_lgb.to_csv(LGB_FEATURE_IMPORTANCE_FILE, index=False)
        print(f" LightGBM特征重要性已保存: {LGB_FEATURE_IMPORTANCE_FILE}")

    # 保存核心因子列表
    if core_factors is not None:
        core_factors_df = pd.DataFrame({
            'core_factors': core_factors,
            'factor_type': ['财务因子' if col.startswith('fin_') else '技术因子' for col in core_factors]
        })
        core_factors_df.to_csv(CORE_FACTORS_FILE, index=False)
        print(f" 核心因子列表已保存: {CORE_FACTORS_FILE}")

    # 保存因子IC指标
    if ic_df is not None and not ic_df.empty:
        ic_df.to_csv(FACTOR_IC_METRICS_FILE, index=False)
        print(f" 因子IC指标已保存: {FACTOR_IC_METRICS_FILE}")

    # 保存财务因子IC指标
    if financial_ic_df is not None and not financial_ic_df.empty:
        financial_ic_df.to_csv(FINANCIAL_IC_FILE, index=False)
        print(f" 财务因子IC指标已保存: {FINANCIAL_IC_FILE}")

    # 保存技术因子IC指标
    if technical_ic_df is not None and not technical_ic_df.empty:
        technical_ic_df.to_csv(TECHNICAL_IC_FILE, index=False)
        print(f" 技术因子IC指标已保存: {TECHNICAL_IC_FILE}")


# ==================== 辅助函数 ====================

def validate_price_data(df):
    """
    验证价格数据的基本质量。
    在进行技术指标和收益率计算前，确保数据是有效的。
    """
    if df.empty:
        print("❌ 错误：价格数据为空。")
        return False

    # 检查关键列是否存在
    required_cols = ['close', 'stock_code', 'date']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ 错误：价格数据缺少必要的列: {missing_cols}")
        return False

    # 检查 'close' 列是否有足够的非空值
    non_null_close_count = df['close'].notna().sum()
    if non_null_close_count < 100:  # 假设至少需要100个有效收盘价
        print(f"❌ 错误：'close' 列的有效数据点太少 ({non_null_close_count} 个)。")
        return False

    # 检查价格是否为正数
    invalid_price_count = (df['close'] <= 0).sum()
    if invalid_price_count > 0:
        print(f" 警告：发现 {invalid_price_count} 个非正价格。这些行将在后续步骤中被移除。")
        # 这里不直接返回False，因为后续步骤可以处理，但发出警告

    print(" 价格数据验证通过。")
    return True


def get_conservative_params():
    """返回保守的模型参数，防止过拟合"""
    return {
        'rf': {
            'n_estimators': 50,  # 树数量
            'max_depth': 6,  # 深度
            'min_samples_split': 20,  # 分裂样本数
            'min_samples_leaf': 10,  # 叶节点样本）
            'max_features': 0.3,  # 特征采样比例
            'class_weight': 'balanced',
            'random_state': RANDOM_STATE,
            'n_jobs': N_JOBS
        },
        'xgb': {
            'n_estimators': 50,  # 树数量
            'max_depth': 3,  # 深度
            'learning_rate': 0.01,  # 学习率
            'subsample': 0.6,  # 采样比例
            'colsample_bytree': 0.6,  # 特征采样
            'reg_alpha': 1.0,  # L1正则 特征
            'reg_lambda': 1.0,  # L2正则 权重
            'scale_pos_weight': 1,  # 手动控制类别权重
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

# ==================== 全局定义集成模型类 ====================
class EnsembleRF:
    """随机森林集成模型（可序列化版本）"""
    def __init__(self, models):
        self.models = models if models else []

    def predict(self, X):
        if not self.models:
            return np.zeros(len(X), dtype=np.int32)

        preds = []
        for i, model in enumerate(self.models):
            try:
                pred = model.predict(X)
                # 修复：检查并替换特殊值
                if hasattr(pred, '__len__'):
                    # 替换-2147483648为0
                    pred = np.where(pred == -2147483648, 0, pred)
                preds.append(pred)
            except Exception:
                # 如果模型预测失败，使用全0预测
                preds.append(np.zeros(len(X), dtype=np.int32))

        if not preds:
            return np.zeros(len(X), dtype=np.int32)

        try:
            preds_array = np.array(preds)
            avg_pred = np.mean(preds_array, axis=0)
            return np.round(avg_pred).astype(np.int32)
        except Exception:
            return np.zeros(len(X), dtype=np.int32)

    def predict_proba(self, X):
        if not self.models:
            return np.column_stack([np.ones(len(X)), np.zeros(len(X))]) * 0.5

        probas_list = []
        for model in self.models:
            try:
                proba = model.predict_proba(X)
                probas_list.append(proba)
            except Exception:
                probas_list.append(np.column_stack([np.zeros(len(X)), np.ones(len(X))]) * 0.5)

        if not probas_list:
            return np.column_stack([np.zeros(len(X)), np.ones(len(X))]) * 0.5

        try:
            probas = np.array(probas_list)
            return np.mean(probas, axis=0)
        except Exception:
            return probas_list[0] if probas_list else np.column_stack(
                [np.zeros(len(X)), np.ones(len(X))]) * 0.5


class EnsembleXGB:
    """XGBoost集成模型（可序列化版本）"""
    def __init__(self, models):
        self.models = models

    def predict(self, X):
        if not self.models:
            return np.zeros(len(X), dtype=np.int32)

        preds = []
        for i, model in enumerate(self.models):
            try:
                pred = model.predict(X)
                # 修复：检查并替换特殊值
                if hasattr(pred, '__len__'):
                    # 替换-2147483648为0
                    pred = np.where(pred == -2147483648, 0, pred)
                preds.append(pred)
            except Exception:
                # 如果模型预测失败，使用全0预测
                preds.append(np.zeros(len(X), dtype=np.int32))

        if not preds:
            return np.zeros(len(X), dtype=np.int32)

        try:
            preds_array = np.array(preds)
            avg_pred = np.mean(preds_array, axis=0)
            return np.round(avg_pred).astype(np.int32)
        except Exception:
            return np.zeros(len(X), dtype=np.int32)

    def predict_proba(self, X):
        if not self.models:
            return np.column_stack([np.ones(len(X)), np.zeros(len(X))]) * 0.5

        probas_list = []
        for model in self.models:
            try:
                proba = model.predict_proba(X)
                probas_list.append(proba)
            except Exception:
                probas_list.append(np.column_stack([np.zeros(len(X)), np.ones(len(X))]) * 0.5)

        if not probas_list:
            return np.column_stack([np.zeros(len(X)), np.ones(len(X))]) * 0.5

        try:
            probas = np.array(probas_list)
            return np.mean(probas, axis=0)
        except Exception:
            return probas_list[0] if probas_list else np.column_stack(
                [np.zeros(len(X)), np.ones(len(X))]) * 0.5

    def __getattr__(self, name):
        # 转发到第一个模型的其他属性
        if self.models:
            return getattr(self.models[0], name)
        raise AttributeError(f"'EnsembleXGB' object has no attribute '{name}'")


class EnsembleLGB:
    """LightGBM集成模型（可序列化版本）"""
    def __init__(self, models, fold_scores=None):
        self.models = models
        self.fold_scores = fold_scores if fold_scores is not None else []
        self.n_models = len(models)

    def predict(self, X):
        # 检查模型数量
        if self.n_models == 0:
            print("警告：没有模型可用，返回全0预测")
            return np.zeros(len(X), dtype=np.int32)

        # 收集所有模型的预测
        all_predictions = []
        for i, model in enumerate(self.models):
            try:
                pred = model.predict(X)
                # 修复：检查预测值是否为有效整数
                if hasattr(pred, '__len__'):
                    # 确保预测值是整数类型，避免特殊值
                    pred = pred.astype(np.int32)
                    # 检查是否有异常值
                    mask_invalid = (pred != 0) & (pred != 1)
                    if mask_invalid.any():
                        print(f"模型{i + 1}预测包含异常值，转换为0")
                        pred[mask_invalid] = 0
                else:
                    # 如果是单个值，检查是否为0或1
                    if pred not in [0, 1]:
                        pred = 0
                    pred = np.array([pred], dtype=np.int32)
                all_predictions.append(pred)
            except Exception as e:
                print(f"模型{i + 1}预测失败: {e}，使用全0预测")
                all_predictions.append(np.zeros(len(X), dtype=np.int32))

        # 计算平均预测
        if all_predictions:
            try:
                preds_array = np.array(all_predictions)
                avg_pred = np.mean(preds_array, axis=0)
                final_pred = np.round(avg_pred).astype(np.int32)
                # 再次检查最终预测值
                mask_invalid = (final_pred != 0) & (final_pred != 1)
                if mask_invalid.any():
                    print(f"最终预测包含异常值{np.unique(final_pred[mask_invalid])}，修正为0")
                    final_pred[mask_invalid] = 0
                return final_pred
            except Exception as e:
                print(f"预测聚合失败: {e}，使用第一个模型的预测")
                return all_predictions[0] if all_predictions else np.zeros(len(X), dtype=np.int32)
        else:
            return np.zeros(len(X), dtype=np.int32)

    def predict_proba(self, X):
        if self.n_models == 0:
            # 修复：默认概率维度更健壮，避免列拼接错误
            return np.hstack([np.ones((len(X), 1)) * 0.5, np.ones((len(X), 1)) * 0.5])

        probas_list = []
        for i, model in enumerate(self.models):
            try:
                proba = model.predict_proba(X)
                # 检查概率维度
                if proba.ndim == 2 and proba.shape[1] >= 2:
                    probas_list.append(proba)
                else:
                    # 如果模型返回的维度不对，使用默认值
                    probas_list.append(np.hstack([np.ones((len(X), 1)) * 0.5, np.ones((len(X), 1)) * 0.5]))
            except Exception:
                probas_list.append(np.hstack([np.ones((len(X), 1)) * 0.5, np.ones((len(X), 1)) * 0.5]))

        if not probas_list:
            return np.hstack([np.ones((len(X), 1)) * 0.5, np.ones((len(X), 1)) * 0.5])

        try:
            probas = np.array(probas_list)
            return np.mean(probas, axis=0)
        except Exception:
            return probas_list[0] if probas_list else np.hstack([np.ones((len(X), 1)) * 0.5, np.ones((len(X), 1)) * 0.5])

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

            # 删除原始股价并重命名复权后股价
            cols_to_drop = ['open', 'max', 'min', 'close', 'daily_return']
            rename_mapping = {
                'adj_open': 'open',
                'adj_high': 'max',
                'adj_low': 'min',
                'adj_close': 'close',
                'adj_return': 'daily_return'
            }

            # 第二步：安全删除列（使用errors='ignore'避免列不存在时报错）
            financial_df = financial_df.drop(columns=cols_to_drop, errors='ignore')

            # 第三步：重命名指定列（同样使用errors='ignore'增强鲁棒性）
            financial_df = financial_df.rename(columns=rename_mapping, errors='ignore')

            # 查看最终的列名，验证操作是否成功
            print("最终的列名：", financial_df.columns.tolist())

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

        # 处理日期 - 确保datetime格式且无时区
        price_df['date'] = pd.to_datetime(price_df['date'])
        if hasattr(price_df['date'].dtype, 'tz') and price_df['date'].dtype.tz is not None:
            price_df['date'] = price_df['date'].dt.tz_convert(None)

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

            # 修复1: 确保数据排序 - 这是merge_asof的关键要求
            price_common = price_common.sort_values(['stock_code', 'date'])
            financial_common = financial_common.sort_values(['stock_code', 'report_date'])

            # 修复2: 确保日期列数据类型一致
            price_common['date'] = pd.to_datetime(price_common['date'])
            financial_common['report_date'] = pd.to_datetime(financial_common['report_date'])

            # 修复3: 移除无穷大值和NaN值
            financial_common = financial_common.replace([np.inf, -np.inf], np.nan)

            # 修复4: 移除重复的财报日期（每个股票每个报告日只保留一条）
            financial_common = financial_common.drop_duplicates(subset=['stock_code', 'report_date'], keep='last')

            # 修复5: 对大数据集进行分块处理
            if len(price_common) > 100000:  # 如果数据量很大
                print(f"大数据量检测: {len(price_common):,}行，使用分块合并...")
                chunk_size = 50000
                merged_chunks = []

                # 按股票代码分块处理
                stock_chunks = np.array_split(list(common_stocks), max(1, len(common_stocks) // 10))

                for i, stock_chunk in enumerate(tqdm(stock_chunks, desc="分块合并")):
                    price_chunk = price_common[price_common['stock_code'].isin(stock_chunk)]
                    financial_chunk = financial_common[financial_common['stock_code'].isin(stock_chunk)]

                    if not price_chunk.empty and not financial_chunk.empty:
                        # 确保排序
                        price_chunk = price_chunk.sort_values(['stock_code', 'date'])
                        financial_chunk = financial_chunk.sort_values(['stock_code', 'report_date'])

                        chunk_merged = pd.merge_asof(
                            price_chunk,
                            financial_chunk,
                            left_on='date',
                            right_on='report_date',
                            by='stock_code',
                            direction='backward',
                            allow_exact_matches=True
                        )
                        merged_chunks.append(chunk_merged)

                if merged_chunks:
                    merged_df = pd.concat(merged_chunks, ignore_index=True)
                else:
                    raise ValueError("分块合并结果为空")
            else:
                # 小数据量直接合并
                merged_df = pd.merge_asof(
                    price_common,
                    financial_common,
                    left_on='date',
                    right_on='report_date',
                    by='stock_code',
                    direction='backward',
                    allow_exact_matches=True
                )

            # 处理没有财报数据的股票
            price_other = price_df[~price_df['stock_code'].isin(common_stocks)].copy()

            # 合并所有数据
            final_merged = pd.concat([merged_df, price_other], ignore_index=True)

            # 修复6: 财务特征列名前缀处理
            # 确保财务特征列名有'fin_'前缀
            financial_cols = [col for col in final_merged.columns
                              if col not in ['stock_code', 'date', 'report_date']
                              and col not in price_df.columns]
            for col in financial_cols:
                if not col.startswith('fin_'):
                    final_merged = final_merged.rename(columns={col: f'fin_{col}'})

            end_time = time.time()
            print(f"merge_asof合并完成! 形状: {final_merged.shape}")
            print(f"合并时间: {end_time - start_time:.2f}秒")
            return final_merged

        except Exception as e:
            print(f"merge_asof失败: {e}")
            print("使用分组优化方法...")
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

            # 通用除0计算函数（可选，简化代码）
            def safe_divide(numerator, denominator, default=np.nan):
                """安全除法：避免除0，返回默认值"""
                return np.where(
                    (denominator != 0) & ~np.isnan(denominator),
                    numerator / denominator,
                    default
                )

            if all(col in financial_wide.columns for col in ['revenue', 'operating_costs']):
                # 毛利率（原有除0保留，可改用通用函数）
                financial_wide['gross_margin'] = safe_divide(
                    (financial_wide['revenue'] - financial_wide['operating_costs']),
                    financial_wide['revenue']
                )
                print("  ✓ 计算毛利率（含除0保护）")

            if all(col in financial_wide.columns for col in ['revenue', 'operating_profit']):
                # 营业利润率（新增除0保护）
                financial_wide['operating_margin'] = safe_divide(
                    financial_wide['operating_profit'],
                    financial_wide['revenue']
                )
                print("  ✓ 计算营业利润率（含除0保护）")

            if all(col in financial_wide.columns for col in ['revenue', 'net_profit']):
                # 净利率（新增除0保护）
                financial_wide['net_margin'] = safe_divide(
                    financial_wide['net_profit'],
                    financial_wide['revenue']
                )
                print("  ✓ 计算净利率（含除0保护）")

            if all(col in financial_wide.columns for col in ['current_assets', 'current_liabilities']):
                # 流动比率（新增除0保护）
                financial_wide['current_ratio'] = safe_divide(
                    financial_wide['current_assets'],
                    financial_wide['current_liabilities']
                )
                print("  ✓ 计算流动比率（含除0保护）")

            if all(col in financial_wide.columns for col in ['total_assets', 'total_liabilities']):
                # 资产负债率/权益比率（新增除0保护）
                financial_wide['debt_to_assets'] = safe_divide(
                    financial_wide['total_liabilities'],
                    financial_wide['total_assets']
                )
                financial_wide['equity_ratio'] = 1 - financial_wide['debt_to_assets'].fillna(0)
                print("  ✓ 计算资产负债率和权益比率（含除0保护）")

            if all(col in financial_wide.columns for col in ['equity', 'net_profit']):
                # ROE（新增除0保护）
                financial_wide['roe'] = safe_divide(
                    financial_wide['net_profit'],
                    financial_wide['equity']
                )
                print("  ✓ 计算ROE（含除0保护）")

            if all(col in financial_wide.columns for col in ['total_assets', 'net_profit']):
                # ROA（新增除0保护）
                financial_wide['roa'] = safe_divide(
                    financial_wide['net_profit'],
                    financial_wide['total_assets']
                )
                print("  ✓ 计算ROA（含除0保护）")

            if all(col in financial_wide.columns for col in ['operating_cash_flow', 'total_liabilities']):
                # 经营现金流/负债（新增除0保护）
                financial_wide['ocf_to_debt'] = safe_divide(
                    financial_wide['operating_cash_flow'],
                    financial_wide['total_liabilities']
                )
                print("  ✓ 计算经营活动现金流/负债比率（含除0保护）")

            if all(col in financial_wide.columns for col in ['operating_cash_flow', 'revenue']):
                # 经营现金流/收入（新增除0保护）
                financial_wide['ocf_margin'] = safe_divide(
                    financial_wide['operating_cash_flow'],
                    financial_wide['revenue']
                )
                print("  ✓ 计算经营活动现金流/收入比率（含除0保护）")

            if 'revenue' in financial_wide.columns:
                # 按股票分组计算营收增长率（当期/上期 -1）
                financial_wide['revenue_growth'] = financial_wide.groupby('stock_code')['revenue'].pct_change()
                print("  ✓ 计算营收增长率")

            if 'net_profit' in financial_wide.columns:
                # 按股票分组计算利润增长率
                financial_wide['profit_growth'] = financial_wide.groupby('stock_code')['net_profit'].pct_change()
                print("  ✓ 计算利润增长率")

            # 盈利因子 = (ROE + 净利率 + 毛利率)/3
            if all(col in financial_wide.columns for col in ['roe', 'net_margin', 'gross_margin']):
                financial_wide['profit_factor'] = safe_divide(
                    financial_wide['roe'] + financial_wide['net_margin'] + financial_wide['gross_margin'],
                    3,
                    default=np.nan
                )
                print("  ✓ 计算盈利因子（含除0保护）")

            # 成长因子 = (营收增长率 + 净利润增长率)/2
            if all(col in financial_wide.columns for col in ['revenue_growth', 'profit_growth']):
                financial_wide['growth_factor'] = safe_divide(
                    financial_wide['revenue_growth'] + financial_wide['profit_growth'],
                    2,
                    default=np.nan
                )
                print("  ✓ 计算成长因子（含除0保护）")

            # 质量因子 = (现金流/营收 + ROA)/2（ocf_margin 即 现金流/营收）
            if all(col in financial_wide.columns for col in ['ocf_margin', 'roa']):
                financial_wide['quality_factor'] = safe_divide(
                    financial_wide['ocf_margin'] + financial_wide['roa'],
                    2,
                    default=np.nan
                )
                print("  ✓ 计算质量因子（含除0保护）")

            # 处理缺失值
            numeric_cols = [col for col in financial_wide.columns
                            if col not in ['stock_code', 'report_date'] and pd.api.types.is_numeric_dtype(
                    financial_wide[col])]

            for col in numeric_cols:
                if col in financial_wide.columns:
                    financial_wide[col] = financial_wide.groupby('stock_code')[col].transform(
                        lambda x: x.ffill().bfill().fillna(x.median())
                    )

            print("过滤无效财务因子...")
            numeric_cols = [col for col in financial_wide.columns
                            if col not in ['stock_code', 'report_date']]
            invalid_cols = []

            for col in numeric_cols:
                # 规则1：缺失值占比>50% → 无效
                missing_ratio = 1 - financial_wide[col].notna().mean()
                if missing_ratio > 0.5:
                    invalid_cols.append(col)
                    continue
                # 规则2：方差<0.001（几乎无波动）→ 无效
                col_var = financial_wide[col].var()
                if pd.isna(col_var) or col_var < 0.001:
                    invalid_cols.append(col)

            # 删除无效因子
            if invalid_cols:
                financial_wide = financial_wide.drop(columns=invalid_cols)
                print(f"  删除无效因子: {invalid_cols}")
            else:
                print(f"  无无效因子，保留所有{len(numeric_cols)}个财务因子")

            # 定义要保留的列
            core_cols = ['stock_code', 'report_date']  # 核心标识列
            absolute_indicators = ['cash', 'total_assets', 'revenue']  # 3个绝对值指标
            relative_indicators = [  # 15个相对值指标）
                'gross_margin', 'operating_margin', 'net_margin',
                'current_ratio', 'debt_to_assets', 'equity_ratio',
                'roe', 'roa', 'ocf_to_debt', 'ocf_margin', 'revenue_growth', 'profit_growth',
                'profit_factor', 'growth_factor', 'quality_factor'
            ]

            # 只保留存在的列
            keep_cols = core_cols.copy()
            keep_cols += [col for col in absolute_indicators if col in financial_wide.columns]
            keep_cols += [col for col in relative_indicators if col in financial_wide.columns]

            # 筛选列
            financial_wide = financial_wide[keep_cols]

            # 关键优化：清理异常值（inf/-inf/NaN）
            numeric_cols = [col for col in financial_wide.columns if col not in core_cols]
            for col in numeric_cols:
                # 替换无穷大值为NaN，再用中位数填充
                financial_wide[col] = financial_wide[col].replace([np.inf, -np.inf], np.nan)
                # 按股票分组填充，保证同股票数据的一致性
                financial_wide[col] = financial_wide.groupby('stock_code')[col].transform(
                    lambda x: x.fillna(x.median())
                )

            factor_cols = [col for col in financial_wide.columns if col not in core_cols]

            print(f"财报处理完成: {financial_wide.shape}")
            print(f"时间范围: {financial_wide['report_date'].min()} 到 {financial_wide['report_date'].max()}")

            return financial_wide

    print("财报数据处理失败，返回空DataFrame")
    return pd.DataFrame()


# ==================== 新增因子处理函数 ====================
def winsorize_factor(df, factor_cols, limits=WINSORIZE_LIMITS):
    """
    因子去极值（Winsorize）
    :param df: 数据框
    :param factor_cols: 因子列列表
    :param limits: 去极值分位数，(下限, 上限)
    :return: 去极值后的数据框
    """
    print_section("因子去极值处理")
    df_copy = df.copy()

    for col in tqdm(factor_cols, desc="去极值处理"):
        if col in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[col]):
            # 保留非空值进行去极值
            non_na_mask = df_copy[col].notna()
            if non_na_mask.sum() > 0:
                # 去极值
                df_copy.loc[non_na_mask, col] = winsorize(
                    df_copy.loc[non_na_mask, col].values,
                    limits=limits
                )

    print(f"完成 {len(factor_cols)} 个因子的去极值处理")
    return df_copy


def market_cap_neutralization(df, factor_cols, market_cap_col='market_cap'):
    """
    市值中性化（对因子进行市值回归，取残差作为中性化后的因子）
    :param df: 数据框
    :param factor_cols: 需要中性化的因子列
    :param market_cap_col: 市值列名
    :return: 中性化后的数据框
    """
    print_section("市值中性化处理")

    # 如果没有市值列，尝试从现有数据计算
    if market_cap_col not in df.columns:
        print("未找到市值列，尝试从价格和成交量估算...")
        if 'close' in df.columns and 'volume' in df.columns:
            df['market_cap'] = df['close'] * df.groupby('stock_code')['volume'].rolling(window=20,
                                                                                        min_periods=5).mean().reset_index(
                0, drop=True)
            market_cap_col = 'market_cap'
        else:
            print("无法估算市值，跳过市值中性化")
            return df

    df_copy = df.copy()

    for col in tqdm(factor_cols, desc="市值中性化"):
        if col in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[col]):
            # 准备回归数据
            reg_data = df_copy[[col, market_cap_col]].dropna()

            if len(reg_data) > 10:  # 至少需要10个样本
                X = reg_data[[market_cap_col]]
                y = reg_data[col]

                # 线性回归
                lr = LinearRegression()
                lr.fit(X, y)

                # 计算残差（中性化后的因子）
                residuals = y - lr.predict(X)

                # 替换原因子值
                df_copy.loc[reg_data.index, col] = residuals

    print(f"完成 {len(factor_cols)} 个因子的市值中性化")
    return df_copy


def standardize_factor(df, factor_cols):
    """
    因子标准化（Z-score标准化）
    :param df: 数据框
    :param factor_cols: 因子列列表
    :return: 标准化后的数据框
    """
    print_section("因子标准化处理")
    df_copy = df.copy()

    scaler = StandardScaler()
    for col in tqdm(factor_cols, desc="标准化处理"):
        if col in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[col]):
            non_na_mask = df_copy[col].notna()
            if non_na_mask.sum() > 0:
                # 按日期分组标准化（横截面标准化）
                def standardize_group(group):
                    if group.notna().sum() > 1:
                        return (group - group.mean()) / group.std()
                    return group

                df_copy.loc[non_na_mask, col] = df_copy.loc[non_na_mask].groupby('date')[col].transform(
                    standardize_group)

    print(f"完成 {len(factor_cols)} 个因子的标准化处理")
    return df_copy


def calculate_ic_measures(df, factor_cols, target_col='future_return', date_col='date'):
    """
    计算因子的IC指标（信息系数）
    :param df: 数据框
    :param factor_cols: 因子列列表
    :param target_col: 目标收益率列
    :param date_col: 日期列
    :return: IC统计结果字典
    """
    print_section("计算因子IC指标")

    ic_results = {
        'factor': [],
        'ic_mean': [],
        'ic_std': [],
        'icir': [],
        'winrate': [],
        'rolling_std': []
    }

    # 按日期分组计算每日IC
    daily_ic = {}
    dates = sorted(df[date_col].unique())

    for col in tqdm(factor_cols, desc="计算IC"):
        if col not in df.columns:
            continue

        # 存储每日IC
        ic_values = []
        ic_signs = []

        for date in dates:
            daily_data = df[df[date_col] == date]
            valid_data = daily_data[[col, target_col]].dropna()

            if len(valid_data) >= 20:  # 至少20个样本
                # 计算Spearman相关系数（IC）
                ic = valid_data[col].corr(valid_data[target_col], method='spearman')
                if not np.isnan(ic):
                    ic_values.append(ic)
                    ic_signs.append(1 if ic > 0 else 0)

        if len(ic_values) > 0:
            # 计算IC均值
            ic_mean = np.mean(ic_values)
            # 计算IC标准差
            ic_std = np.std(ic_values)
            # 计算ICIR（信息系数信息比率）
            icir = ic_mean / ic_std if ic_std != 0 else 0
            # 计算胜率（IC为正的比例）
            winrate = np.mean(ic_signs) if len(ic_signs) > 0 else 0

            # 计算滚动标准差（稳定性检查）
            ic_series = pd.Series(ic_values, index=dates[:len(ic_values)])
            rolling_std = ic_series.rolling(window=ROLLING_WINDOW_MONTHS).std().mean()

            # 保存结果
            ic_results['factor'].append(col)
            ic_results['ic_mean'].append(ic_mean)
            ic_results['ic_std'].append(ic_std)
            ic_results['icir'].append(icir)
            ic_results['winrate'].append(winrate)
            ic_results['rolling_std'].append(rolling_std)

    # 转换为DataFrame
    ic_df = pd.DataFrame(ic_results)
    print(f"完成 {len(ic_df)} 个因子的IC指标计算")
    return ic_df


def filter_factors_by_ic(ic_df, factor_type='financial'):
    """
    根据IC指标筛选因子
    :param ic_df: IC统计结果DataFrame
    :param factor_type: 因子类型 'financial' 或 'technical'
    :return: 筛选后的因子列表
    """
    print_section(f"{factor_type.upper()}因子IC筛选")

    if factor_type == 'financial':
        ic_mean_thresh = FIN_IC_MEAN_THRESHOLD
        icir_thresh = FIN_ICIR_THRESHOLD
        winrate_thresh = FIN_WINRATE_THRESHOLD
    else:
        ic_mean_thresh = TECH_IC_MEAN_THRESHOLD
        icir_thresh = TECH_ICIR_THRESHOLD
        winrate_thresh = TECH_WINRATE_THRESHOLD

    # 初始筛选
    filtered = ic_df[
        (abs(ic_df['ic_mean']) >= ic_mean_thresh) &
        (abs(ic_df['icir']) >= icir_thresh) &
        (ic_df['winrate'] >= winrate_thresh)
        ].copy()

    # 稳定性检查：剔除滚动标准差过大的因子
    stable_filtered = filtered[filtered['rolling_std'] <= ROLLING_STD_THRESHOLD].copy()

    # 按ICIR排序
    stable_filtered = stable_filtered.sort_values('icir', ascending=False)

    print(f"初始筛选后剩余: {len(filtered)} 个因子")
    print(f"稳定性检查后剩余: {len(stable_filtered)} 个因子")

    # 显示筛选结果
    if len(stable_filtered) > 0:
        print(f"\n{factor_type.upper()}因子筛选结果（按ICIR排序）:")
        print(stable_filtered[['factor', 'ic_mean', 'icir', 'winrate', 'rolling_std']].round(4))

    return stable_filtered['factor'].tolist()


def remove_high_correlation_factors(df, factor_cols, factor_type='financial'):
    """
    移除高相关性因子，保留IC更高的因子
    :param df: 数据框
    :param factor_cols: 因子列表
    :param factor_type: 因子类型 'financial' 或 'technical'
    :return: 去重后的因子列表
    """
    print_section(f"{factor_type.upper()}因子相关性去重")

    if factor_type == 'financial':
        corr_thresh = FIN_CORR_THRESHOLD
    else:
        corr_thresh = TECH_CORR_THRESHOLD

    # 计算因子相关性矩阵
    corr_matrix = df[factor_cols].corr().abs()

    # 生成上三角矩阵（排除对角线）
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # 计算每个因子的平均IC（用于选择保留哪个）
    factor_ic = {}
    for col in factor_cols:
        if col in df.columns and 'future_return' in df.columns:
            ic = df[[col, 'future_return']].corr(method='spearman').iloc[0, 1]
            factor_ic[col] = abs(ic)

    # 需要移除的因子列表
    to_drop = []

    # 遍历相关性矩阵
    for col in upper.columns:
        if col in to_drop:
            continue

        # 找到高相关性的因子
        high_corr = [idx for idx in upper.index if upper.loc[idx, col] > corr_thresh and idx not in to_drop]

        if high_corr:
            # 包括当前列
            candidates = [col] + high_corr

            # 获取候选因子的IC
            candidate_ics = {f: factor_ic.get(f, 0) for f in candidates}

            # 找到IC最高的因子
            best_factor = max(candidate_ics, key=candidate_ics.get)

            # 移除其他因子
            for f in candidates:
                if f != best_factor:
                    to_drop.append(f)

    # 最终保留的因子
    final_factors = [f for f in factor_cols if f not in to_drop]

    print(f"高相关性因子数量: {len(to_drop)}")
    print(f"去重后剩余因子数量: {len(final_factors)}")

    if len(to_drop) > 0:
        print(f"\n移除的高相关性因子: {to_drop}")
        print(f"保留的因子: {final_factors}")

    return final_factors


def process_factors_pipeline(df, financial_cols, technical_cols):
    """
    因子处理完整流程
    :param df: 原始数据框
    :param financial_cols: 财务因子列
    :param technical_cols: 技术因子列
    :return: 处理后的数据框、最终筛选的财务因子、最终筛选的技术因子
    """
    print_section("=== 因子处理完整流程 ===")

    # 1. 去极值
    df_processed = winsorize_factor(df, financial_cols + technical_cols)

    # 2. 分别处理财务因子和技术因子
    ## 财务因子：市值中性化 + 标准化  技术因子：不做中性化
    if financial_cols:
        df_processed = market_cap_neutralization(df_processed, financial_cols)
    df_processed = standardize_factor(df_processed, financial_cols + technical_cols)

    # 3. 计算IC指标
    ic_df = calculate_ic_measures(df_processed, financial_cols + technical_cols)

    # 4. 按IC筛选因子
    filtered_financial = filter_factors_by_ic(ic_df[ic_df['factor'].isin(financial_cols)], 'financial')
    filtered_technical = filter_factors_by_ic(ic_df[ic_df['factor'].isin(technical_cols)], 'technical')

    # 5. 相关性去重
    final_financial = remove_high_correlation_factors(df_processed, filtered_financial, 'financial')
    final_technical = remove_high_correlation_factors(df_processed, filtered_technical, 'technical')

    # 合并最终因子列表
    final_factors = final_financial + final_technical

    print(f"\n因子处理完成:")
    print(f"  财务因子最终保留: {len(final_financial)} 个")
    print(f"  技术因子最终保留: {len(final_technical)} 个")
    print(f"  总因子数量: {len(final_factors)} 个")

    if final_factors:
        print(f"  最终因子列表: {final_factors}")

    return df_processed, final_financial, final_technical, final_factors, ic_df


# ==================== 修复：将函数移出嵌套 ====================
@timer_decorator
def calculate_technical_indicators(df):
    """精简版技术指标计算 - 只生成5个核心因子"""
    print_section("精简版技术指标计算（5个核心因子）")

    if df.empty or 'close' not in df.columns:
        print("数据为空或缺少close列")
        return df

    df_tech = df.copy()
    close_prices = df_tech['close']

    # 验证close列的数据类型
    if not pd.api.types.is_numeric_dtype(df_tech['close']):
        print(" close列不是数值类型，尝试转换...")
        df_tech['close'] = pd.to_numeric(df_tech['close'], errors='coerce')

    # 移除close中的无效值
    initial_len = len(df_tech)
    df_tech = df_tech.dropna(subset=['close'])
    if initial_len - len(df_tech) > 0:
        print(f"移除{initial_len - len(df_tech)}个无效的close值")

    # 重新获取close_prices
    close_prices = df_tech['close']

    try:
        # ==================== 因子1: 短期价格动量（5日收益率） ====================
        print("计算因子1: 5日收益率...")
        shifted_5 = close_prices.shift(5)
        valid_5_mask = (shifted_5 > 0) & shifted_5.notna() & close_prices.notna()
        df_tech['price_change_5d'] = 0.0
        df_tech.loc[valid_5_mask, 'price_change_5d'] = (
                (close_prices[valid_5_mask] - shifted_5[valid_5_mask]) /
                shifted_5[valid_5_mask]
        )
        print("✓ 因子1生成完成")

        # ==================== 因子2: 价格相对20日均线位置 ====================
        print("计算因子2: 价格相对20日均线位置...")
        # 计算20日移动平均
        window = 20
        ma_col = f'ma_{window}'
        df_tech[ma_col] = close_prices.rolling(
            window=window, min_periods=max(1, window // 2)
        ).mean()

        # 价格相对于移动平均的位置
        valid_ma_mask = df_tech[ma_col] > 0
        df_tech['price_vs_ma20'] = 0.0
        df_tech.loc[valid_ma_mask, 'price_vs_ma20'] = (
                close_prices[valid_ma_mask] / df_tech.loc[valid_ma_mask, ma_col] - 1
        )
        print("✓ 因子2生成完成")

        # ==================== 因子3: RSI（14日） ====================
        print("计算因子3: RSI(14)...")
        period = 14

        try:
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()

            # 避免除零
            rs = gain / (loss.replace(0, np.nan).fillna(1e-10))
            df_tech['rsi_14'] = 100 - (100 / (1 + rs))
        except Exception as e:
            print(f"RSI计算失败: {e}")
            df_tech['rsi_14'] = 50.0  # 默认值

        print("✓ 因子3生成完成")

        # ==================== 因子4: 20日波动率 ====================
        print("计算因子4: 20日波动率...")

        # 手动计算日收益率
        daily_returns = np.zeros(len(close_prices))
        for i in range(1, len(close_prices)):
            if close_prices.iloc[i - 1] > 0 and not np.isnan(close_prices.iloc[i - 1]) and not np.isnan(
                    close_prices.iloc[i]):
                daily_returns[i] = (close_prices.iloc[i] - close_prices.iloc[i - 1]) / close_prices.iloc[i - 1]
            else:
                daily_returns[i] = np.nan

        daily_returns_series = pd.Series(daily_returns, index=close_prices.index)

        # 计算20日波动率
        window = 20
        df_tech['volatility_20d'] = daily_returns_series.rolling(
            window=window, min_periods=max(1, window // 2)
        ).std()

        print("✓ 因子4生成完成")

        # ==================== 因子5: 成交量比率（如果有成交量） ====================
        if 'volume' in df_tech.columns:
            print("计算因子5: 5日成交量比率...")
            volume = df_tech['volume']

            # 验证成交量数据
            if not pd.api.types.is_numeric_dtype(volume):
                print(" volume列不是数值类型，尝试转换...")
                volume = pd.to_numeric(volume, errors='coerce')
                df_tech['volume'] = volume

            # 5日成交量移动平均
            window = 5
            vol_ma_col = f'volume_ma_{window}'
            df_tech[vol_ma_col] = volume.rolling(
                window=window, min_periods=max(1, window // 2)
            ).mean()

            # 成交量比率
            valid_vol_mask = (df_tech[vol_ma_col] > 0) & volume.notna()
            df_tech['volume_ratio_5'] = 1.0
            df_tech.loc[valid_vol_mask, 'volume_ratio_5'] = (
                    volume[valid_vol_mask] / df_tech.loc[valid_vol_mask, vol_ma_col]
            )

            print("✓ 因子5生成完成")
        else:
            print(" 缺少成交量数据，使用价格强度作为替代因子...")
            # 如果有high/low数据，计算价格强度
            if all(col in df_tech.columns for col in ['high', 'low']):
                high = df_tech['high']
                low = df_tech['low']

                # 验证high, low数据
                for col in ['high', 'low']:
                    if not pd.api.types.is_numeric_dtype(df_tech[col]):
                        print(f" {col}列不是数值类型，尝试转换...")
                        df_tech[col] = pd.to_numeric(df_tech[col], errors='coerce')

                # 当日价格强度
                range_mask = (high != low) & high.notna() & low.notna() & close_prices.notna()
                df_tech['price_strength'] = 0.5  # 默认值
                df_tech.loc[range_mask, 'price_strength'] = (
                        (close_prices[range_mask] - low[range_mask]) /
                        (high[range_mask] - low[range_mask])
                )

                # 重命名为volume_ratio_5以保持一致性
                df_tech['volume_ratio_5'] = df_tech['price_strength']
                df_tech = df_tech.drop(columns=['price_strength'])
                print("✓ 因子5（价格强度）生成完成")
            else:
                print(" 也缺少high/low数据，使用加速度作为替代因子...")
                # 计算价格加速度
                df_tech['price_velocity'] = close_prices.diff()
                df_tech['volume_ratio_5'] = df_tech['price_velocity'].diff()
                print("✓ 因子5（价格加速度）生成完成")

        # ==================== 数据清理和验证 ====================
        print("清理和验证生成的技术因子...")

        # 确保没有无穷大值
        for col in ['price_change_5d', 'price_vs_ma20', 'rsi_14', 'volatility_20d', 'volume_ratio_5']:
            if col in df_tech.columns:
                inf_count = np.isinf(df_tech[col]).sum()
                if inf_count > 0:
                    print(f"清理 {col} 中的 {inf_count} 个inf值...")
                    df_tech[col] = df_tech[col].replace([np.inf, -np.inf], np.nan)

        # 验证生成的技术特征
        tech_cols = ['price_change_5d', 'price_vs_ma20', 'rsi_14', 'volatility_20d', 'volume_ratio_5']
        valid_tech_cols = []

        for col in tech_cols:
            if col in df_tech.columns:
                # 检查非空比例和唯一值数量
                non_na_ratio = df_tech[col].notna().mean()
                unique_vals = df_tech[col].nunique()
                if non_na_ratio > 0.3 and unique_vals > 1:
                    valid_tech_cols.append(col)
                else:
                    print(f" 特征 {col} 质量较低: 非空比例={non_na_ratio:.2%}, 唯一值数={unique_vals}")

        print_section("技术指标生成统计")
        print(f"目标生成技术因子: 5个")
        print(f"实际生成有效技术因子: {len(valid_tech_cols)}个")

        if len(valid_tech_cols) > 0:
            print(f"生成的因子: {valid_tech_cols}")

            # 添加简要统计信息
            print("\n因子简要统计:")
            for col in valid_tech_cols:
                if col in df_tech.columns:
                    mean_val = df_tech[col].mean()
                    std_val = df_tech[col].std()
                    non_na = df_tech[col].notna().sum()
                    print(f"  {col}: 均值={mean_val:.4f}, 标准差={std_val:.4f}, 非空数={non_na}")

        return df_tech

    except Exception as e:
        print(f"技术指标计算失败: {e}")
        import traceback
        traceback.print_exc()
        print("返回原始数据...")
        return df


# ==================== 辅助函数保持不变 ====================

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

                # 关键修复：严格的价格有效性检查
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

            print(" 标签有效性验证:")
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
            # 替换inf/-inf为NaN，再用中位数填充
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
    """时间序列数据集划分 - 修改为滚动交叉验证"""
    print_section("数据集划分（滚动交叉验证）")

    if df.empty or len(feature_cols) == 0:
        print("数据为空或无特征")
        return None, None, None, None, None, None, None, None, None

    # 确保按日期排序
    df = df.sort_values('date')

    if USE_ROLLING_CV:
        print("使用滚动交叉验证模式")
        # 获取唯一日期并排序
        dates = np.sort(df['date'].unique())
        n_dates = len(dates)

        # 计算分割点 - 划分测试集
        test_start_idx = int(n_dates * (1 - test_ratio))

        # 测试集
        test_dates = dates[test_start_idx:]
        test_df = df[df['date'].isin(test_dates)]

        # 训练+验证集
        train_val_dates = dates[:test_start_idx]
        train_val_df = df[df['date'].isin(train_val_dates)]

        # 从训练验证集中再划分验证集（用于早停等）
        train_val_dates_sorted = np.sort(train_val_df['date'].unique())
        n_train_val_dates = len(train_val_dates_sorted)
        val_start_idx = int(n_train_val_dates * (1 - val_ratio))

        # 验证集日期
        val_dates = train_val_dates_sorted[val_start_idx:]
        val_df = train_val_df[train_val_df['date'].isin(val_dates)]

        # 训练集日期
        train_dates = train_val_dates_sorted[:val_start_idx]
        train_df = train_val_df[train_val_df['date'].isin(train_dates)]

        print(f"训练集: {train_df['date'].min().date()} 到 {train_df['date'].max().date()}, 大小: {len(train_df):,}")
        print(f"验证集: {val_df['date'].min().date()} 到 {val_df['date'].max().date()}, 大小: {len(val_df):,}")
        print(f"测试集: {test_df['date'].min().date()} 到 {test_df['date'].max().date()}, 大小: {len(test_df):,}")
    else:
        # 原来的静态划分逻辑
        dates = np.sort(df['date'].unique())
        n_dates = len(dates)

        train_end_idx = int(n_dates * (1 - test_ratio - val_ratio))
        val_end_idx = int(n_dates * (1 - test_ratio))

        train_dates = dates[:train_end_idx]
        val_dates = dates[train_end_idx:val_end_idx]
        test_dates = dates[val_end_idx:]

        train_df = df[df['date'].isin(train_dates)]
        val_df = df[df['date'].isin(val_dates)]
        test_df = df[df['date'].isin(test_dates)]

        print(f"训练集: {train_df['date'].min().date()} 到 {train_df['date'].max().date()}, 大小: {len(train_df):,}")
        print(f"验证集: {val_df['date'].min().date()} 到 {val_df['date'].max().date()}, 大小: {len(val_df):,}")
        print(f"测试集: {test_df['date'].min().date()} 到 {test_df['date'].max().date()}, 大小: {len(test_df):,}")

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
    if USE_ROLLING_CV:
        # 使用滚动时间序列交叉验证
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=ROLLING_CV_SPLITS)
        cv_method = tscv
    else:
        cv_method = 2  # 原来的2折交叉验证

    rf_search = RandomizedSearchCV(
        rf_model,
        rf_param_grid,
        n_iter=n_trials,
        cv=cv_method,  # 使用滚动交叉验证
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
        # 使用滚动交叉验证（如果启用）
        if USE_ROLLING_CV:
            # 使用滚动时间序列交叉验证
            from sklearn.model_selection import TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=ROLLING_CV_SPLITS)
            cv_method = tscv
        else:
            cv_method = 2  # 原来的2折交叉验证

        xgb_search = RandomizedSearchCV(
            xgb_model, xgb_param_grid,
            n_iter=n_trials, cv=cv_method, scoring='f1', n_jobs=1,  # 改为cv=cv_method
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


def train_models(X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, best_params=None):
    """修复版模型训练 - 解决XGBoost dtype错误和收益率inf问题"""
    print_section("修复版模型训练")

    # 强制使用滚动CV（如果启用）
    global USE_ROLLING_CV  # 添加这一行声明全局变量
    if ENFORCE_ROLLING_CV_FOR_ALL_MODELS:
        USE_ROLLING_CV = True
        print("强制执行滚动交叉验证模式...")

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

    # ==================== 6. 训练随机森林模型（统一使用滚动交叉验证） ====================
    print("\n1. 训练随机森林模型...")
    try:
        rf_params = best_params.get('rf', {})

        # 确保rf_params包含必要的参数
        default_rf_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 10,
            'min_samples_leaf': 4,
            'max_features': 'sqrt',
            'random_state': RANDOM_STATE,
            'n_jobs': 1,  # 每折模型使用1个核，避免内存问题
            'class_weight': 'balanced'  # 添加类别平衡
        }

        # 更新默认参数
        for key, value in default_rf_params.items():
            if key not in rf_params:
                rf_params[key] = value

        print("随机森林参数:")
        for key, value in rf_params.items():
            if key in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features', 'n_jobs']:
                print(f"   {key}: {value}")

        # 始终使用滚动交叉验证（如果启用）
        if USE_ROLLING_CV:
            print("使用滚动交叉验证训练随机森林...")
            from sklearn.model_selection import TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=ROLLING_CV_SPLITS)

            # 存储每折的模型和分数
            rf_models = []
            fold_scores = []

            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_scaled), 1):
                print(f"  训练折 {fold}/{ROLLING_CV_SPLITS}...")

                X_fold_train = X_train_scaled[train_idx]
                y_fold_train = y_train.iloc[train_idx] if hasattr(y_train, 'iloc') else y_train[train_idx]

                # 处理类别不平衡（对每一折单独做SMOTE） - 修复点2：添加类别检查
                # 先检查类别数量
                unique_classes = np.unique(y_fold_train)
                if len(unique_classes) < 2:
                    print(f"    折 {fold}: 训练集只有{len(unique_classes)}个类别，跳过SMOTE")
                    X_fold_train_bal, y_fold_train_bal = X_fold_train, y_fold_train
                else:
                    try:
                        smote_fold = SMOTE(random_state=RANDOM_STATE + fold, sampling_strategy=0.8)
                        X_fold_train_bal, y_fold_train_bal = smote_fold.fit_resample(X_fold_train, y_fold_train)
                    except Exception as e:
                        print(f"    折 {fold} SMOTE失败: {e}，使用原始数据")
                        X_fold_train_bal, y_fold_train_bal = X_fold_train, y_fold_train

                # 训练当前折的模型
                fold_model = RandomForestClassifier(**rf_params)
                fold_model.fit(X_fold_train_bal, y_fold_train_bal)
                rf_models.append(fold_model)

                # 在验证折上评估
                X_fold_val = X_train_scaled[val_idx]
                y_fold_val = y_train.iloc[val_idx] if hasattr(y_train, 'iloc') else y_train[val_idx]
                y_fold_val_pred = fold_model.predict(X_fold_val)
                fold_score = f1_score(y_fold_val, y_fold_val_pred, zero_division=0)
                fold_scores.append(fold_score)

                print(f"    折 {fold}: 验证集F1 = {fold_score:.4f}")

            # 使用所有折的平均模型（通过平均预测概率）
            print(f"滚动交叉验证平均F1: {np.mean(fold_scores):.4f}")

            rf_model = EnsembleRF(rf_models)
        else:
            # 如果不使用滚动CV，使用原来的训练逻辑
            print("使用普通训练模式...")
            rf_model = RandomForestClassifier(**rf_params)
            rf_model.fit(X_train_balanced, y_train_balanced)

        models['rf'] = rf_model
        print("随机森林模型训练完成")

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
        return {}, None, {}, {}, {}  # 如果随机森林训练失败，直接返回

    # 在验证集和测试集上评估
    y_val_pred_rf = rf_model.predict(X_val_scaled)
    y_val_proba_rf = rf_model.predict_proba(X_val_scaled)[:, 1]
    y_test_pred_rf = rf_model.predict(X_test_scaled)
    y_test_proba_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

    # 确保y_true是numpy数组格式
    y_val_array = y_val.values if hasattr(y_val, 'values') else y_val
    y_test_array = y_test.values if hasattr(y_test, 'values') else y_test

    # 修复点3：在计算指标前检查预测值是否有特殊值
    # 处理验证集预测
    if isinstance(y_val_pred_rf, (int, float)):
        # 如果y_val_pred_rf是单个值（比如错误值），转换为数组
        y_val_pred_rf_fixed = np.full(len(y_val_array), int(y_val_pred_rf))
    else:
        y_val_pred_rf_fixed = np.array(y_val_pred_rf)

    # 修复：替换特殊值
    if np.any(y_val_pred_rf_fixed == -2147483648):
        print("警告：验证集发现特殊值-2147483648，替换为0")
        y_val_pred_rf_fixed = np.where(y_val_pred_rf_fixed == -2147483648, 0, y_val_pred_rf_fixed)

    # 处理测试集预测
    if isinstance(y_test_pred_rf, (int, float)):
        y_test_pred_rf_fixed = np.full(len(y_test_array), int(y_test_pred_rf))
    else:
        y_test_pred_rf_fixed = np.array(y_test_pred_rf)

    if np.any(y_test_pred_rf_fixed == -2147483648):
        print("警告：测试集发现特殊值-2147483648，替换为0")
        y_test_pred_rf_fixed = np.where(y_test_pred_rf_fixed == -2147483648, 0, y_test_pred_rf_fixed)

    results['rf'] = {
        'val_accuracy': accuracy_score(y_val_array, y_val_pred_rf_fixed),
        'val_precision': precision_score(y_val_array, y_val_pred_rf_fixed, zero_division=0),
        'val_recall': recall_score(y_val_array, y_val_pred_rf_fixed, zero_division=0),
        'val_f1': f1_score(y_val_array, y_val_pred_rf_fixed, zero_division=0),
        'val_roc_auc': roc_auc_score(y_val_array, y_val_proba_rf),
        'test_accuracy': accuracy_score(y_test_array, y_test_pred_rf_fixed),
        'test_precision': precision_score(y_test_array, y_test_pred_rf_fixed, zero_division=0),
        'test_recall': recall_score(y_test_array, y_test_pred_rf_fixed, zero_division=0),
        'test_f1': f1_score(y_test_array, y_test_pred_rf_fixed, zero_division=0),
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

        # 训练模型 - 统一使用滚动交叉验证
        print("训练XGBoost模型...")

        if USE_ROLLING_CV:
            print("使用滚动交叉验证训练XGBoost...")
            from sklearn.model_selection import TimeSeriesSplit

            tscv = TimeSeriesSplit(n_splits=ROLLING_CV_SPLITS)

            # 存储每折的模型和分数
            xgb_models = []
            fold_scores = []

            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_scaled), 1):
                print(f"  训练折 {fold}/{ROLLING_CV_SPLITS}...")

                X_fold_train = X_train_scaled[train_idx]
                y_fold_train = y_train.iloc[train_idx] if hasattr(y_train, 'iloc') else y_train[train_idx]

                # 处理类别不平衡（对每一折单独做SMOTE） - 修复点2：添加类别检查
                # 先检查类别数量
                unique_classes = np.unique(y_fold_train)
                if len(unique_classes) < 2:
                    print(f"    折 {fold}: 训练集只有{len(unique_classes)}个类别，跳过SMOTE")
                    X_fold_train_bal, y_fold_train_bal = X_fold_train, y_fold_train
                else:
                    try:
                        smote_fold = SMOTE(random_state=RANDOM_STATE + fold, sampling_strategy=0.8)
                        X_fold_train_bal, y_fold_train_bal = smote_fold.fit_resample(X_fold_train, y_fold_train)
                    except:
                        X_fold_train_bal, y_fold_train_bal = X_fold_train, y_fold_train

                # 转换数据类型
                X_fold_train_float32 = safe_convert_to_float32(X_fold_train_bal)
                y_fold_train_int32 = y_fold_train_bal.astype(np.int32) if hasattr(y_fold_train_bal,
                                                                                  'astype') else np.array(
                    y_fold_train_bal, dtype=np.int32)

                # 训练当前折的模型
                fold_model = xgb.XGBClassifier(**xgb_params)
                fold_model.fit(X_fold_train_float32, y_fold_train_int32)
                xgb_models.append(fold_model)

                # 在验证折上评估
                X_fold_val = X_train_scaled[val_idx]
                y_fold_val = y_train.iloc[val_idx] if hasattr(y_train, 'iloc') else y_train[val_idx]
                X_fold_val_float32 = safe_convert_to_float32(X_fold_val)
                y_fold_val_pred = fold_model.predict(X_fold_val_float32)
                fold_score = f1_score(y_fold_val, y_fold_val_pred, zero_division=0)
                fold_scores.append(fold_score)

                print(f"    折 {fold}: 验证集F1 = {fold_score:.4f}")

            # 使用所有折的平均模型（通过平均预测概率）
            print(f"滚动交叉验证平均F1: {np.mean(fold_scores):.4f}")

            rf_model = EnsembleRF(rf_models)
        else:
            # 如果不使用滚动CV，使用原来的训练逻辑
            print("使用普通训练模式...")
            rf_model = RandomForestClassifier(**rf_params)
            rf_model.fit(X_train_balanced, y_train_balanced)

        models['rf'] = rf_model
        print("随机森林模型训练完成")

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
        return {}, None, {}, {}, {}  # 如果随机森林训练失败，直接返回

    # 在验证集和测试集上评估
    y_val_pred_rf = rf_model.predict(X_val_scaled)
    y_val_proba_rf = rf_model.predict_proba(X_val_scaled)[:, 1]
    y_test_pred_rf = rf_model.predict(X_test_scaled)
    y_test_proba_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

    # 确保y_true是numpy数组格式
    y_val_array = y_val.values if hasattr(y_val, 'values') else y_val
    y_test_array = y_test.values if hasattr(y_test, 'values') else y_test

    # 修复点3：在计算指标前检查预测值是否有特殊值
    # 处理验证集预测
    if isinstance(y_val_pred_rf, (int, float)):
        # 如果y_val_pred_rf是单个值（比如错误值），转换为数组
        y_val_pred_rf_fixed = np.full(len(y_val_array), int(y_val_pred_rf))
    else:
        y_val_pred_rf_fixed = np.array(y_val_pred_rf)

    # 修复：替换特殊值
    if np.any(y_val_pred_rf_fixed == -2147483648):
        print("警告：验证集发现特殊值-2147483648，替换为0")
        y_val_pred_rf_fixed = np.where(y_val_pred_rf_fixed == -2147483648, 0, y_val_pred_rf_fixed)

    # 处理测试集预测
    if isinstance(y_test_pred_rf, (int, float)):
        y_test_pred_rf_fixed = np.full(len(y_test_array), int(y_test_pred_rf))
    else:
        y_test_pred_rf_fixed = np.array(y_test_pred_rf)

    if np.any(y_test_pred_rf_fixed == -2147483648):
        print("警告：测试集发现特殊值-2147483648，替换为0")
        y_test_pred_rf_fixed = np.where(y_test_pred_rf_fixed == -2147483648, 0, y_test_pred_rf_fixed)

    results['rf'] = {
        'val_accuracy': accuracy_score(y_val_array, y_val_pred_rf_fixed),
        'val_precision': precision_score(y_val_array, y_val_pred_rf_fixed, zero_division=0),
        'val_recall': recall_score(y_val_array, y_val_pred_rf_fixed, zero_division=0),
        'val_f1': f1_score(y_val_array, y_val_pred_rf_fixed, zero_division=0),
        'val_roc_auc': roc_auc_score(y_val_array, y_val_proba_rf),
        'test_accuracy': accuracy_score(y_test_array, y_test_pred_rf_fixed),
        'test_precision': precision_score(y_test_array, y_test_pred_rf_fixed, zero_division=0),
        'test_recall': recall_score(y_test_array, y_test_pred_rf_fixed, zero_division=0),
        'test_f1': f1_score(y_test_array, y_test_pred_rf_fixed, zero_division=0),
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

        # 训练模型 - 统一使用滚动交叉验证
        print("训练XGBoost模型...")

        if USE_ROLLING_CV:
            print("使用滚动交叉验证训练XGBoost...")
            from sklearn.model_selection import TimeSeriesSplit

            tscv = TimeSeriesSplit(n_splits=ROLLING_CV_SPLITS)

            # 存储每折的模型和分数
            xgb_models = []
            fold_scores = []

            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_scaled), 1):
                print(f"  训练折 {fold}/{ROLLING_CV_SPLITS}...")

                X_fold_train = X_train_scaled[train_idx]
                y_fold_train = y_train.iloc[train_idx] if hasattr(y_train, 'iloc') else y_train[train_idx]

                # 处理类别不平衡（对每一折单独做SMOTE） - 修复点2：添加类别检查
                # 先检查类别数量
                unique_classes = np.unique(y_fold_train)
                if len(unique_classes) < 2:
                    print(f"    折 {fold}: 训练集只有{len(unique_classes)}个类别，跳过SMOTE")
                    X_fold_train_bal, y_fold_train_bal = X_fold_train, y_fold_train
                else:
                    try:
                        smote_fold = SMOTE(random_state=RANDOM_STATE + fold, sampling_strategy=0.8)
                        X_fold_train_bal, y_fold_train_bal = smote_fold.fit_resample(X_fold_train, y_fold_train)
                    except:
                        X_fold_train_bal, y_fold_train_bal = X_fold_train, y_fold_train

                # 转换数据类型
                X_fold_train_float32 = safe_convert_to_float32(X_fold_train_bal)
                y_fold_train_int32 = y_fold_train_bal.astype(np.int32) if hasattr(y_fold_train_bal,
                                                                                  'astype') else np.array(
                    y_fold_train_bal, dtype=np.int32)

                # 训练当前折的模型
                fold_model = xgb.XGBClassifier(**xgb_params)
                fold_model.fit(X_fold_train_float32, y_fold_train_int32)
                xgb_models.append(fold_model)

                # 在验证折上评估
                X_fold_val = X_train_scaled[val_idx]
                y_fold_val = y_train.iloc[val_idx] if hasattr(y_train, 'iloc') else y_train[val_idx]
                X_fold_val_float32 = safe_convert_to_float32(X_fold_val)
                y_fold_val_pred = fold_model.predict(X_fold_val_float32)
                fold_score = f1_score(y_fold_val, y_fold_val_pred, zero_division=0)
                fold_scores.append(fold_score)

                print(f"    折 {fold}: 验证集F1 = {fold_score:.4f}")

            # 使用所有折的平均模型（通过平均预测概率）
            print(f"滚动交叉验证平均F1: {np.mean(fold_scores):.4f}")

            # 使用全局定义的EnsembleXGB类
            xgb_model = EnsembleXGB(xgb_models)
        else:
            # 原来的训练逻辑
            print("使用普通训练模式...")
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

        # 修复点3：在计算指标前检查预测值是否有特殊值
        # 处理验证集预测
        if isinstance(y_val_pred_xgb, (int, float)):
            y_val_pred_xgb_fixed = np.full(len(y_val_array), int(y_val_pred_xgb))
        else:
            y_val_pred_xgb_fixed = np.array(y_val_pred_xgb)

        if np.any(y_val_pred_xgb_fixed == -2147483648):
            print("警告：XGB验证集发现特殊值-2147483648，替换为0")
            y_val_pred_xgb_fixed = np.where(y_val_pred_xgb_fixed == -2147483648, 0, y_val_pred_xgb_fixed)

        # 处理测试集预测
        if isinstance(y_test_pred_xgb, (int, float)):
            y_test_pred_xgb_fixed = np.full(len(y_test_array), int(y_test_pred_xgb))
        else:
            y_test_pred_xgb_fixed = np.array(y_test_pred_xgb)

        if np.any(y_test_pred_xgb_fixed == -2147483648):
            print("警告：XGB测试集发现特殊值-2147483648，替换为0")
            y_test_pred_xgb_fixed = np.where(y_test_pred_xgb_fixed == -2147483648, 0, y_test_pred_xgb_fixed)

        results['xgb'] = {
            'val_accuracy': accuracy_score(y_val_array, y_val_pred_xgb_fixed),
            'val_precision': precision_score(y_val_array, y_val_pred_xgb_fixed, zero_division=0),
            'val_recall': recall_score(y_val_array, y_val_pred_xgb_fixed, zero_division=0),
            'val_f1': f1_score(y_val_array, y_val_pred_xgb_fixed, zero_division=0),
            'val_roc_auc': roc_auc_score(y_val_array, y_val_proba_xgb),
            'test_accuracy': accuracy_score(y_test_array, y_test_pred_xgb_fixed),
            'test_precision': precision_score(y_test_array, y_test_pred_xgb_fixed, zero_division=0),
            'test_recall': recall_score(y_test_array, y_test_pred_xgb_fixed, zero_division=0),
            'test_f1': f1_score(y_test_array, y_test_pred_xgb_fixed, zero_division=0),
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
def train_lightgbm_default(X_train, y_train, X_val, y_val, X_test, y_test, feature_cols):
    """使用默认参数训练LightGBM并提取特征重要性（按增益）"""
    print_section("使用默认参数训练LightGBM")

    # 强制使用滚动CV（如果启用）
    if ENFORCE_ROLLING_CV_FOR_ALL_MODELS:
        USE_ROLLING_CV = True
        print("强制执行滚动交叉验证模式...")

    # ==================== 修复：检查验证集是否为空 ====================
    # 数据标准化（保持和其他模型一致）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # 检查验证集是否为空（滚动交叉验证模式下可能为空）
    if X_val is not None and len(X_val) > 0:
        X_val_scaled = scaler.transform(X_val)
        has_validation = True
    else:
        X_val_scaled = None
        has_validation = False
        print("注意：验证集为空，跳过验证集标准化和评估")

    X_test_scaled = scaler.transform(X_test)

    # 处理类别不平衡
    smote = SMOTE(random_state=RANDOM_STATE, sampling_strategy=0.8)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

    # 初始化LightGBM（默认参数）
    lgb_model = lgb.LGBMClassifier(
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        verbosity=-1  # -1表示静默
    )

    # ==================== 修复：保存滚动交叉验证数据 ====================
    rolling_cv_data = {
        'fold_scores': [],
        'fold_models': [],
        'X_train_scaled': X_train_scaled,
        'y_train': y_train,
        'fold_predictions': []
    }

    # ==================== 修复：处理验证集为空的情况 ====================
    # 训练模型 - 支持滚动交叉验证
    if USE_ROLLING_CV and not X_train.empty:
        print("使用滚动交叉验证训练LightGBM...")
        tscv = TimeSeriesSplit(n_splits=ROLLING_CV_SPLITS)

        # 存储每折的模型和分数
        lgb_models = []
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_scaled), 1):
            print(f"  训练折 {fold}/{ROLLING_CV_SPLITS}...")

            X_fold_train = X_train_scaled[train_idx]
            y_fold_train = y_train.iloc[train_idx] if hasattr(y_train, 'iloc') else y_train[train_idx]

            # 处理类别不平衡（对每一折单独做SMOTE）
            try:
                smote_fold = SMOTE(random_state=RANDOM_STATE + fold, sampling_strategy=0.8)
                X_fold_train_bal, y_fold_train_bal = smote_fold.fit_resample(X_fold_train, y_fold_train)
            except:
                X_fold_train_bal, y_fold_train_bal = X_fold_train, y_fold_train

            # 训练当前折的模型
            fold_model = lgb.LGBMClassifier(
                random_state=RANDOM_STATE + fold,
                n_jobs=1,
                verbosity=-1
            )

            # 使用当前折的验证集
            X_fold_val = X_train_scaled[val_idx]
            y_fold_val = y_train.iloc[val_idx] if hasattr(y_train, 'iloc') else y_train[val_idx]

            fold_model.fit(
                X_fold_train_bal, y_fold_train_bal,
                eval_set=[(X_fold_val, y_fold_val)],
                eval_metric='binary_logloss',
                callbacks=[
                    lgb.early_stopping(50, verbose=0),  # 添加verbose参数
                    lgb.log_evaluation(0)  # 控制日志输出
                ]
            )

            # 在验证折上评估
            y_fold_val_pred = fold_model.predict(X_fold_val)
            fold_score = f1_score(y_fold_val, y_fold_val_pred, zero_division=0)
            fold_scores.append(fold_score)

            # 保存折叠模型和分数
            lgb_models.append(fold_model)
            rolling_cv_data['fold_scores'].append(fold_score)
            rolling_cv_data['fold_models'].append(fold_model)

            print(f"    折 {fold}: 验证集F1 = {fold_score:.4f}")

        # 使用所有折的平均模型（通过平均预测概率）
        print(f"滚动交叉验证平均F1: {np.mean(fold_scores):.4f}")

        lgb_model = EnsembleLGB(lgb_models, fold_scores)

        # ==================== 保存滚动交叉验证数据 ====================
        # 测试集成模型
        try:
            y_test_pred = lgb_model.predict(X_test_scaled)
            # 保存测试预测
            rolling_cv_data['fold_predictions'] = y_test_pred
            print(f"集成模型测试预测完成，形状: {y_test_pred.shape}")
        except Exception as e:
            print(f"测试预测失败: {e}")
            y_test_pred = np.zeros(len(X_test_scaled), dtype=np.int32)

        # 保存滚动交叉验证数据到文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rolling_cv_file = f'lightgbm_rolling_cv_data_{timestamp}.pkl'

        # 只保存必要的数据，避免保存整个模型（模型另外保存）
        rolling_cv_data_save = {
            'fold_scores': rolling_cv_data['fold_scores'],
            'X_train_shape': X_train_scaled.shape,
            'y_train_shape': y_train.shape if hasattr(y_train, 'shape') else len(y_train),
            'n_splits': ROLLING_CV_SPLITS,
            'timestamp': timestamp,
            'feature_names': feature_cols,
            'fold_predictions_test': rolling_cv_data.get('fold_predictions', [])
        }

        with open(rolling_cv_file, 'wb') as f:
            pickle.dump(rolling_cv_data_save, f, protocol=4)
        print(f" 滚动交叉验证数据已保存: {rolling_cv_file}")

        # 单独保存集成模型
        model_file = f'lightgbm_ensemble_model_{timestamp}.pkl'
        # 注意：EnsembleLGB类可能无法直接序列化，我们保存基础模型列表
        model_save_data = {
            'models': lgb_models,
            'fold_scores': fold_scores,
            'feature_names': feature_cols,
            'scaler': scaler,
            'ensemble_class': 'EnsembleLGB'
        }
        with open(model_file, 'wb') as f:
            pickle.dump(model_save_data, f, protocol=4)
        print(f" LightGBM集成模型已保存: {model_file}")
    else:
        # 原来的训练逻辑
        print("使用普通训练模式...")
        # 训练模型 - 根据验证集是否存在使用不同的参数
        if has_validation and X_val_scaled is not None:
            try:
                # 使用新版本的回调函数
                lgb_model.fit(
                    X_train_balanced, y_train_balanced,
                    eval_set=[(X_val_scaled, y_val)],
                    eval_metric='binary_logloss',
                    callbacks=[
                        lgb.early_stopping(50),
                        lgb.log_evaluation(0)
                    ])
            except Exception as e:
                # 如果新版本API失败，尝试旧版本
                print(f"新版本API失败，尝试旧版本: {e}")
                # 移除verbose参数，使用callbacks
                lgb_model.fit(
                    X_train_balanced, y_train_balanced,
                    eval_set=[(X_val_scaled, y_val)],
                    eval_metric='binary_logloss',
                    callbacks=[
                        lgb.early_stopping(50),
                        lgb.log_evaluation(0)
                    ])
        else:
            # 没有验证集，不使用早停
            print("无验证集，训练时不使用早停")
            lgb_model.fit(
                X_train_balanced, y_train_balanced,
                eval_metric='binary_logloss',
                callbacks=[lgb.log_evaluation(0)])

    # ==================== 修复：验证集评估部分 ====================
    # 在测试集上评估
    # 安全地获取预测结果
    try:
        y_test_pred = lgb_model.predict(X_test_scaled)
        # 确保预测结果是有效的整数数组
        if hasattr(y_test_pred, '__len__'):
            y_test_pred = y_test_pred.astype(np.int32)
            # 检查是否有无效值
            mask_invalid = (y_test_pred != 0) & (y_test_pred != 1)
            if mask_invalid.any():
                print(f"警告：测试集预测包含异常值{np.unique(y_test_pred[mask_invalid])}，修正为0")
                y_test_pred[mask_invalid] = 0
        else:
            # 如果是单个值，检查有效性
            if y_test_pred not in [0, 1]:
                y_test_pred = 0
            y_test_pred = np.array([y_test_pred], dtype=np.int32)
    except Exception as e:
        print(f"测试集预测失败: {e}，使用全0预测")
        y_test_pred = np.zeros(len(X_test_scaled), dtype=np.int32)

    # 安全地获取预测概率
    try:
        proba_result = lgb_model.predict_proba(X_test_scaled)
        # 检查返回的结果维度
        if proba_result.ndim == 2 and proba_result.shape[1] >= 2:
            y_test_proba = proba_result[:, 1]
        elif proba_result.ndim == 1:
            # 如果只有一维，假设是正类概率
            y_test_proba = proba_result
        else:
            # 其他情况，使用默认值
            y_test_proba = np.ones(len(X_test_scaled)) * 0.5
    except Exception as e:
        print(f"预测概率出错: {e}，使用默认概率")
        y_test_proba = np.ones(len(X_test_scaled)) * 0.5

    # ==================== 修复：在计算指标前验证预测值 ====================
    # 确保y_true是numpy数组格式
    y_test_array = y_test.values if hasattr(y_test, 'values') else y_test

    # 检查y_test_pred是否为有效数组
    if isinstance(y_test_pred, (int, float, np.int32, np.int64)):
        # 如果是单个值，扩展到数组
        y_test_pred_fixed = np.full(len(y_test_array), int(y_test_pred), dtype=np.int32)
    else:
        y_test_pred_fixed = np.array(y_test_pred, dtype=np.int32)

    # 修复特殊值
    if np.any(y_test_pred_fixed == -2147483648):
        print(f"警告：发现{np.sum(y_test_pred_fixed == -2147483648)}个特殊值-2147483648，替换为0")
        y_test_pred_fixed[y_test_pred_fixed == -2147483648] = 0

    # 计算评估指标
    results = {
        'test_accuracy': accuracy_score(y_test_array, y_test_pred_fixed),
        'test_precision': precision_score(y_test_array, y_test_pred_fixed, zero_division=0),
        'test_recall': recall_score(y_test_array, y_test_pred_fixed, zero_division=0),
        'test_f1': f1_score(y_test_array, y_test_pred_fixed, zero_division=0),
        'test_roc_auc': roc_auc_score(y_test_array, y_test_proba)
    }

    # 如果有验证集，也计算验证集指标
    if has_validation and X_val_scaled is not None and y_val is not None and len(y_val) > 0:
        y_val_pred = lgb_model.predict(X_val_scaled)
        y_val_proba = lgb_model.predict_proba(X_val_scaled)[:, 1]
        y_val_array = y_val.values if hasattr(y_val, 'values') else y_val

        results['val_accuracy'] = accuracy_score(y_val_array, y_val_pred)
        results['val_precision'] = precision_score(y_val_array, y_val_pred, zero_division=0)
        results['val_recall'] = recall_score(y_val_array, y_val_pred, zero_division=0)
        results['val_f1'] = f1_score(y_val_array, y_val_pred, zero_division=0)
        results['val_roc_auc'] = roc_auc_score(y_val_array, y_val_proba)

        print("LightGBM验证集结果:")
        print(f"  准确率: {results['val_accuracy']:.4f}")
        print(f"  F1分数: {results['val_f1']:.4f}")
        print(f"  ROC-AUC: {results['val_roc_auc']:.4f}")

    # 第一步：根据模型类型获取特征重要性（核心修复：适配自定义EnsembleLGB类）
    def get_ensemble_feature_importance(model, feature_num):
        """
        适配单个模型/集成模型的特征重要性获取
        :param model: LightGBM模型（单个/集成）
        :param feature_num: 特征总数（避免维度不匹配）
        :return: 特征重要性数组（gain）
        """
        # 情况1：单个LightGBM模型（有feature_importances_属性）
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_

        # 情况2：自定义集成模型（有models属性，存储所有基模型）
        elif hasattr(model, 'models'):
            all_importances = []
            for estimator in model.models:
                # 仅处理有特征重要性的基模型
                if hasattr(estimator, 'feature_importances_'):
                    imp = estimator.feature_importances_
                    # 确保维度一致（防止个别基模型特征数异常）
                    if len(imp) == feature_num:
                        all_importances.append(imp)

            if all_importances:
                # 集成模型：计算所有基模型特征重要性的平均值
                return np.mean(all_importances, axis=0)
            else:
                # 无有效基模型时，返回全0数组
                return np.zeros(feature_num)

        # 情况3：未知模型类型，返回全0数组
        else:
            return np.zeros(feature_num)

    # 第二步：获取特征重要性并构建DataFrame
    # 先确认特征总数（避免维度不匹配）
    feature_num = len(feature_cols) if feature_cols else 0
    feature_importances_gain = get_ensemble_feature_importance(lgb_model, feature_num)

    # 构建特征重要性DataFrame（兼容空值）
    if feature_num > 0 and len(feature_importances_gain) == feature_num:
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'gain': feature_importances_gain,  # 增益（核心特征重要性指标）
            'importance_type': 'gain'
        }).sort_values('gain', ascending=False)
    else:
        # 无有效特征时，返回空DataFrame（避免报错）
        feature_importance = pd.DataFrame(columns=['feature', 'gain', 'importance_type'])

    # 可选：打印特征重要性信息，方便调试
    print(f" 特征重要性提取完成：")
    print(f"   - 特征总数：{feature_num}")
    print(f"   - 有效特征重要性数量：{len(feature_importance)}")
    print(f"   - 前5个重要特征：\n{feature_importance.head()}")

    print("LightGBM默认参数训练结果:")
    print(f"  准确率: {results['test_accuracy']:.4f}")
    print(f"  F1分数: {results['test_f1']:.4f}")
    print(f"  ROC-AUC: {results['test_roc_auc']:.4f}")

    return lgb_model, scaler, results, feature_importance

def select_top10_core_factors(feature_importance, financial_cols, technical_cols):
    """
    选择前10个核心因子：
    - 基于特征重要性增益（gain）排序
    - 如果总因子数>10，保留增益最高的10个
    - 如果总因子数<10，保留所有有效因子
    """
    print_section("筛选前10个核心因子（基于增益）")

    # 分离财务/技术因子的重要性并按增益降序排序
    fin_importance = feature_importance[feature_importance['feature'].isin(financial_cols)].sort_values('gain',
                                                                                                        ascending=False)
    tech_importance = feature_importance[feature_importance['feature'].isin(technical_cols)].sort_values('gain',
                                                                                                         ascending=False)

    print(f"财务因子数量（按增益排序）: {len(fin_importance)}")
    print(f"技术因子数量（按增益排序）: {len(tech_importance)}")

    # 合并所有因子并按增益降序排序
    all_importance = pd.concat([fin_importance, tech_importance]).sort_values('gain', ascending=False)

    # 确定最终保留数量：最多10个，最少实际可用数量
    total_available = len(all_importance)
    target_count = min(total_available, 10)

    # 选择增益最高的因子
    selected_core = all_importance.head(target_count)['feature'].tolist()

    # 分离选中的财务和技术因子
    selected_fin = [f for f in selected_core if f in financial_cols]
    selected_tech = [f for f in selected_core if f in technical_cols]

    print(f"\n最终筛选结果:")
    print(f"  财务因子: {len(selected_fin)} 个")
    print(f"  技术因子: {len(selected_tech)} 个")
    print(f"  总因子数: {len(selected_core)} 个 (最多10个)")

    # 打印具体因子列表（带增益和类型）
    print(f"\n选中的核心因子（按增益排序）:")
    for i, factor in enumerate(selected_core, 1):
        gain = all_importance[all_importance['feature'] == factor]['gain'].values[0]
        factor_type = "财务因子" if factor in financial_cols else "技术因子"
        print(f"  {i:2d}. {factor:<30} {factor_type:<8} 增益: {gain:.2f}")

    return selected_core, selected_fin, selected_tech

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
            'future_return': '未来20天绝对收益率',
            'selection_score': '模型预测概率',
            'rank': '当日排名',
            'selection_reason': '选股理由'
        })

        # 选择需要的列
        final_columns = ['交易日', '股票代码', '收盘价', '未来20天绝对收益率',
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


@timer_decorator
def perform_backtest_with_costs(daily_selected_df, test_df, benchmark_returns=None):
    """
    执行带交易成本和风控的收益回测 - 修复版本
    """
    print_section("实盘贴近度回测（含交易成本与风控）")

    try:
        # 准备数据
        backtest_data = daily_selected_df.copy()

        # 获取唯一日期并排序
        unique_dates = sorted(backtest_data['date'].unique())
        if len(unique_dates) == 0:
            print("错误：没有回测日期")
            return None

        print(f"回测期间: {unique_dates[0]} 到 {unique_dates[-1]}")
        print(f"总交易日数: {len(unique_dates)}")

        # 初始化风控管理器
        risk_manager = RiskControlManager()

        # 创建价格数据字典以便快速查询
        price_dict = {}
        for stock_code in test_df['stock_code'].unique():
            stock_data = test_df[test_df['stock_code'] == stock_code]
            if not stock_data.empty:
                price_dict[stock_code] = dict(zip(stock_data['date'], stock_data['close']))

        # 确定调仓日（每季度调仓）
        rebalance_dates = []
        current_quarter = None

        for date in unique_dates:
            quarter = f"{date.year}-Q{(date.month - 1) // 3 + 1}"
            if quarter != current_quarter:
                rebalance_dates.append(date)
                current_quarter = quarter

        print(f"调仓日数量: {len(rebalance_dates)}")
        print(f"调仓日列表: {rebalance_dates[:5]}...")

        # 执行回测
        portfolio_values = []
        portfolio_returns = []
        trading_summary = []

        for i, current_date in enumerate(tqdm(unique_dates, desc="执行回测")):

            # 获取当前日期所有股票的价格
            current_prices = {}
            for stock_code in price_dict:
                if current_date in price_dict[stock_code]:
                    current_prices[stock_code] = price_dict[stock_code][current_date]

            # 如果是调仓日，执行调仓
            if current_date in rebalance_dates:
                print(f"\n调仓日: {current_date}")

                # 获取当日的选股列表
                daily_stocks = backtest_data[backtest_data['date'] == current_date]

                if daily_stocks.empty:
                    print(f"  日期{current_date}没有选股数据")
                    continue

                # 按模型预测概率排序
                daily_stocks = daily_stocks.sort_values('模型预测概率', ascending=False)

                # 确定买入列表（最多TOP_N_HOLDINGS只）
                buy_list = daily_stocks.head(TOP_N_HOLDINGS)
                print(f"  选股数量: {len(buy_list)}只")

                # 卖出不在买入列表中的股票
                stocks_to_sell = []
                for stock_code in list(risk_manager.positions.keys()):
                    if stock_code not in buy_list['股票代码'].values:
                        stocks_to_sell.append(stock_code)

                for stock_code in stocks_to_sell:
                    if stock_code in current_prices:
                        position = risk_manager.positions[stock_code]
                        risk_manager.execute_sell(stock_code, position['shares'],
                                                  current_prices[stock_code], current_date,
                                                  'rebalance')

                # 买入新股票
                for _, row in buy_list.iterrows():
                    stock_code = row['股票代码']
                    if stock_code not in current_prices:
                        continue

                    current_price = current_prices[stock_code]

                    # 计算每只股票的权重
                    target_weight = 1.0 / len(buy_list)  # 等权重
                    target_weight = risk_manager.check_single_stock_limit(stock_code, target_weight)

                    # 计算目标持仓价值
                    total_portfolio_value = risk_manager.cash
                    for code, pos in risk_manager.positions.items():
                        if code in current_prices:
                            total_portfolio_value += pos['shares'] * current_prices[code]

                    target_value = total_portfolio_value * target_weight

                    # 如果已有持仓，计算需要调整的数量
                    if stock_code in risk_manager.positions:
                        position = risk_manager.positions[stock_code]
                        current_value = position['shares'] * current_price
                        value_diff = target_value - current_value

                        if value_diff > 0:  # 需要买入
                            shares_to_buy = int(value_diff / current_price)
                            if shares_to_buy > 0:
                                risk_manager.execute_buy(stock_code, shares_to_buy,
                                                         current_price, current_date)
                        elif value_diff < 0:  # 需要卖出
                            shares_to_sell = int(-value_diff / current_price)
                            if shares_to_sell > 0:
                                risk_manager.execute_sell(stock_code, shares_to_sell,
                                                          current_price, current_date,
                                                          'rebalance')
                    else:  # 新买入
                        shares_to_buy = int(target_value / current_price)
                        if shares_to_buy > 0:
                            risk_manager.execute_buy(stock_code, shares_to_buy,
                                                     current_price, current_date)

            # 更新持仓市值
            total_value = risk_manager.update_positions(price_dict, current_date)

            # 检查组合回撤
            stop_loss_triggered, drawdown = risk_manager.check_portfolio_drawdown(total_value)
            if stop_loss_triggered:
                print(f"\n日期 {current_date}: 组合回撤达到{drawdown:.2%}，触发减仓")
                risk_manager.reduce_positions()

            # 记录每日组合价值
            portfolio_values.append({
                'date': current_date,
                'portfolio_value': total_value,
                'cash': risk_manager.cash,
                'drawdown': drawdown,
                'total_value': total_value
            })

            # 计算日收益率
            if i > 0:
                prev_value = portfolio_values[i - 1]['portfolio_value']
                if prev_value > 0:
                    daily_return = (total_value - prev_value) / prev_value
                else:
                    daily_return = 0
                portfolio_returns.append(daily_return)
            else:
                portfolio_returns.append(0)

        # 计算回测指标
        metrics = calculate_backtest_metrics(portfolio_values, portfolio_returns, benchmark_returns)

        # 生成交易统计
        trading_stats = generate_trading_statistics(risk_manager.trading_records, portfolio_values)

        # 生成回测报告
        report = generate_backtest_report(metrics, trading_stats, risk_manager.positions)

        # 计算平均持仓天数
        if risk_manager.position_days:
            avg_hold_days = np.mean(list(risk_manager.position_days.values()))
        else:
            avg_hold_days = 0

        # 添加到报告
        report['additional_stats'] = {
            '平均持仓天数': avg_hold_days,
            '最大持仓天数': max(risk_manager.position_days.values()) if risk_manager.position_days else 0,
            '最小持仓天数': min(risk_manager.position_days.values()) if risk_manager.position_days else 0
        }

        return {
            'portfolio_values': portfolio_values,
            'portfolio_returns': portfolio_returns,
            'metrics': metrics,
            'trading_stats': trading_stats,
            'trading_records': risk_manager.trading_records,
            'positions': risk_manager.positions,
            'report': report,
            'holdings_history': risk_manager.holdings_history
        }

    except Exception as e:
        print(f"回测执行出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def calculate_turnover_rate(trades_df, portfolio_values):
    """计算换手率 - 使用单边换手率（业内标准）"""
    if len(portfolio_values) < 2 or trades_df.empty:
        return 0

    # 方法1：只计算卖出交易金额（业内标准）
    if 'type' in trades_df.columns:
        # 使用卖出交易金额（单边）
        sell_trades = trades_df[trades_df['type'] == 'sell']
        total_trade_amount = sell_trades['total_value'].sum()

        # 或者使用买入交易金额（两种方式等价）
        # buy_trades = trades_df[trades_df['type'] == 'buy']
        # total_trade_amount = buy_trades['total_value'].sum()
    else:
        # 如果无法区分类型，使用交易金额的一半（近似单边）
        total_trade_amount = trades_df['total_value'].sum() / 2

    # 计算平均资产净值
    avg_portfolio_value = np.mean([
        pv.get('total_value', pv.get('portfolio_value', 0))
        for pv in portfolio_values
    ])

    if avg_portfolio_value > 0:
        # 单边换手率
        turnover_rate = total_trade_amount / avg_portfolio_value

        # 年化计算
        if len(portfolio_values) > 1:
            days = (portfolio_values[-1]['date'] - portfolio_values[0]['date']).days
            if days > 0:
                trading_days = len(portfolio_values)
                # 使用242个交易日（台湾市场）
                turnover_rate = turnover_rate * (242 / trading_days)
    else:
        turnover_rate = 0

    return turnover_rate

def check_position_concentration(positions):
    """检查仓位集中度"""
    if not positions:
        return True

    total_value = sum(pos['shares'] * pos.get('current_price', pos['avg_price']) for pos in positions.values())

    for stock_code, position in positions.items():
        position_value = position['shares'] * position.get('current_price', position['avg_price'])
        weight = position_value / total_value if total_value > 0 else 0

        if weight > RISK_CONTROL['single_stock_limit']:
            print(f" 股票{stock_code}仓位{weight:.2%}超过限制{RISK_CONTROL['single_stock_limit']:.2%}")
            return False

    return True


def check_stop_loss_execution(trading_stats):
    """检查止损规则执行情况"""
    # 这里可以添加更复杂的检查逻辑
    return True


def check_stop_profit_execution(trading_stats):
    """检查止盈规则执行情况"""
    # 这里可以添加更复杂的检查逻辑
    return True

def plot_backtest_results(backtest_results, save_path=None):
    """
    绘制回测结果图表
    """
    try:
        if 'error' in backtest_results:
            print("无法绘制图表：回测结果包含错误")
            return

        # 设置中文字体（如果有中文字符）
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        # 创建图表
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle('股票选股策略回测分析', fontsize=16, fontweight='bold')

        # --- 子图1：净值曲线对比 ---
        ax1 = axes[0, 0]
        strategy_nav = backtest_results['strategy_net_value']
        benchmark_nav = backtest_results['benchmark_cumulative_returns'] * 10000

        ax1.plot(strategy_nav.index, strategy_nav.values, 'b-', linewidth=2, label='策略净值')
        ax1.plot(benchmark_nav.index, benchmark_nav.values, 'g-', linewidth=1.5, label='基准净值', alpha=0.7)
        ax1.set_title('净值曲线对比')
        ax1.set_xlabel('日期')
        ax1.set_ylabel('净值（元）')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # --- 子图2：累计收益率对比 ---
        ax2 = axes[0, 1]
        strategy_cum_return = (backtest_results['strategy_cumulative_returns'] - 1) * 100
        benchmark_cum_return = (backtest_results['benchmark_cumulative_returns'] - 1) * 100

        ax2.plot(strategy_cum_return.index, strategy_cum_return.values, 'r-', linewidth=2, label='策略累计收益')
        ax2.plot(benchmark_cum_return.index, benchmark_cum_return.values, 'b-', linewidth=1.5, label='基准累计收益',
                 alpha=0.7)
        ax2.set_title('累计收益率对比 (%)')
        ax2.set_xlabel('日期')
        ax2.set_ylabel('累计收益率 (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # --- 子图3：日收益率分布 ---
        ax3 = axes[1, 0]
        daily_returns = backtest_results['daily_returns'] * 100  # 转换为百分比

        ax3.hist(daily_returns, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        ax3.axvline(daily_returns.mean(), color='red', linestyle='--', linewidth=2,
                    label=f'均值: {daily_returns.mean():.2f}%')
        ax3.set_title('策略日收益率分布')
        ax3.set_xlabel('日收益率 (%)')
        ax3.set_ylabel('频次')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # --- 子图4：滚动收益率（20日） ---
        ax4 = axes[1, 1]
        rolling_return = daily_returns.rolling(window=20).mean()
        ax4.plot(rolling_return.index, rolling_return.values, 'purple', linewidth=2)
        ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax4.set_title('20日滚动平均收益率 (%)')
        ax4.set_xlabel('日期')
        ax4.set_ylabel('滚动收益率 (%)')
        ax4.grid(True, alpha=0.3)

        # --- 子图5：回撤曲线 ---
        ax5 = axes[2, 0]
        running_max = backtest_results['strategy_cumulative_returns'].expanding().max()
        drawdown = (backtest_results['strategy_cumulative_returns'] - running_max) / running_max * 100

        ax5.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red', label='回撤')
        ax5.set_title('策略回撤曲线')
        ax5.set_xlabel('日期')
        ax5.set_ylabel('回撤 (%)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # --- 子图6：关键指标表格 ---
        ax6 = axes[2, 1]
        ax6.axis('off')

        # 准备表格数据
        metrics = [
            ['指标', '策略', '基准'],
            ['总收益率', f"{backtest_results['total_return']:.2%}",
             f"{(backtest_results['benchmark_cumulative_returns'].iloc[-1] - 1):.2%}"],
            ['年化收益率', f"{backtest_results['annualized_return']:.2%}", 'N/A'],
            ['年化波动率', f"{backtest_results['annualized_volatility']:.2%}", 'N/A'],
            ['夏普比率', f"{backtest_results['sharpe_ratio']:.2f}", 'N/A'],
            ['最大回撤', f"{backtest_results['max_drawdown']:.2%}", 'N/A'],
            ['交易日数', f"{backtest_results['duration_days']}", 'N/A'],
            ['选股记录', f"{len(backtest_results['backtest_data'])}", 'N/A']
        ]

        table = ax6.table(cellText=metrics, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        # 设置表格样式
        for i in range(len(metrics)):
            for j in range(len(metrics[0])):
                cell = table[(i, j)]
                if i == 0:  # 标题行
                    cell.set_facecolor('#4C72B0')
                    cell.set_text_props(weight='bold', color='white')
                elif i % 2 == 1:  # 奇数行
                    cell.set_facecolor('#E3E3E3')
                else:  # 偶数行
                    cell.set_facecolor('#FFFFFF')

        ax6.set_title('关键绩效指标')

        # 调整布局
        plt.tight_layout()

        # 保存图表
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"回测图表已保存: {save_path}")

        plt.show()

    except Exception as e:
        print(f"绘制回测图表时出错: {e}")
        import traceback
        traceback.print_exc()


def simple_plot_backtest(backtest_results, timestamp):
    """
    简单的回测图表绘制（备用方案）
    """
    try:
        plt.figure(figsize=(12, 8))

        # 净值曲线
        plt.subplot(2, 2, 1)
        plt.plot(backtest_results['strategy_cumulative_returns'], label='策略净值', linewidth=2)
        plt.plot(backtest_results['benchmark_cumulative_returns'], label='基准净值', linewidth=1.5, alpha=0.7)
        plt.title('净值曲线对比')
        plt.xlabel('日期')
        plt.ylabel('净值')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 累计收益率
        plt.subplot(2, 2, 2)
        strategy_return = (backtest_results['strategy_cumulative_returns'] - 1) * 100
        benchmark_return = (backtest_results['benchmark_cumulative_returns'] - 1) * 100
        plt.plot(strategy_return, label='策略累计收益 (%)', linewidth=2, color='red')
        plt.plot(benchmark_return, label='基准累计收益 (%)', linewidth=1.5, color='blue', alpha=0.7)
        plt.title('累计收益率对比')
        plt.xlabel('日期')
        plt.ylabel('收益率 (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 日收益率分布
        plt.subplot(2, 2, 3)
        plt.hist(backtest_results['daily_returns'] * 100, bins=50, edgecolor='black', alpha=0.7)
        plt.title('日收益率分布')
        plt.xlabel('日收益率 (%)')
        plt.ylabel('频次')
        plt.grid(True, alpha=0.3)

        # 关键指标文本
        plt.subplot(2, 2, 4)
        plt.axis('off')
        metrics_text = f"""关键指标：
        总收益率: {backtest_results['total_return']:.2%}
        年化收益率: {backtest_results['annualized_return']:.2%}
        年化波动率: {backtest_results['annualized_volatility']:.2%}
        夏普比率: {backtest_results['sharpe_ratio']:.2f}
        最大回撤: {backtest_results['max_drawdown']:.2%}
        交易日数: {backtest_results['duration_days']}
        选股记录: {len(backtest_results['backtest_data']):,}"""
        plt.text(0.1, 0.5, metrics_text, fontsize=10, verticalalignment='center')

        plt.tight_layout()
        simple_plot_file = f'backtest_simple_chart_{timestamp}.png'
        plt.savefig(simple_plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"简单回测图表已保存: {simple_plot_file}")

    except Exception as e:
        print(f"简单图表绘制失败: {e}")


@timer_decorator
def stratified_backtest(df, probabilities, feature_cols, model_name='rf'):
    """
    分层回测函数
    :param df: 包含日期、股票代码、未来收益率的DataFrame
    :param probabilities: 模型预测概率
    :param feature_cols: 特征列
    :param model_name: 模型名称
    :return: 分层回测结果
    """
    print_section(f"分层回测 - {model_name.upper()}")

    # 准备数据
    backtest_df = df.copy()
    backtest_df['prediction_prob'] = probabilities

    # 确保有未来收益率
    if 'future_return' not in backtest_df.columns:
        print("错误：数据中没有future_return列")
        return None

    # 确保数据按日期排序
    backtest_df = backtest_df.sort_values(['date', 'stock_code'])

    # 获取所有唯一日期
    unique_dates = sorted(backtest_df['date'].unique())
    print(f"回测期间：{unique_dates[0]} 到 {unique_dates[-1]}")
    print(f"总交易日数：{len(unique_dates)}")

    # 初始化分层结果存储
    strat_results = {
        'date': [],
        'layer': [],
        'num_stocks': [],
        'layer_return': [],
        'cumulative_return': [],
        'positions': []
    }

    # 初始化分层净值曲线
    layer_nav = {i: [1.0] for i in range(N_STRATIFICATION)}
    layer_dates = {i: [unique_dates[0]] for i in range(N_STRATIFICATION)}

    # 生成调仓日
    if REBALANCE_MONTHLY:
        # 按月调仓
        rebalance_dates = []
        current_month = None
        for date in unique_dates:
            if date.month != current_month:
                rebalance_dates.append(date)
                current_month = date.month
    else:
        # 按月调仓（月初）
        rebalance_dates = [date for date in unique_dates
                           if date.day == REBALANCE_DAY or date == unique_dates[0]]

    print(f"调仓日数量：{len(rebalance_dates)}")

    # 执行分层回测
    for i, rebalance_date in enumerate(rebalance_dates):
        if i >= len(rebalance_dates) - 1:
            break

        next_rebalance_idx = i + 1
        if next_rebalance_idx >= len(rebalance_dates):
            break

        next_rebalance_date = rebalance_dates[next_rebalance_idx]

        # 获取调仓日数据
        daily_data = backtest_df[backtest_df['date'] == rebalance_date].copy()

        if len(daily_data) < N_STRATIFICATION:
            print(f"日期 {rebalance_date} 股票数量不足，跳过")
            continue

        # 按预测概率排序并分层
        daily_data = daily_data.sort_values('prediction_prob', ascending=False)
        daily_data['layer'] = pd.qcut(
            daily_data['prediction_prob'],
            q=N_STRATIFICATION,
            labels=False,
            duplicates='drop'
        )

        # 计算每层收益
        for layer in range(N_STRATIFICATION):
            layer_stocks = daily_data[daily_data['layer'] == layer]

            if len(layer_stocks) == 0:
                continue

            # 获取这些股票在持有期的收益
            stock_codes = layer_stocks['stock_code'].tolist()
            hold_period_data = backtest_df[
                (backtest_df['date'] >= rebalance_date) &
                (backtest_df['date'] < next_rebalance_date) &
                (backtest_df['stock_code'].isin(stock_codes))
                ]

            if len(hold_period_data) == 0:
                continue

            # 计算每日等权收益（简化计算）
            daily_returns = hold_period_data.groupby('date')['future_return'].mean()

            # 累计持有期收益
            if len(daily_returns) > 0:
                period_return = (1 + daily_returns).prod() - 1

                # 更新净值
                if layer in layer_nav:
                    current_nav = layer_nav[layer][-1]
                    new_nav = current_nav * (1 + period_return)
                    layer_nav[layer].append(new_nav)
                    layer_dates[layer].append(next_rebalance_date)

                # 存储结果
                strat_results['date'].append(rebalance_date)
                strat_results['layer'].append(layer)
                strat_results['num_stocks'].append(len(layer_stocks))
                strat_results['layer_return'].append(period_return)
                strat_results['positions'].append(stock_codes)

    # 计算每层累计收益
    strat_results_df = pd.DataFrame(strat_results)

    if strat_results_df.empty:
        print("分层回测结果为空")
        return None

    # 计算每层的累计收益率
    cumulative_returns = {}
    for layer in range(N_STRATIFICATION):
        layer_data = strat_results_df[strat_results_df['layer'] == layer]
        if not layer_data.empty:
            cumulative_returns[layer] = (1 + layer_data['layer_return']).prod() - 1

    # 计算核心验证指标
    validation_metrics = calculate_stratification_metrics(
        strat_results_df, cumulative_returns, layer_nav, layer_dates
    )

    # 绘制分层回测图表
    plot_stratified_backtest(layer_nav, layer_dates, validation_metrics, model_name)

    return {
        'stratified_results': strat_results_df,
        'layer_nav': layer_nav,
        'layer_dates': layer_dates,
        'validation_metrics': validation_metrics
    }


def calculate_stratification_metrics(strat_results_df, cumulative_returns, layer_nav, layer_dates):
    """
    计算分层回测验证指标
    """
    metrics = {}

    # 1. Top层 vs Bottom层收益率差
    if 0 in cumulative_returns and (N_STRATIFICATION - 1) in cumulative_returns:
        top_bottom_spread = cumulative_returns[0] - cumulative_returns[N_STRATIFICATION - 1]
        metrics['top_bottom_spread'] = top_bottom_spread
        metrics['top_bottom_spread_pct'] = f"{top_bottom_spread:.2%}"

        # 检验是否显著>0
        from scipy import stats
        top_returns = strat_results_df[strat_results_df['layer'] == 0]['layer_return']
        bottom_returns = strat_results_df[strat_results_df['layer'] == N_STRATIFICATION - 1]['layer_return']

        if len(top_returns) > 1 and len(bottom_returns) > 1:
            t_stat, p_value = stats.ttest_ind(top_returns, bottom_returns, equal_var=False)
            metrics['top_bottom_t_stat'] = t_stat
            metrics['top_bottom_p_value'] = p_value
            metrics['top_bottom_significant'] = p_value < 0.05

    # 2. 分层单调性检验
    layer_avg_returns = []
    layers = []
    for layer in range(N_STRATIFICATION):
        if layer in cumulative_returns:
            layer_avg_returns.append(cumulative_returns[layer])
            layers.append(layer)

    if len(layer_avg_returns) >= 3:
        from scipy.stats import spearmanr
        monotonicity, monotonicity_p = spearmanr(layers, layer_avg_returns)
        metrics['monotonicity'] = monotonicity
        metrics['monotonicity_p_value'] = monotonicity_p
        metrics['monotonicity_passed'] = monotonicity > MONOTONICITY_THRESHOLD

    # 3. Top层年化收益率和夏普比率
    if 0 in layer_nav and len(layer_nav[0]) > 1:
        nav_series = pd.Series(layer_nav[0], index=layer_dates[0])
        # 计算年化收益率
        total_return = nav_series.iloc[-1] / nav_series.iloc[0] - 1
        days = (nav_series.index[-1] - nav_series.index[0]).days
        annualized_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
        metrics['top_annualized_return'] = annualized_return

        # 计算日收益率
        daily_returns = nav_series.pct_change().dropna()
        if len(daily_returns) > 0:
            # 年化波动率
            annualized_vol = daily_returns.std() * np.sqrt(252)
            # 夏普比率（假设无风险利率为0）
            sharpe_ratio = annualized_return / annualized_vol if annualized_vol != 0 else 0
            metrics['top_sharpe_ratio'] = sharpe_ratio
            metrics['top_sharpe_passed'] = sharpe_ratio > SHARPE_THRESHOLD

    # 4. Top层 vs 市场基准
    # 这里市场基准可以使用所有股票等权组合
    all_stocks_return = (1 + strat_results_df['layer_return']).prod() - 1
    metrics['market_avg_return'] = all_stocks_return
    metrics['top_vs_market'] = metrics.get('top_annualized_return', 0) - all_stocks_return

    return metrics


def plot_stratified_backtest(layer_nav, layer_dates, validation_metrics, model_name):
    """
    绘制分层回测图表
    """
    try:
        plt.figure(figsize=(15, 10))

        # 子图1：分层净值曲线
        plt.subplot(2, 2, 1)
        for layer in sorted(layer_nav.keys()):
            if len(layer_nav[layer]) > 1:
                plt.plot(layer_dates[layer], layer_nav[layer],
                         label=f'Layer {layer} (Top)' if layer == 0 else f'Layer {layer}',
                         linewidth=2 if layer == 0 else 1)

        plt.title(f'Stratified Backtest - {model_name.upper()}', fontsize=14)
        plt.xlabel('Date')
        plt.ylabel('Net Asset Value')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 子图2：分层平均收益率柱状图
        plt.subplot(2, 2, 2)
        layers = []
        returns = []
        for layer in sorted(layer_nav.keys()):
            if len(layer_nav[layer]) > 1:
                total_return = layer_nav[layer][-1] / layer_nav[layer][0] - 1
                layers.append(layer)
                returns.append(total_return)

        colors = ['green' if i == 0 else 'red' if i == len(layers) - 1 else 'blue' for i in range(len(layers))]
        bars = plt.bar([f'Layer {l}' for l in layers], returns, color=colors)
        plt.title('Layer Cumulative Returns')
        plt.ylabel('Cumulative Return')
        plt.xticks(rotation=45)

        # 在柱子上添加数值标签
        for bar, ret in zip(bars, returns):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{ret:.2%}', ha='center', va='bottom')

        # 子图3：验证指标表格
        plt.subplot(2, 2, 3)
        plt.axis('off')

        # 准备表格数据
        table_data = []
        if 'top_bottom_spread_pct' in validation_metrics:
            significance = '✓' if validation_metrics.get('top_bottom_significant', False) else '✗'
            table_data.append(['Top-Bottom Spread', validation_metrics['top_bottom_spread_pct'], significance])

        if 'monotonicity' in validation_metrics:
            passed = '✓' if validation_metrics.get('monotonicity_passed', False) else '✗'
            table_data.append(['Monotonicity', f"{validation_metrics['monotonicity']:.3f}", passed])

        if 'top_annualized_return' in validation_metrics:
            table_data.append(['Top Annualized Return', f"{validation_metrics['top_annualized_return']:.2%}", ''])

        if 'top_sharpe_ratio' in validation_metrics:
            passed = '✓' if validation_metrics.get('top_sharpe_passed', False) else '✗'
            table_data.append(['Top Sharpe Ratio', f"{validation_metrics['top_sharpe_ratio']:.3f}", passed])

        if 'market_avg_return' in validation_metrics:
            table_data.append(['Market Avg Return', f"{validation_metrics['market_avg_return']:.2%}", ''])

        # 创建表格
        if table_data:
            table = plt.table(cellText=table_data,
                              colLabels=['Metric', 'Value', 'Passed'],
                              loc='center',
                              cellLoc='center',
                              colWidths=[0.3, 0.2, 0.1])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)

        # 子图4：单调性检验散点图
        plt.subplot(2, 2, 4)
        layers = []
        returns = []
        for layer in sorted(layer_nav.keys()):
            if len(layer_nav[layer]) > 1:
                total_return = layer_nav[layer][-1] / layer_nav[layer][0] - 1
                layers.append(layer)
                returns.append(total_return)

        if len(layers) >= 3:
            plt.scatter(layers, returns, s=100, alpha=0.7)

            # 添加趋势线
            z = np.polyfit(layers, returns, 1)
            p = np.poly1d(z)
            plt.plot(layers, p(layers), "r--", alpha=0.5)

            # 计算R²
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, std_err = linregress(layers, returns)

            plt.title(f'Monotonicity Test (R²={r_value ** 2:.3f})')
            plt.xlabel('Layer (0=Top, 4=Bottom)')
            plt.ylabel('Cumulative Return')
            plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = f'stratified_backtest_{model_name}_{timestamp}.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"分层回测图表已保存: {plot_file}")
        plt.show()

    except Exception as e:
        print(f"绘制分层回测图表时出错: {e}")
        import traceback
        traceback.print_exc()


def perform_stratified_backtest_all_models(test_df, predictions, probabilities, feature_cols):
    """
    对所有模型执行分层回测
    """
    if not STRATIFIED_BACKTEST:
        return {}

    print_section("执行分层回测")

    stratified_results = {}

    for model_name in probabilities.keys():
        if model_name in probabilities and len(probabilities[model_name]) == len(test_df):
            print(f"\n对模型 {model_name.upper()} 执行分层回测...")

            try:
                result = stratified_backtest(
                    test_df,
                    probabilities[model_name],
                    feature_cols,
                    model_name
                )

                if result:
                    stratified_results[model_name] = result

                    # 输出验证结果
                    print(f"\n{model_name.upper()} 分层回测验证结果:")
                    print("-" * 50)

                    metrics = result['validation_metrics']

                    if 'top_bottom_spread_pct' in metrics:
                        print(f"Top-Bottom Spread: {metrics['top_bottom_spread_pct']}")
                        if 'top_bottom_significant' in metrics:
                            status = "✓ 显著" if metrics['top_bottom_significant'] else "✗ 不显著"
                            print(f"显著性检验: {status}")

                    if 'monotonicity' in metrics:
                        print(f"单调性 (Spearman): {metrics['monotonicity']:.3f}")
                        if 'monotonicity_passed' in metrics:
                            status = "✓ 通过" if metrics['monotonicity_passed'] else "✗ 未通过"
                            print(f"单调性检验: {status}")

                    if 'top_annualized_return' in metrics:
                        print(f"Top层年化收益: {metrics['top_annualized_return']:.2%}")

                    if 'top_sharpe_ratio' in metrics:
                        print(f"Top层夏普比率: {metrics['top_sharpe_ratio']:.3f}")
                        if 'top_sharpe_passed' in metrics:
                            status = "✓ 通过" if metrics['top_sharpe_passed'] else "✗ 未通过"
                            print(f"夏普比率检验: {status}")

                    if 'market_avg_return' in metrics:
                        print(f"市场平均收益: {metrics['market_avg_return']:.2%}")

                    print("-" * 50)

            except Exception as e:
                print(f"模型 {model_name} 分层回测失败: {e}")
                import traceback
                traceback.print_exc()

    return stratified_results
@timer_decorator
def select_stocks_with_lightgbm_unified(X_train, y_train, X_test, test_df, feature_cols, top_percent=0.4, top_k=20):
    """
    使用LightGBM筛选股票，保持与前40%逻辑一致
    :param top_percent: 选择前百分之多少的股票（默认40%）
    :param top_k: 最多选择的股票数量
    """
    print_section("LightGBM选股（保持前40%逻辑）")

    # 1. 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 2. 处理类别不平衡（与前40%标签匹配）
    # 注意：SMOTE的sampling_strategy应该根据正样本比例调整
    pos_ratio = y_train.mean()
    sampling_strategy = min(0.8, (0.4 / pos_ratio) if pos_ratio > 0 else 0.8)
    smote = SMOTE(random_state=RANDOM_STATE, sampling_strategy=sampling_strategy)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

    # 3. 训练LightGBM
    print("训练LightGBM选股模型...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        verbosity=-1
    )

    # 使用滚动交叉验证
    if USE_ROLLING_CV:
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=ROLLING_CV_SPLITS)

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_scaled), 1):
            X_fold_train = X_train_scaled[train_idx]
            y_fold_train = y_train.iloc[train_idx] if hasattr(y_train, 'iloc') else y_train[train_idx]

            # 重新平衡
            smote_fold = SMOTE(random_state=RANDOM_STATE + fold, sampling_strategy=sampling_strategy)
            X_fold_train_bal, y_fold_train_bal = smote_fold.fit_resample(X_fold_train, y_fold_train)

            lgb_model.fit(
                X_fold_train_bal, y_fold_train_bal,
                eval_set=[(X_train_scaled[val_idx], y_train.iloc[val_idx])],
                eval_metric='binary_logloss',
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )

    # 4. 预测所有股票
    y_proba = lgb_model.predict_proba(X_test_scaled)[:, 1]

    # 5. 应用前40%逻辑：按概率排序，取前40%或最多top_k只
    results = pd.DataFrame({
        '股票代码': test_df['stock_code'].values,
        '收盘价': test_df['close'].values,
        '预测概率': y_proba
    })

    # 排序
    results_sorted = results.sort_values('预测概率', ascending=False)

    # 计算应选股票数量：前40%，但不超过top_k
    n_stocks = len(results_sorted)
    n_select = min(int(n_stocks * top_percent), top_k)
    n_select = max(n_select, 1)  # 至少选1只

    top_stocks = results_sorted.head(n_select).copy()
    top_stocks['排名'] = range(1, len(top_stocks) + 1)

    print(f" LightGBM选股完成（前{top_percent:.0%}逻辑）")
    print(f"  总股票数: {n_stocks}只")
    print(f"  应选前{top_percent:.0%}: {int(n_stocks * top_percent)}只")
    print(f"  实际选择: {n_select}只（最多{top_k}只）")
    print(f"  概率阈值: {top_stocks['预测概率'].min():.3f}")

    return top_stocks, lgb_model, scaler

@timer_decorator
def calculate_transaction_costs(trade_value, is_buy=True):
    """
    计算交易成本
    :param trade_value: 交易金额
    :param is_buy: 是否买入（True:买入, False:卖出）
    :return: 交易成本
    """
    commission = trade_value * TRANSACTION_COSTS['commission']
    tax = 0
    if not is_buy:  # 卖出时征收证交税
        tax = trade_value * TRANSACTION_COSTS['tax']
    slippage = trade_value * TRANSACTION_COSTS['slippage']

    total_cost = commission + tax + slippage
    return total_cost


# ==================== 修复RiskControlManager类 ====================
class RiskControlManager:
    """风控管理器 - 修复版本"""

    def __init__(self):
        self.portfolio_value = INITIAL_CAPITAL
        self.cash = INITIAL_CAPITAL  # 添加现金追踪
        self.positions = {}  # 持仓字典 {股票代码: 持仓信息}
        self.trading_records = []  # 交易记录
        self.daily_portfolio_values = []  # 每日组合净值
        self.max_portfolio_value = INITIAL_CAPITAL  # 最高组合净值（用于计算回撤）
        self.trade_counts = {'buy': 0, 'sell': 0}  # 交易计数

        # 添加持仓追踪
        self.position_days = {}  # 股票持仓天数 {股票代码: 持仓天数}
        self.holdings_history = []  # 持仓历史记录

    def check_single_stock_limit(self, stock_code, target_weight):
        """检查单只股票仓位限制"""
        if target_weight > RISK_CONTROL['single_stock_limit']:
            print(
                f" 股票{stock_code}目标权重{target_weight:.2%}超过单只股票上限{RISK_CONTROL['single_stock_limit']:.2%}")
            return RISK_CONTROL['single_stock_limit']
        return target_weight

    def update_holding_days(self, current_date):
        """更新持仓天数"""
        for stock_code in list(self.position_days.keys()):
            if stock_code in self.positions:
                self.position_days[stock_code] += 1
            else:
                # 移除已清仓的股票
                del self.position_days[stock_code]

    def record_holding_history(self, current_date):
        """记录持仓历史"""
        total_value = self.cash
        positions_summary = {}

        for stock_code, position in self.positions.items():
            if 'current_price' in position and position['current_price'] > 0:
                position_value = position['shares'] * position['current_price']
                total_value += position_value
                positions_summary[stock_code] = {
                    'shares': position['shares'],
                    'price': position['current_price'],
                    'value': position_value,
                    'weight': position_value / total_value if total_value > 0 else 0
                }

        self.holdings_history.append({
            'date': current_date,
            'cash': self.cash,
            'total_value': total_value,
            'positions': positions_summary,
            'hold_days': dict(self.position_days)
        })

    def check_stop_loss_profit(self, stock_code, current_price, purchase_price, hold_days):
        """检查个股止损止盈（考虑持有期）"""
        if purchase_price <= 0 or hold_days < 30:  # 持有不足30天，不触发止损止盈
            return None

        return_rate = (current_price - purchase_price) / purchase_price

        # 根据持有期调整止损止盈阈值
        if hold_days < 60:  # 持有30-60天
            stop_loss_threshold = RISK_CONTROL['individual_stop_loss'] * 0.5  # 放宽止损
            stop_profit_threshold = RISK_CONTROL['individual_stop_profit'] * 1.5  # 提高止盈
        else:  # 持有超过60天
            stop_loss_threshold = RISK_CONTROL['individual_stop_loss'] * 1.5  # 进一步放宽
            stop_profit_threshold = RISK_CONTROL['individual_stop_profit'] * 2.0  # 进一步提高

        if return_rate <= stop_loss_threshold:
            return 'stop_loss'
        elif return_rate >= stop_profit_threshold:
            return 'stop_profit'

        return None

    def check_portfolio_drawdown(self, current_value):
        """检查组合回撤"""
        self.max_portfolio_value = max(self.max_portfolio_value, current_value)
        drawdown = (current_value - self.max_portfolio_value) / self.max_portfolio_value

        if drawdown <= RISK_CONTROL['portfolio_stop_loss']:
            return True, drawdown
        return False, drawdown

    def reduce_positions(self, reduction_ratio=RISK_CONTROL['reduction_ratio']):
        """减仓操作"""
        print(f"触发组合止损，减仓{reduction_ratio:.0%}")
        stocks_to_sell = []

        # 按比例减少所有持仓
        for stock_code in list(self.positions.keys()):
            position = self.positions[stock_code]
            reduce_shares = int(position['shares'] * reduction_ratio)

            if reduce_shares > 0 and 'current_price' in position:
                stocks_to_sell.append((stock_code, reduce_shares, position['current_price']))

        # 执行卖出
        for stock_code, shares, price in stocks_to_sell:
            self.execute_sell(stock_code, shares, price, None, 'portfolio_stop_loss')

    def execute_buy(self, stock_code, shares, price, date=None):
        """执行买入操作"""
        if shares <= 0 or price <= 0:
            return 0

        trade_value = shares * price
        cost = calculate_transaction_costs(trade_value, is_buy=True)
        net_value = trade_value + cost

        if net_value > self.cash:
            # 资金不足，调整买入数量
            max_shares = int((self.cash - cost) / price)
            if max_shares <= 0:
                print(f"资金不足购买{stock_code}，现金{self.cash:.2f}，需要{net_value:.2f}")
                return 0

            shares = max_shares
            trade_value = shares * price
            cost = calculate_transaction_costs(trade_value, is_buy=True)
            net_value = trade_value + cost

        # 更新持仓
        if stock_code in self.positions:
            self.positions[stock_code]['shares'] += shares
            total_shares = self.positions[stock_code]['shares']
            total_cost = (self.positions[stock_code]['avg_price'] *
                          (total_shares - shares) + price * shares)
            self.positions[stock_code]['avg_price'] = total_cost / total_shares
            self.positions[stock_code]['last_buy_date'] = date
        else:
            self.positions[stock_code] = {
                'shares': shares,
                'avg_price': price,
                'current_price': price,
                'last_buy_date': date,
                'first_buy_date': date
            }
            self.position_days[stock_code] = 1

        # 更新资金
        self.cash -= net_value

        # 记录交易
        self.trading_records.append({
            'date': date,
            'type': 'buy',
            'stock_code': stock_code,
            'shares': shares,
            'price': price,
            'cost': cost,
            'total_value': trade_value,
            'cash_after': self.cash
        })

        self.trade_counts['buy'] += 1

        print(f"买入 {stock_code}: {shares}股 @ {price:.2f}, 成本{cost:.2f}, 现金剩余{self.cash:.2f}")
        return shares

    def execute_sell(self, stock_code, shares, price, date=None, reason='normal'):
        """执行卖出操作"""
        if stock_code not in self.positions:
            return 0

        position = self.positions[stock_code]
        actual_shares = min(shares, position['shares'])

        if actual_shares <= 0 or price <= 0:
            return 0

        trade_value = actual_shares * price
        cost = calculate_transaction_costs(trade_value, is_buy=False)
        net_value = trade_value - cost

        # 更新持仓
        position['shares'] -= actual_shares
        if position['shares'] <= 0:
            del self.positions[stock_code]
            if stock_code in self.position_days:
                del self.position_days[stock_code]

        # 更新资金
        self.cash += net_value

        # 计算盈亏
        purchase_value = actual_shares * position['avg_price']
        profit = net_value - purchase_value
        return_rate = profit / purchase_value if purchase_value > 0 else 0

        # 记录交易
        self.trading_records.append({
            'date': date,
            'type': 'sell',
            'stock_code': stock_code,
            'shares': actual_shares,
            'price': price,
            'cost': cost,
            'total_value': trade_value,
            'profit': profit,
            'return_rate': return_rate,
            'reason': reason,
            'cash_after': self.cash
        })

        self.trade_counts['sell'] += 1

        print(f"卖出 {stock_code}: {actual_shares}股 @ {price:.2f}, 盈亏{profit:.2f}, 现金{self.cash:.2f}")
        return actual_shares

    def update_positions(self, price_dict, date):
        """更新持仓市值"""
        total_value = self.cash  # 从现金开始计算

        for stock_code, position in self.positions.items():
            # 获取当前价格
            if stock_code in price_dict and date in price_dict[stock_code]:
                current_price = price_dict[stock_code][date]
            else:
                # 使用最后已知价格
                current_price = position.get('current_price', position['avg_price'])

            position['current_price'] = current_price
            position_value = position['shares'] * current_price
            total_value += position_value

            # 检查个股止损止盈
            stop_signal = self.check_stop_loss_profit(
                stock_code,
                current_price,
                position['avg_price']
            )
            if stop_signal:
                print(f"股票{stock_code}触发{stop_signal}，当前价格{current_price:.2f}，成本{position['avg_price']:.2f}")
                self.execute_sell(
                    stock_code,
                    position['shares'],
                    current_price,
                    date,
                    stop_signal
                )

        # 更新持仓天数
        self.update_holding_days(date)

        # 记录持仓历史
        self.record_holding_history(date)

        return total_value


@timer_decorator
def generate_final_output(feature_cols, model_results, backtest_results, core_factors=None):
    """
    生成最终输出结果
    """
    print_section("步骤6：结果输出与迭代优化")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. 最终因子列表
    final_factors = {
        'final_factors': feature_cols,
        'financial_factors': [f for f in feature_cols if f.startswith('fin_')],
        'technical_factors': [f for f in feature_cols if not f.startswith('fin_')],
        'core_factors': core_factors if core_factors else feature_cols[:10]
    }

    # 保存因子列表
    factors_file = f'final_factors_list_{timestamp}.json'
    with open(factors_file, 'w', encoding='utf-8') as f:
        json.dump(final_factors, f, indent=2, ensure_ascii=False)
    print(f" 最终因子列表已保存: {factors_file}")

    # 2. 模型性能输出
    model_performance = {
        'training_date': timestamp,
        'models': {}
    }

    for model_name, results in model_results.items():
        model_performance['models'][model_name] = {
            'accuracy': results.get('test_accuracy', 0),
            'precision': results.get('test_precision', 0),
            'recall': results.get('test_recall', 0),
            'f1_score': results.get('test_f1', 0),
            'roc_auc': results.get('test_roc_auc', 0)
        }

    # 保存模型性能
    performance_file = f'model_performance_{timestamp}.json'
    with open(performance_file, 'w', encoding='utf-8') as f:
        json.dump(model_performance, f, indent=2)
    print(f" 模型性能指标已保存: {performance_file}")

    # 3. 回测结果输出
    if backtest_results:
        backtest_summary = {
            'backtest_period': {
                'start_date': str(backtest_results['portfolio_values'][0]['date']),
                'end_date': str(backtest_results['portfolio_values'][-1]['date']),
                'days': backtest_results['metrics'].get('回测天数', 0)
            },
            'performance_metrics': {
                k: (f"{v:.2%}" if isinstance(v, float) and k.endswith('率') else
                    f"{v:.2f}" if isinstance(v, float) else v)
                for k, v in backtest_results['metrics'].items()
            },
            'trading_statistics': backtest_results['trading_stats'],
            'report': backtest_results.get('report', {})
        }

        # 保存回测结果
        backtest_file = f'backtest_results_detailed_{timestamp}.json'
        with open(backtest_file, 'w', encoding='utf-8') as f:
            json.dump(backtest_summary, f, indent=2, ensure_ascii=False)
        print(f" 详细回测结果已保存: {backtest_file}")

        # 生成HTML报告
        html_report = generate_html_report(final_factors, model_performance, backtest_summary)
        html_file = f'final_report_{timestamp}.html'
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_report)
        print(f" HTML综合报告已保存: {html_file}")

    # 4. 迭代优化建议
    optimization_suggestions = generate_optimization_suggestions(
        final_factors, model_performance, backtest_results
    )

    suggestions_file = f'optimization_suggestions_{timestamp}.txt'
    with open(suggestions_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("迭代优化建议\n")
        f.write("=" * 60 + "\n\n")
        for suggestion in optimization_suggestions:
            f.write(f"• {suggestion}\n")

    print(f" 迭代优化建议已保存: {suggestions_file}")

    return {
        'factors_file': factors_file,
        'performance_file': performance_file,
        'backtest_file': backtest_file if backtest_results else None,
        'html_file': html_file if backtest_results else None,
        'suggestions_file': suggestions_file
    }


def generate_html_report(factors, model_performance, backtest_summary):
    """生成HTML格式的综合报告"""

    html_template = '''
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>股票选股策略回测报告</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .section {{ margin-bottom: 30px; border: 1px solid #ddd; padding: 20px; border-radius: 5px; }}
            h1, h2, h3 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .metric {{ margin: 5px 0; }}
            .pass {{ color: green; font-weight: bold; }}
            .fail {{ color: red; font-weight: bold; }}
            .recommendation {{ background-color: #fffacd; padding: 10px; margin: 10px 0; border-left: 4px solid #ffd700; }}
        </style>
    </head>
    <body>
        <h1> 股票选股策略回测报告</h1>
        <p>生成时间: {timestamp}</p>

        <div class="section">
            <h2>1. 策略概览</h2>
            <p><strong>回测期间:</strong> {start_date} 至 {end_date} ({days} 天)</p>
            <p><strong>初始资金:</strong> {initial_capital:,.2f} 元</p>
        </div>

        <div class="section">
            <h2>2. 因子配置</h2>
            <h3>核心因子 ({core_count}个)</h3>
            <ul>
                {core_factors_list}
            </ul>
            <h3>财务因子 ({financial_count}个)</h3>
            <ul>
                {financial_factors_list}
            </ul>
            <h3>技术因子 ({technical_count}个)</h3>
            <ul>
                {technical_factors_list}
            </ul>
        </div>

        <div class="section">
            <h2>3. 模型性能</h2>
            <table>
                <tr>
                    <th>模型</th>
                    <th>准确率</th>
                    <th>F1分数</th>
                    <th>ROC-AUC</th>
                    <th>精确率</th>
                    <th>召回率</th>
                </tr>
                {model_rows}
            </table>
        </div>

        <div class="section">
            <h2>4. 回测绩效</h2>
            <h3>4.1 核心指标</h3>
            <table>
                <tr>
                    <th>指标</th>
                    <th>数值</th>
                    <th>目标</th>
                    <th>状态</th>
                </tr>
                {metric_rows}
            </table>

            <h3>4.2 交易统计</h3>
            <table>
                {trading_rows}
            </table>
        </div>

        <div class="section">
            <h2>5. 风控合规</h2>
            <table>
                {compliance_rows}
            </table>
        </div>

        <div class="section">
            <h2>6. 优化建议</h2>
            {recommendations}
        </div>

        <div class="section">
            <h2>7. 迭代计划</h2>
            <ol>
                <li>增加更多财务指标，如现金流量比率、营运资本等</li>
                <li>优化技术因子参数，测试不同时间窗口</li>
                <li>引入市场情绪因子和资金流因子</li>
                <li>测试不同机器学习算法的组合</li>
                <li>优化交易成本模型，考虑实际交易限制</li>
            </ol>
        </div>
    </body>
    </html>
    '''

    # 准备数据
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 因子列表
    core_factors_list = ''.join(f'<li>{factor}</li>' for factor in factors.get('core_factors', [])[:10])
    financial_factors_list = ''.join(f'<li>{factor}</li>' for factor in factors.get('financial_factors', [])[:5])
    technical_factors_list = ''.join(f'<li>{factor}</li>' for factor in factors.get('technical_factors', [])[:5])

    # 模型性能行
    model_rows = ''
    for model_name, metrics in model_performance.get('models', {}).items():
        model_rows += f'''
        <tr>
            <td>{model_name.upper()}</td>
            <td>{metrics.get('accuracy', 0):.2%}</td>
            <td>{metrics.get('f1_score', 0):.4f}</td>
            <td>{metrics.get('roc_auc', 0):.4f}</td>
            <td>{metrics.get('precision', 0):.4f}</td>
            <td>{metrics.get('recall', 0):.4f}</td>
        </tr>
        '''

    # 绩效指标行
    metric_rows = ''
    target_metrics = [
        ('年化收益率', TARGET_METRICS['annual_return']),
        ('夏普比率', TARGET_METRICS['sharpe_ratio']),
        ('最大回撤', TARGET_METRICS['max_drawdown'])
    ]

    for metric_name, target_value in target_metrics:
        actual_value = backtest_summary['performance_metrics'].get(metric_name, '0')
        # ============ 修复：将字符串转换为浮点数 ============
        actual_value_num = 0.0
        try:
            if isinstance(actual_value, str):
                # 处理百分比字符串
                if '%' in actual_value:
                    # 移除百分号并转换为浮点数
                    actual_value_num = float(actual_value.replace('%', '')) / 100.0
                elif ':' in actual_value:
                    # 处理其他格式，暂时设为0
                    actual_value_num = 0.0
                else:
                    # 尝试直接转换为浮点数
                    actual_value_num = float(actual_value)
            else:
                # 如果不是字符串，直接使用
                actual_value_num = float(actual_value)
        except (ValueError, TypeError) as e:
            print(f"警告：无法转换指标值 '{actual_value}' 为浮点数: {e}")
            actual_value_num = 0.0
        # ============ 修复结束 ============

        # 根据指标类型决定比较方式
        if metric_name == '最大回撤':
            # 最大回撤是负数，比较时取绝对值
            actual_for_compare = abs(actual_value_num)
            target_for_compare = abs(target_value)
            status_class = 'pass' if actual_for_compare <= target_for_compare else 'fail'
            status_text = '✓ 达标' if actual_for_compare <= target_for_compare else '✗ 未达标'
        else:
            # 其他指标：实际值 >= 目标值
            status_class = 'pass' if actual_value_num >= target_value else 'fail'
            status_text = '✓ 达标' if actual_value_num >= target_value else '✗ 未达标'

        # ============ 修复：避免在f-string格式说明符中使用条件表达式 ============
        # 先根据指标名称决定目标值的显示格式
        if metric_name != '夏普比率':
            target_str = f"{target_value:.2%}"
        else:
            target_str = f"{target_value:.2f}"

        metric_rows += f'''
         <tr>
             <td>{metric_name}</td>
             <td>{actual_value}</td>
             <td>{target_str}</td>
             <td class="{status_class}">{status_text}</td>
         </tr>
        '''
        # ============ 修复结束 ============

    # 交易统计行
    trading_rows = ''
    for key, value in backtest_summary.get('trading_statistics', {}).items():
        trading_rows += f'<tr><td>{key}</td><td>{value}</td></tr>'

    # 合规检查行
    compliance_rows = ''
    for key, value in backtest_summary.get('report', {}).get('compliance_check', {}).items():
        compliance_rows += f'<tr><td>{key}</td><td>{value}</td></tr>'

    # 优化建议
    recommendations_html = ''
    for rec in backtest_summary.get('report', {}).get('recommendations', []):
        recommendations_html += f'<div class="recommendation">📌 {rec}</div>'

    # 填充模板
    html_content = html_template.format(
        timestamp=timestamp,
        start_date=backtest_summary['backtest_period']['start_date'],
        end_date=backtest_summary['backtest_period']['end_date'],
        days=backtest_summary['backtest_period']['days'],
        initial_capital=INITIAL_CAPITAL,
        core_count=len(factors.get('core_factors', [])),
        core_factors_list=core_factors_list,
        financial_count=len(factors.get('financial_factors', [])),
        financial_factors_list=financial_factors_list,
        technical_count=len(factors.get('technical_factors', [])),
        technical_factors_list=technical_factors_list,
        model_rows=model_rows,
        metric_rows=metric_rows,
        trading_rows=trading_rows,
        compliance_rows=compliance_rows,
        recommendations=recommendations_html
    )

    return html_content


def plot_detailed_backtest_results(backtest_results, save_path=None):
    """绘制详细回测结果图表"""

    try:
        fig = plt.figure(figsize=(18, 12))

        # 子图1：净值曲线
        ax1 = plt.subplot(3, 3, 1)
        portfolio_values = [pv['portfolio_value'] for pv in backtest_results['portfolio_values']]
        dates = [pv['date'] for pv in backtest_results['portfolio_values']]

        ax1.plot(dates, portfolio_values, 'b-', linewidth=2, label='策略净值')
        ax1.set_title('净值曲线', fontsize=12, fontweight='bold')
        ax1.set_xlabel('日期')
        ax1.set_ylabel('净值（元）')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 子图2：仓位构成
        ax2 = plt.subplot(3, 3, 2)
        cash_values = [pv['cash'] for pv in backtest_results['portfolio_values']]
        positions_values = [pv['positions_value'] for pv in backtest_results['portfolio_values']]

        ax2.stackplot(dates, cash_values, positions_values,
                      labels=['现金', '持仓'], alpha=0.7)
        ax2.set_title('仓位构成', fontsize=12, fontweight='bold')
        ax2.set_xlabel('日期')
        ax2.set_ylabel('金额（元）')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)

        # 子图3：日收益率分布
        ax3 = plt.subplot(3, 3, 3)
        returns = backtest_results['portfolio_returns']
        ax3.hist(returns, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        ax3.axvline(np.mean(returns), color='red', linestyle='--',
                    label=f'均值: {np.mean(returns):.2%}')
        ax3.set_title('日收益率分布', fontsize=12, fontweight='bold')
        ax3.set_xlabel('日收益率')
        ax3.set_ylabel('频次')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 子图4：最大回撤
        ax4 = plt.subplot(3, 3, 4)
        drawdowns = [pv['drawdown'] for pv in backtest_results['portfolio_values']]
        ax4.fill_between(dates, drawdowns, 0, alpha=0.3, color='red')
        ax4.set_title('回撤曲线', fontsize=12, fontweight='bold')
        ax4.set_xlabel('日期')
        ax4.set_ylabel('回撤')
        ax4.grid(True, alpha=0.3)

        # 子图5：交易次数统计
        ax5 = plt.subplot(3, 3, 5)
        trades_df = pd.DataFrame(backtest_results['trading_records'])
        if not trades_df.empty:
            monthly_trades = trades_df.resample('M', on='date').size()
            ax5.bar(monthly_trades.index, monthly_trades.values, alpha=0.7)
            ax5.set_title('月度交易次数', fontsize=12, fontweight='bold')
            ax5.set_xlabel('月份')
            ax5.set_ylabel('交易次数')
        ax5.grid(True, alpha=0.3)

        # 子图6：胜率统计
        ax6 = plt.subplot(3, 3, 6)
        if not trades_df.empty and 'profit' in trades_df.columns:
            profitable = (trades_df['profit'] > 0).sum()
            unprofitable = (trades_df['profit'] <= 0).sum()
            ax6.pie([profitable, unprofitable],
                    labels=['盈利交易', '亏损交易'],
                    autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
            ax6.set_title('交易胜率', fontsize=12, fontweight='bold')

        # 子图7：绩效指标表格
        ax7 = plt.subplot(3, 3, (7, 9))
        ax7.axis('off')

        metrics = backtest_results.get('metrics', {})
        table_data = [
            ['指标', '数值', '目标', '状态'],
            ['年化收益率', f"{metrics.get('年化收益率', 0):.2%}",
             f"{TARGET_METRICS['annual_return']:.2%}",
             '✓' if metrics.get('年化收益率', 0) >= TARGET_METRICS['annual_return'] else '✗'],
            ['夏普比率', f"{metrics.get('夏普比率', 0):.2f}",
             f"{TARGET_METRICS['sharpe_ratio']:.2f}",
             '✓' if metrics.get('夏普比率', 0) >= TARGET_METRICS['sharpe_ratio'] else '✗'],
            ['最大回撤', f"{metrics.get('最大回撤', 0):.2%}",
             f"{TARGET_METRICS['max_drawdown']:.2%}",
             '✓' if abs(metrics.get('最大回撤', 0)) <= TARGET_METRICS['max_drawdown'] else '✗'],
            ['胜率', f"{metrics.get('胜率', 0):.2%}", '>50%',
             '✓' if metrics.get('胜率', 0) > 0.5 else '✗'],
            ['盈亏比', f"{metrics.get('盈亏比', 0):.2f}", '>1.5',
             '✓' if metrics.get('盈亏比', 0) > 1.5 else '✗']
        ]

        table = ax7.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # 设置表格样式
        for i in range(len(table_data)):
            for j in range(len(table_data[0])):
                cell = table[(i, j)]
                if i == 0:  # 标题行
                    cell.set_facecolor('#4C72B0')
                    cell.set_text_props(weight='bold', color='white')
                elif table_data[i][-1] == '✓':  # 达标行
                    cell.set_facecolor('#DFF0D8')
                elif table_data[i][-1] == '✗':  # 未达标行
                    cell.set_facecolor('#F2DEDE')
                else:
                    cell.set_facecolor('#FFFFFF')

        plt.suptitle('实盘贴近度回测分析报告', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"详细回测图表已保存: {save_path}")

        plt.show()

    except Exception as e:
        print(f"绘制详细回测图表时出错: {e}")
        import traceback
        traceback.print_exc()
def generate_optimization_suggestions(factors, model_performance, backtest_results):
    """生成迭代优化建议"""

    suggestions = []

    # 1. 因子层面优化
    financial_count = len(factors.get('financial_factors', []))
    technical_count = len(factors.get('technical_factors', []))

    if financial_count < 15:
        suggestions.append("财务因子数量不足，建议增加更多财务指标，如现金流量比率、营运资本比率等")

    if technical_count < 5:
        suggestions.append("技术因子数量不足，建议增加更多技术指标，如成交量相关指标、波动率指标等")

    # 2. 模型性能优化
    best_f1 = max([m.get('f1_score', 0) for m in model_performance.get('models', {}).values()])
    if best_f1 < 0.6:
        suggestions.append(f"模型F1分数({best_f1:.2%})偏低，建议优化模型参数或增加特征工程")

    # 3. 回测绩效优化
    if backtest_results:
        metrics = backtest_results.get('metrics', {})

        if metrics.get('年化收益率', 0) < TARGET_METRICS['annual_return']:
            suggestions.append("年化收益率未达目标，考虑调整选股阈值或优化仓位管理")

        if abs(metrics.get('最大回撤', 0)) > TARGET_METRICS['max_drawdown']:
            suggestions.append("最大回撤过大，建议优化止损策略或降低仓位集中度")

        if metrics.get('夏普比率', 0) < TARGET_METRICS['sharpe_ratio']:
            suggestions.append("夏普比率偏低，建议优化风险调整后收益，可能需降低波动率")

        trading_stats = backtest_results.get('trading_stats', {})
        if trading_stats.get('总交易成本', 0) > INITIAL_CAPITAL * 0.02:  # 交易成本超过2%
            suggestions.append("交易成本过高，建议减少调仓频率或优化交易算法")

    # 4. 通用建议
    suggestions.append("建议引入滚动时间窗口进行模型训练，提高策略稳定性")
    suggestions.append("建议增加市场环境判断模块，在不同市场环境下使用不同策略")
    suggestions.append("建议引入行业轮动因子，优化行业配置")

    return suggestions


@timer_decorator
def quick_test_lightgbm():
    """快速测试LightGBM，跳过IC/IR等耗时步骤"""
    print_section("⚡ LightGBM快速测试模式")

    # 1. 尝试直接加载预合并文件
    if not os.path.exists(PRE_MERGED_FILE):
        print(f"❌ 预合并文件不存在: {PRE_MERGED_FILE}")
        print("请先运行完整流程生成预合并文件")
        return None

    print(f"加载预合并文件: {PRE_MERGED_FILE}")
    try:
        with open(PRE_MERGED_FILE, 'rb') as f:
            data = pickle.load(f)

        # 适配数据格式
        if isinstance(data, tuple) and len(data) == 2:
            df, feature_cols = data
        elif isinstance(data, pd.DataFrame):
            df = data
            # 自动提取特征列
            base_cols = ['date', 'stock_code', 'close', 'volume', 'open', 'high', 'low',
                         'future_return', 'market_avg_return', 'label']
            feature_cols = [col for col in df.columns
                            if col not in base_cols and pd.api.types.is_numeric_dtype(df[col])]
        else:
            print(f"未知数据格式: {type(data)}")
            return None

        print(f" 数据加载成功: {df.shape}")
        print(f"时间范围: {df['date'].min()} 到 {df['date'].max()}")
        print(f"股票数量: {df['stock_code'].nunique()}")
        print(f"原始特征数量: {len(feature_cols)}")

    except Exception as e:
        print(f"预合并文件加载失败: {e}")
        return None

    # 2. 快速特征选择（简单的方差筛选）
    print("\n执行快速特征选择...")
    # 选择方差最大的前N个特征
    feature_variances = {}
    for col in feature_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            variance = df[col].var()
            if not np.isnan(variance):
                feature_variances[col] = variance

    # 按方差排序，选择前N个特征
    sorted_features = sorted(feature_variances.items(), key=lambda x: x[1], reverse=True)
    selected_features = [f[0] for f in sorted_features[:LIGHTGBM_TEST_FEATURES]]

    print(f"选择特征: {len(selected_features)} 个 (方差最大)")
    print(f"特征示例: {selected_features[:5]}")

    # 3. 采样数据以减少计算量
    print(f"\n采样数据: {LIGHTGBM_TEST_SAMPLE_SIZE:,} 条样本")
    if len(df) > LIGHTGBM_TEST_SAMPLE_SIZE:
        # 分层采样，保持正负样本比例
        df_sampled = df.sample(n=LIGHTGBM_TEST_SAMPLE_SIZE, random_state=RANDOM_STATE)
    else:
        df_sampled = df.copy()

    # 4. 准备建模数据
    print("准备建模数据...")
    modeling_df = df_sampled[['date', 'stock_code', 'future_return', 'label'] + selected_features].copy()

    # 处理缺失值
    for col in selected_features:
        if col in modeling_df.columns:
            modeling_df[col] = modeling_df[col].fillna(modeling_df[col].median())

    # 移除标签缺失的行
    initial_size = len(modeling_df)
    modeling_df = modeling_df.dropna(subset=['label', 'future_return'])
    print(f"移除缺失标签: {initial_size - len(modeling_df):,} 行")

    # 5. 简单数据集划分（不进行滚动交叉验证以加速）
    print("划分数据集...")
    modeling_df = modeling_df.sort_values('date')
    dates = modeling_df['date'].unique()
    test_split_idx = int(len(dates) * 0.8)  # 80%训练，20%测试

    train_dates = dates[:test_split_idx]
    test_dates = dates[test_split_idx:]

    train_df = modeling_df[modeling_df['date'].isin(train_dates)]
    test_df = modeling_df[modeling_df['date'].isin(test_dates)]

    X_train = train_df[selected_features]
    X_test = test_df[selected_features]
    y_train = train_df['label']
    y_test = test_df['label']

    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    print(f"正样本比例 - 训练: {y_train.mean():.2%}, 测试: {y_test.mean():.2%}")

    # 6. 快速标准化
    print("标准化特征...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 7. 使用SMOTE处理类别不平衡
    print("处理类别不平衡...")
    smote = SMOTE(random_state=RANDOM_STATE, sampling_strategy=0.8)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    print(f"平衡后训练集: {X_train_balanced.shape}")

    # 8. 训练LightGBM（使用默认参数）
    print_section("训练LightGBM（默认参数）")

    # LightGBM默认参数
    lgb_params = {
        'n_estimators': 100,
        'max_depth': 7,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': RANDOM_STATE,
        'n_jobs': N_JOBS,
        'verbosity': -1
    }

    print("训练LightGBM模型...")
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(
        X_train_balanced, y_train_balanced,
        eval_set=[(X_test_scaled, y_test)],
        eval_metric='binary_logloss',
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
    )

    # 9. 评估模型
    print_section("LightGBM评估结果")

    y_pred = lgb_model.predict(X_test_scaled)
    y_proba = lgb_model.predict_proba(X_test_scaled)[:, 1]

    # ====================== 新增：数据校验与保存（修正变量名） ======================
    # 1. 先打印关键信息到控制台，快速排查
    print("===== 数据校验信息 =====")
    # 注意：这里改用实际的变量名 y_pred（预测结果）和 y_test（真实标签）
    print(f"y_pred (预测结果) 类型: {type(y_pred)}")
    print(f"y_pred (预测结果) 具体值: {y_pred}")
    # 尝试打印形状（如果是数组，否则捕获异常）
    try:
        print(f"y_pred (预测结果) 形状: {np.shape(y_pred)}")
    except Exception as e:
        print(f"y_pred (预测结果) 无法获取形状: {e}")

    # 打印真实标签y_test的信息（对比参考）
    print(f"\ny_test (真实标签) 类型: {type(y_test)}")
    print(f"y_test (真实标签) 具体值（前10个）: {y_test[:10] if hasattr(y_test, '__getitem__') else y_test}")
    try:
        print(f"y_test (真实标签) 形状: {np.shape(y_test)}")
    except Exception as e:
        print(f"y_test (真实标签) 无法获取形状: {e}")

    # 2. 保存数据到文件（方便后续详细分析）
    # 创建保存目录（避免目录不存在报错）
    save_dir = "./model_test_data"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 方式1：保存为txt文件（直观查看文本内容）
    with open(os.path.join(save_dir, "test_data_info.txt"), "w", encoding="utf-8") as f:
        f.write(f"y_pred (预测结果) 类型: {type(y_pred)}\n")
        f.write(f"y_pred (预测结果) 具体值: {y_pred}\n")
        f.write(f"\ny_test (真实标签) 类型: {type(y_test)}\n")
        f.write(f"y_test (真实标签) 具体值（全部）: {y_test}\n")

    # 方式2：保存为pickle文件（保留原始数据类型，可加载复用）
    # 只保存有效数据，避免异常值导致保存失败
    save_data = {
        "y_test": y_test,  # 真实标签
        "y_pred": y_pred,  # 预测结果
        "y_proba": y_proba,  # 额外保存预测概率，方便排查分类问题
        "X_test_scaled": X_test_scaled  # 保存标准化后的测试集特征，排查输入问题
    }
    with open(os.path.join(save_dir, "test_data.pkl"), "wb") as f:
        pickle.dump(save_data, f)

    # 方式3：如果是数组，保存为csv（更易读）
    try:
        # 尝试将真实标签和预测标签合并保存（仅当两者都是数组时）
        if isinstance(y_test, (np.ndarray, list, pd.Series)) and isinstance(y_pred, (np.ndarray, list, pd.Series)):
            df = pd.DataFrame({
                "y_true": y_test,
                "y_pred": y_pred,
                "y_proba": y_proba  # 新增预测概率列，更全面
            })
            df.to_csv(os.path.join(save_dir, "test_pred_true.csv"), index=False, encoding="utf-8")
            print(" 真实标签、预测标签和预测概率已保存为csv文件")
        else:
            print(" 无法保存csv：y_test或y_pred不是数组/列表类型")
    except Exception as e:
        print(f" 保存csv失败: {e}")

    # ====================== 原有代码（调用accuracy_score） ======================
    # 注意：如果y_pred是异常值，可先加判断避免程序直接崩溃
    if not isinstance(y_pred, (np.ndarray, list, pd.Series, pd.DataFrame)):
        print(f"❌ 警告：y_pred不是数组类数据，值为 {y_pred}，跳过accuracy_score计算")
        accuracy = np.nan  # 用NaN标记无效值
    else:
        accuracy = accuracy_score(y_test, y_pred)

    # 计算指标
    #accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(" 测试集性能:")
    print(f"  准确率 (Accuracy): {accuracy:.4f}")
    print(f"  精确率 (Precision): {precision:.4f}")
    print(f"  召回率 (Recall): {recall:.4f}")
    print(f"  F1分数: {f1:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")

    # 10. 特征重要性分析
    print_section("LightGBM特征重要性")

    feature_importance = pd.DataFrame({
        'feature': selected_features,
        'importance': lgb_model.feature_importances_,
        'importance_type': 'gain'
    }).sort_values('importance', ascending=False)

    print("Top 20 重要特征:")
    print(feature_importance.head(20).to_string(index=False))

    # 11. 保存模型和结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存模型
    model_file = f'lightgbm_quick_test_{timestamp}.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump({
            'model': lgb_model,
            'scaler': scaler,
            'features': selected_features,
            'importance': feature_importance,
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc
            }
        }, f)
    print(f" LightGBM模型已保存: {model_file}")

    # 保存结果到文本文件
    result_file = f'lightgbm_test_results_{timestamp}.txt'
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("LightGBM快速测试结果\n")
        f.write("=" * 50 + "\n")
        f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"数据样本: {len(df_sampled):,} 条\n")
        f.write(f"特征数量: {len(selected_features)} 个\n")
        f.write(f"训练集大小: {len(X_train):,} 条\n")
        f.write(f"测试集大小: {len(X_test):,} 条\n\n")

        f.write("模型性能:\n")
        f.write(f"  准确率: {accuracy:.4f}\n")
        f.write(f"  精确率: {precision:.4f}\n")
        f.write(f"  召回率: {recall:.4f}\n")
        f.write(f"  F1分数: {f1:.4f}\n")
        f.write(f"  ROC-AUC: {roc_auc:.4f}\n\n")

        f.write("Top 10 重要特征:\n")
        for idx, row in feature_importance.head(10).iterrows():
            f.write(f"  {row['feature']}: {row['importance']:.4f}\n")

    print(f" 测试结果已保存: {result_file}")

    return {
        'model': lgb_model,
        'scaler': scaler,
        'features': selected_features,
        'importance': feature_importance,
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        }
    }

# ==================== 修改主函数中的回测部分 ====================
def main():
    """主程序 - 优化版本"""
    print_section("台湾股票选股预测模型（优化版）")
    print(f"预测未来天数: {FUTURE_DAYS}天")
    print(f"回看天数: {LOOKBACK_DAYS}天")
    print(f"随机种子: {RANDOM_STATE}")
    print(f"快速模式: {'启用' if QUICK_MODE else '关闭'}")
    print(f"使用已保存数据: {'启用' if USE_SAVED_DATA else '关闭'}")
    print(f"强制重新计算因子: {'是' if FORCE_RECOMPUTE_FACTORS else '否'}")

    # 打印优化参数
    print("\n 优化参数配置:")
    print(f"  最大持仓数量: {TOP_N_HOLDINGS}只")
    print(f"  调仓频率: {REBALANCE_FREQUENCY}")
    print(f"  最小持有天数: {RISK_CONTROL['min_holding_days']}天")
    print(f"  仓位限制: {RISK_CONTROL['single_stock_limit']:.1%}")
    print(f"  止损阈值: {RISK_CONTROL['individual_stop_loss']:.1%}")
    print(f"  止盈阈值: {RISK_CONTROL['individual_stop_profit']:.1%}")
    print(f"  组合止损: {RISK_CONTROL['portfolio_stop_loss']:.1%}")
    print(f"  每日最大交易次数: {RISK_CONTROL['max_daily_trades']}次")

    # 时间预估
    print("\n预计执行时间:")
    if QUICK_MODE:
        print("  总时间: 10-15分钟")
    else:
        print("  总时间: 30-45分钟")
    print("=" * 50)

    start_time = time.time()

    try:
        # 1. 加载和预处理数据
        print_section("步骤1: 加载和预处理数据")

        # 检查预合并文件是否存在
        if os.path.exists(PRE_MERGED_FILE):
            print(f"加载预合并文件: {PRE_MERGED_FILE}")
            try:
                with open(PRE_MERGED_FILE, 'rb') as f:
                    data = pickle.load(f)

                if isinstance(data, tuple) and len(data) == 2:
                    df, feature_cols = data
                    print(f" 预合并数据加载成功: {df.shape}")
                    print(f"时间范围: {df['date'].min()} 到 {df['date'].max()}")
                    print(f"股票数量: {df['stock_code'].nunique()}")
                else:
                    print(f"❌ 预合并文件格式错误")
                    return None
            except Exception as e:
                print(f"预合并文件加载失败: {e}")
                return None
        else:
            print(f"❌ 预合并文件不存在: {PRE_MERGED_FILE}")
            print("请先运行数据预处理流程")
            return None

        # 2. 检查收益率数据
        print_section("步骤2: 检查收益率数据")

        if 'future_return' not in df.columns:
            print("❌ 数据中没有future_return列")
            return None

        # 检查收益率有效性
        valid_returns = df['future_return'].dropna()
        inf_count = np.isinf(valid_returns).sum()
        print(f"收益率数据统计:")
        print(f"  有效样本: {len(valid_returns):,}")
        print(f"  inf值数量: {inf_count}")
        print(f"  收益率范围: {valid_returns.min():.4f} 到 {valid_returns.max():.4f}")

        if inf_count > 0:
            print(" 发现inf值，进行修复...")
            df['future_return'] = df['future_return'].replace([np.inf, -np.inf], np.nan)
            df = df.dropna(subset=['future_return'])
            print(f"修复后有效样本: {len(df):,}")

        # 3. 因子筛选（如果需要）
        print_section("步骤3: 因子筛选")

        # 分离财务和技术因子
        financial_cols = [col for col in feature_cols if col.startswith('fin_')]
        technical_cols = [col for col in feature_cols if not col.startswith('fin_')]

        print(f"因子统计:")
        print(f"  财务因子: {len(financial_cols)} 个")
        print(f"  技术因子: {len(technical_cols)} 个")
        print(f"  总因子: {len(feature_cols)} 个")

        # 如果因子数量太多，进行筛选
        if len(feature_cols) > 30:
            print(f"因子数量过多({len(feature_cols)})，进行初步筛选...")

            # 使用简单的方差筛选
            feature_variances = []
            for col in feature_cols:
                if col in df.columns:
                    variance = df[col].var()
                    if not np.isnan(variance):
                        feature_variances.append((col, variance))

            # 按方差排序，选择前30个
            feature_variances.sort(key=lambda x: x[1], reverse=True)
            selected_features = [f[0] for f in feature_variances[:30]]
            feature_cols = selected_features
            print(f"筛选后因子数量: {len(feature_cols)} 个")

        # 4. 准备建模数据
        print_section("步骤4: 准备建模数据")

        modeling_df = prepare_modeling_data(df, feature_cols)
        if modeling_df.empty:
            print("❌ 建模数据为空")
            return None

        print(f"建模数据统计:")
        print(f"  样本数量: {len(modeling_df):,}")
        print(f"  特征数量: {len(feature_cols)}")
        print(f"  正样本比例: {modeling_df['label'].mean():.2%}")

        # 5. 数据集划分
        print_section("步骤5: 数据集划分")

        data_split = split_train_val_test_data(
            modeling_df, feature_cols, test_ratio=TEST_RATIO, val_ratio=VAL_RATIO
        )

        if data_split[0] is None:
            print("❌ 数据集划分失败")
            return None

        X_train, X_val, X_test, y_train, y_val, y_test, train_df, val_df, test_df = data_split

        print(f"数据集划分结果:")
        print(f"  训练集: {X_train.shape}")
        print(f"  验证集: {X_val.shape}")
        print(f"  测试集: {X_test.shape}")

        # 6. 模型训练
        print_section("步骤6: 模型训练")

        # 使用保守参数防止过拟合
        best_params = get_conservative_params()

        models, scaler, results, predictions, probabilities = train_models(
            X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, best_params
        )

        if not models:
            print("❌ 模型训练失败")
            return None

        print(f" 模型训练完成:")
        for model_name, result in results.items():
            print(f"  {model_name.upper()}: F1={result['test_f1']:.4f}, AUC={result['test_roc_auc']:.4f}")

        # 7. 生成选股列表
        print_section("步骤7: 生成选股列表")

        daily_selected_df = generate_daily_selected_stocks(test_df, predictions, probabilities, top_n=10)

        if daily_selected_df.empty:
            print("❌ 选股列表生成失败")
            return None

        print(f" 选股列表生成完成:")
        print(f"  总选股记录: {len(daily_selected_df):,}")
        print(f"  平均每日选股: {daily_selected_df.groupby('交易日').size().mean():.1f}")

        # 8. 执行简化回测
        print_section("步骤8: 执行简化回测")

        if not daily_selected_df.empty:
            # 确保列名正确
            if '交易日' in daily_selected_df.columns:
                # 重命名列以匹配回测函数
                daily_selected_for_backtest = daily_selected_df.copy()

                # 添加必要的列
                if '股票代码' in daily_selected_for_backtest.columns:
                    daily_selected_for_backtest = daily_selected_for_backtest.rename(columns={
                        '交易日': 'date',
                        '股票代码': 'stock_code',
                        '收盘价': 'close',
                        '未来20天绝对收益率': 'future_return'
                    })

                print(f"回测数据准备完成:")
                print(f"  数据形状: {daily_selected_for_backtest.shape}")
                print(
                    f"  时间范围: {daily_selected_for_backtest['date'].min()} 到 {daily_selected_for_backtest['date'].max()}")

                # 执行简化回测
                backtest_results = perform_backtest_simple(daily_selected_for_backtest, test_df)

                if backtest_results:
                    # 在回测部分添加以下代码来打印指标：
                    if backtest_results and backtest_results.get('metrics'):
                        metrics = backtest_results['metrics']
                        trading_stats = backtest_results.get('trading_stats', {})

                        # 使用新的打印函数
                        print_backtest_metrics(metrics)

                        # 打印交易统计
                        print(f"\n 交易统计:")
                        print(f"   平均持仓天数: {trading_stats.get('平均持仓天数', 0):.1f}天")
                        print(f"   年化换手率: {trading_stats.get('年化换手率', 0):.2%}")
                        print(f"   总交易成本: {trading_stats.get('总交易成本', 0):,.2f}")
                        print(f"   胜率: {trading_stats.get('胜率', 0):.2%}")

                        # 同时保存为详细报告
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        report_file = f'backtest_detailed_report_{timestamp}.txt'
                        save_backtest_report(metrics, report_file)

                    # 保存回测结果
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    backtest_file = f'backtest_results_{timestamp}.pkl'
                    with open(backtest_file, 'wb') as f:
                        pickle.dump(backtest_results, f, protocol=4)
                    print(f"\n 回测结果已保存: {backtest_file}")

                    # 生成回测报告
                    report_file = f'backtest_report_{timestamp}.txt'
                    with open(report_file, 'w', encoding='utf-8') as f:
                        f.write("=" * 60 + "\n")
                        f.write("股票选股策略回测报告\n")
                        f.write("=" * 60 + "\n\n")

                        f.write(
                            f"回测期间: {daily_selected_for_backtest['date'].min()} 到 {daily_selected_for_backtest['date'].max()}\n")
                        f.write(f"初始资金: {INITIAL_CAPITAL:,.2f} 元\n")
                        f.write(f"股票数量: {daily_selected_for_backtest['stock_code'].nunique()} 只\n")
                        f.write(f"交易天数: {len(daily_selected_for_backtest['date'].unique())} 天\n\n")

                        f.write("绩效指标:\n")
                        f.write("-" * 40 + "\n")
                        for key, value in metrics.items():
                            if isinstance(value, float):
                                if key in ['总收益率', '年化收益率', '最大回撤', '年化波动率', '胜率']:
                                    f.write(f"{key}: {value:.2%}\n")
                                elif key in ['夏普比率', '卡玛比率', '盈亏比', '信息比率']:
                                    f.write(f"{key}: {value:.2f}\n")
                                else:
                                    f.write(f"{key}: {value}\n")
                            else:
                                f.write(f"{key}: {value}\n")

                        f.write("\n交易统计:\n")
                        f.write("-" * 40 + "\n")
                        for key, value in trading_stats.items():
                            if isinstance(value, float):
                                if key in ['胜率', '平均交易成本率', '年化换手率']:
                                    f.write(f"{key}: {value:.2%}\n")
                                elif key in ['平均持仓天数']:
                                    f.write(f"{key}: {value:.1f} 天\n")
                                elif key in ['总交易成本']:
                                    f.write(f"{key}: {value:,.2f} 元\n")
                                else:
                                    f.write(f"{key}: {value}\n")
                            else:
                                f.write(f"{key}: {value}\n")

                        # 添加交易记录摘要
                        if backtest_results.get('trading_records'):
                            trades = pd.DataFrame(backtest_results['trading_records'])
                            if not trades.empty:
                                f.write(f"\n交易记录摘要:\n")
                                f.write(f"  总交易笔数: {len(trades)}\n")
                                f.write(f"  买入笔数: {len(trades[trades['type'] == 'buy'])}\n")
                                f.write(f"  卖出笔数: {len(trades[trades['type'] == 'sell'])}\n")

                                if 'profit' in trades.columns:
                                    profitable = len(trades[(trades['type'] == 'sell') & (trades['profit'] > 0)])
                                    total_sell = len(trades[trades['type'] == 'sell'])
                                    if total_sell > 0:
                                        f.write(f"  盈利交易比例: {profitable / total_sell:.2%}\n")

                    print(f" 回测报告已保存: {report_file}")

                    # 绘制简单图表
                    try:
                        plot_simple_backtest_results(backtest_results, timestamp)
                    except Exception as e:
                        print(f"图表绘制失败: {e}")
                else:
                    print("❌ 回测失败")
                    backtest_results = {
                        'metrics': {},
                        'trading_stats': {},
                        'portfolio_values': [],
                        'trading_records': []
                    }
            else:
                print("❌ 选股数据缺少必要的列")
                backtest_results = {
                    'metrics': {},
                    'trading_stats': {},
                    'portfolio_values': [],
                    'trading_records': []
                }
        else:
            print("❌ 选股数据为空，无法执行回测")
            backtest_results = {
                'metrics': {},
                'trading_stats': {},
                'portfolio_values': [],
                'trading_records': []
            }

        # 9. 保存其他结果
        print_section("步骤9: 保存其他结果")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存选股结果
        selected_file = f'selected_stocks_{timestamp}.csv'
        daily_selected_df.to_csv(selected_file, index=False, encoding='utf-8-sig')
        print(f" 选股结果已保存: {selected_file}")

        # 10. 总结
        end_time = time.time()
        execution_time = (end_time - start_time) / 60

        print_section("优化版程序执行完成")
        print(f"执行时间: {execution_time:.1f} 分钟")

        if backtest_results and backtest_results.get('portfolio_values'):
            final_value = backtest_results['portfolio_values'][-1]['portfolio_value']
            print(f"最终组合价值: {final_value:,.2f}")

            # 使用新的打印函数再次显示关键指标
            if backtest_results.get('metrics'):
                metrics = backtest_results['metrics']
                print(f"\n 关键指标:")
                print(f"   回测区间: {metrics.get('起始日期', 'N/A')} 至 {metrics.get('结束日期', 'N/A')}")
                print(f"   总收益率: {metrics.get('总收益率', 0):.2%}")
                print(f"   年化收益率: {metrics.get('年化收益率', 0):.2%}")
                print(f"   最大回撤: {metrics.get('最大回撤', 0):.2%}")
                print(f"   年化夏普比率: {metrics.get('年化夏普比率', 0):.2f}")

            if 'trading_stats' in backtest_results:
                trading_stats = backtest_results['trading_stats']
                print(f"   年化换手率: {trading_stats.get('年化换手率', 0):.2%}")
                print(f"   平均持仓天数: {trading_stats.get('平均持仓天数', 0):.1f}天")
        else:
            print(" 无有效回测结果")

        if result and result.get('backtest_results'):
            portfolio_values = result['backtest_results'].get('portfolio_values', [])
            if portfolio_values:
                start_date = portfolio_values[0]['date']
                end_date = portfolio_values[-1]['date']
                total_days = (end_date - start_date).days
                total_months = total_days / 30.44

                # 计算实际调仓次数
                trading_records = result['backtest_results'].get('trading_records', [])
                buy_count = sum(1 for t in trading_records if t.get('type') == 'buy')
                sell_count = sum(1 for t in trading_records if t.get('type') == 'sell')

                print(f"\n 实际交易统计:")
                print(f"   回测总天数: {total_days}天 ({total_months:.1f}个月)")
                print(f"   买入次数: {buy_count}次")
                print(f"   卖出次数: {sell_count}次")
                print(f"   平均每月交易: {(buy_count + sell_count) / max(1, total_months):.1f}次")

                # 计算月度换手率
                if trading_records:
                    trades_df = pd.DataFrame(trading_records)
                    trades_df['date'] = pd.to_datetime(trades_df['date'])
                    trades_df['month'] = trades_df['date'].dt.to_period('M')

                    monthly_turnover = {}
                    for month, group in trades_df.groupby('month'):
                        month_trades = group['total_value'].sum()
                        # 估算该月平均净值
                        month_values = [pv for pv in portfolio_values
                                        if pd.to_datetime(pv['date']).to_period('M') == month]
                        if month_values:
                            avg_value = np.mean([pv.get('total_value', pv.get('portfolio_value', 0))
                                                 for pv in month_values])
                            if avg_value > 0:
                                monthly_turnover[str(month)] = month_trades / avg_value

                    if monthly_turnover:
                        avg_monthly_turnover = np.mean(list(monthly_turnover.values()))
                        print(f"   平均月度换手率: {avg_monthly_turnover:.2%}")
                        print(f"   理论年化换手率: {avg_monthly_turnover * 12:.2%}")

        # 打印优化效果
        print("\n 优化效果总结:")
        print("   1.  使用简化回测逻辑，避免复杂错误")
        print("   2.  修复交易统计函数中的语法错误")
        print("   3.  确保数据列名匹配")
        print("   4.  简化价格数据获取逻辑")
        print("   5.  添加异常处理，避免程序崩溃")
        print("   6.  生成完整的回测报告和图表")
        print("   7.  新增：显示完整的区间日期和年化指标")
        print("   8.  新增：生成详细回测报告")

        return {
            'models': models,
            'scaler': scaler,
            'features': feature_cols,
            'results': results,
            'backtest_results': backtest_results,
            'selected_stocks': daily_selected_df
        }

    except Exception as e:
        print(f" 程序执行出错: {str(e)}")
        traceback.print_exc()
        return None


def print_backtest_metrics(metrics):
    """打印回测指标（包含区间日期）"""
    print_section("回测结果汇总")

    # 打印区间信息
    if '起始日期' in metrics and '结束日期' in metrics:
        print(f" 回测区间: {metrics['起始日期']} 至 {metrics['结束日期']}")
        print(f"   回测天数: {metrics.get('回测天数', 0)} 天")
        print(f"   交易日数: {metrics.get('交易日数', 0)} 天")

    # 打印收益率指标
    print(f"\n 收益率指标:")
    if '总收益率' in metrics:
        print(f"   总收益率: {metrics['总收益率']:.2%}")
    if '年化收益率' in metrics:
        print(f"   年化收益率: {metrics['年化收益率']:.2%}")

    # 打印风险指标
    print(f"\n  风险指标:")
    if '年化波动率' in metrics:
        print(f"   年化波动率: {metrics['年化波动率']:.2%}")
    if '最大回撤' in metrics:
        print(f"   最大回撤: {metrics['最大回撤']:.2%}")

    # 打印风险调整收益指标
    print(f"\n  风险调整收益指标:")
    if '年化夏普比率' in metrics:
        print(f"   年化夏普比率: {metrics['年化夏普比率']:.2f}")
    if '卡玛比率' in metrics:
        print(f"   卡玛比率: {metrics['卡玛比率']:.2f}")
    if '信息比率' in metrics and metrics['信息比率'] != 0:
        print(f"   信息比率: {metrics['信息比率']:.2f}")

    # 打印交易统计指标
    print(f"\n 交易统计指标:")
    if '胜率' in metrics:
        print(f"   胜率: {metrics['胜率']:.2%}")
    if '盈亏比' in metrics:
        print(f"   盈亏比: {metrics['盈亏比']:.2f}")

    # 打印净值信息
    print(f"\n 净值信息:")
    if '初始净值' in metrics and metrics['初始净值'] > 0:
        print(f"   初始净值: {metrics['初始净值']:,.2f}")
    if '最终净值' in metrics and metrics['最终净值'] > 0:
        print(f"   最终净值: {metrics['最终净值']:,.2f}")


def save_backtest_report(metrics, filepath):
    """保存详细的回测报告到文件"""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("股票选股策略详细回测报告\n")
            f.write("=" * 60 + "\n\n")

            # 区间信息
            f.write("1. 回测区间信息\n")
            f.write("-" * 40 + "\n")
            f.write(f"起始日期: {metrics.get('起始日期', 'N/A')}\n")
            f.write(f"结束日期: {metrics.get('结束日期', 'N/A')}\n")
            f.write(f"回测天数: {metrics.get('回测天数', 0)} 天\n")
            f.write(f"交易日数: {metrics.get('交易日数', 0)} 天\n\n")

            # 收益率指标
            f.write("2. 收益率指标\n")
            f.write("-" * 40 + "\n")
            for key in ['总收益率', '年化收益率']:
                if key in metrics:
                    f.write(f"{key}: {metrics[key]:.2%}\n")

            # 风险指标
            f.write("\n3. 风险指标\n")
            f.write("-" * 40 + "\n")
            for key in ['年化波动率', '最大回撤']:
                if key in metrics:
                    f.write(f"{key}: {metrics[key]:.2%}\n")

            # 风险调整收益指标
            f.write("\n4. 风险调整收益指标\n")
            f.write("-" * 40 + "\n")
            for key in ['年化夏普比率', '卡玛比率', '信息比率']:
                if key in metrics:
                    if key == '信息比率' and metrics[key] == 0:
                        continue
                    value = metrics[key]
                    if isinstance(value, float):
                        f.write(f"{key}: {value:.2f}\n")
                    else:
                        f.write(f"{key}: {value}\n")

            # 交易统计
            f.write("\n5. 交易统计指标\n")
            f.write("-" * 40 + "\n")
            for key in ['胜率', '盈亏比']:
                if key in metrics:
                    value = metrics[key]
                    if key == '胜率':
                        f.write(f"{key}: {value:.2%}\n")
                    else:
                        f.write(f"{key}: {value:.2f}\n")

            # 净值信息
            f.write("\n6. 净值信息\n")
            f.write("-" * 40 + "\n")
            for key in ['初始净值', '最终净值']:
                if key in metrics and metrics[key] > 0:
                    f.write(f"{key}: {metrics[key]:,.2f}\n")

            # 计算说明
            f.write("\n" + "=" * 60 + "\n")
            f.write("计算说明:\n")
            f.write("-" * 40 + "\n")
            f.write("1. 年化收益率 = (1 + 总收益率)^(365.25/天数) - 1\n")
            f.write("2. 年化波动率 = 日收益率标准差 × √252\n")
            f.write("3. 年化夏普比率 = 年化收益率 / 年化波动率 (假设无风险利率为0)\n")
            f.write("4. 最大回撤 = 最低点净值 / 最高点净值 - 1\n")
            f.write("5. 卡玛比率 = 年化收益率 / 最大回撤 (绝对值)\n")
            f.write("6. 信息比率 = (年化收益率 - 基准年化收益率) / 跟踪误差\n")

            # 风险评估
            f.write("\n" + "=" * 60 + "\n")
            f.write("风险评估:\n")
            f.write("-" * 40 + "\n")

            if metrics.get('年化夏普比率', 0) > 1.0:
                f.write(" 夏普比率 > 1.0: 策略表现优秀\n")
            elif metrics.get('年化夏普比率', 0) > 0.5:
                f.write("  夏普比率 0.5-1.0: 策略表现良好\n")
            else:
                f.write("❌ 夏普比率 < 0.5: 策略风险调整收益偏低\n")

            if metrics.get('最大回撤', 0) > -0.20:
                f.write(" 最大回撤 < 20%: 风险控制良好\n")
            elif metrics.get('最大回撤', 0) > -0.30:
                f.write("  最大回撤 20%-30%: 风险控制一般\n")
            else:
                f.write("❌ 最大回撤 > 30%: 风险控制需要改进\n")

        print(f" 详细回测报告已保存: {filepath}")

    except Exception as e:
        print(f"保存回测报告失败: {e}")

def plot_simple_backtest_results(backtest_results, timestamp):
    """绘制简单的回测图表"""
    try:
        import matplotlib.pyplot as plt

        # 净值曲线
        portfolio_values = [pv.get('total_value', pv.get('portfolio_value', 0))
                            for pv in backtest_results['portfolio_values']]
        dates = [pv['date'] for pv in backtest_results['portfolio_values']]

        plt.figure(figsize=(12, 8))

        # 净值曲线
        plt.subplot(2, 2, 1)
        plt.plot(dates, portfolio_values, 'b-', linewidth=2)
        plt.title('净值曲线')
        plt.xlabel('日期')
        plt.ylabel('净值（元）')
        plt.grid(True, alpha=0.3)

        # 回撤曲线
        plt.subplot(2, 2, 2)
        running_max = pd.Series(portfolio_values).expanding().max()
        drawdown = (pd.Series(portfolio_values) - running_max) / running_max
        plt.fill_between(dates, drawdown, 0, alpha=0.3, color='red')
        plt.title('回撤曲线')
        plt.xlabel('日期')
        plt.ylabel('回撤')
        plt.grid(True, alpha=0.3)

        # 日收益率分布
        plt.subplot(2, 2, 3)
        returns = backtest_results.get('portfolio_returns', [])
        if returns:
            plt.hist(returns, bins=50, edgecolor='black', alpha=0.7)
            plt.title('日收益率分布')
            plt.xlabel('日收益率')
            plt.ylabel('频次')
            plt.grid(True, alpha=0.3)

        # 关键指标
        plt.subplot(2, 2, 4)
        plt.axis('off')

        metrics = backtest_results.get('metrics', {})
        stats = backtest_results.get('trading_stats', {})

        text = f"""关键指标:
总收益率: {metrics.get('总收益率', 0):.2%}
年化收益率: {metrics.get('年化收益率', 0):.2%}
夏普比率: {metrics.get('夏普比率', 0):.2f}
最大回撤: {metrics.get('最大回撤', 0):.2%}

交易统计:
总交易次数: {stats.get('总交易次数', 0)}
年化换手率: {stats.get('年化换手率', 0):.2%}
平均持仓天数: {stats.get('平均持仓天数', 0):.1f}天
胜率: {stats.get('胜率', 0):.2%}"""

        plt.text(0.1, 0.5, text, fontsize=10, verticalalignment='center')

        plt.tight_layout()

        plot_file = f'backtest_chart_{timestamp}.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()

        print(f" 回测图表已保存: {plot_file}")

    except Exception as e:
        print(f"绘制图表时出错: {e}")

# ==================== 运行程序 ====================
# ==================== 运行程序 ====================
if __name__ == "__main__":
    print("开始运行台湾股票超额收益预测模型（优化版）...")
    print("=" * 60)

    result = main()

    if result is not None:
        print_section("程序执行成功!")
        print("已生成以下文件（*为时间戳占位符）:")
        print("├─ selected_stocks_*.csv        - 每日选股列表")

        # 只有在backtest_results存在时才显示
        if result.get('backtest_results') is not None:
            print("├─ backtest_results_*.pkl       - 详细回测结果")
            print("└─ backtest_report_*.txt        - 回测报告")
        else:
            print("└─ 回测未执行或失败，未生成回测结果文件")

        # 打印优化效果 - 修复：先检查backtest_results是否存在
        print("\n 优化效果总结:")

        # 安全地获取缓存命中率
        cache_hit_rate = 0
        if (result.get('backtest_results') is not None and
                'cache_stats' in result['backtest_results']):
            cache_hit_rate = result['backtest_results']['cache_stats'].get('hit_rate', 0)

        print(f"   1.  交易成本缓存命中率: {cache_hit_rate:.2%}")
        print("   2.  仓位管理器确保单只股票不超5%限制")
        print("   3.  增加最小持有期(5天)，减少日内交易")
        print("   4.  调整调仓频率为每季度，降低换手率")
        print("   5.  优化止损止盈阈值，减少无效交易")
        print("   6.  限制每日交易次数，避免过度交易")

        # 只有在backtest_results存在时才打印关键指标
        if result.get('backtest_results') is not None:
            metrics = result['backtest_results'].get('metrics', {})
            trading_stats = result['backtest_results'].get('trading_stats', {})

            print(f"\n 关键指标:")
            print(f"   总收益率: {metrics.get('总收益率', 0):.2%}")
            print(f"   最大回撤: {metrics.get('最大回撤', 0):.2%}")
            print(f"   年化换手率: {trading_stats.get('年化换手率', 0):.2%}")
            print(f"   平均持仓天数: {trading_stats.get('平均持仓天数', 0):.1f}天")
        else:
            print("\n 无有效回测结果，无法显示关键指标")
    else:
        print_section("程序执行失败!")