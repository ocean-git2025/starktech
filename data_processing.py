import pandas as pd
import numpy as np
import os


# --------------------------
# 全局配置：统一路径管理
# --------------------------
def get_project_paths():
    """获取项目所有关键路径，统一管理"""
    project_root = r"D:\XX\XX\XX" # 换成本地文件夹
    raw_dir = os.path.join(project_root, "raw")
    processed_dir = os.path.join(project_root, "processed")

    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
        print(f"自动创建processed文件夹：{processed_dir}")

    stock_raw_path = os.path.join(raw_dir, "taiwan_stock_price_202511122027.csv")
    stock_processed_path = os.path.join(processed_dir, "taiwan_stock_cleaned_adjusted.csv")
    report_raw_path = os.path.join(raw_dir, "reports_202511122033.csv")
    report_processed_path = os.path.join(processed_dir, "reports_cleaned.csv")

    return {
        "stock_raw": stock_raw_path,
        "stock_processed": stock_processed_path,
        "report_raw": report_raw_path,
        "report_processed": report_processed_path
    }


# --------------------------
# 后复权计算函数
# --------------------------
def calculate_backward_adjusted_prices(df):
    """
    后复权计算方法
    基于价格连续性，使用滚动统计识别异常跳空
    返回后复权价格序列
    """
    # 确保数据按股票和时间排序
    df = df.sort_values(['stock_id', 'date']).copy()

    # 为每只股票计算复权价格
    all_adjusted = []

    for stock_id, group in df.groupby('stock_id'):
        group = group.copy().reset_index(drop=True)

        # 计算日收益率
        group['daily_return'] = group['close'].pct_change()

        # 使用滚动窗口统计识别异常下跌（可能除权）
        window = 60  # 60天窗口
        min_periods = 20  # 最小观察期

        # 计算滚动均值和标准差
        group['rolling_mean'] = group['daily_return'].rolling(
            window=window, min_periods=min_periods).mean()
        group['rolling_std'] = group['daily_return'].rolling(
            window=window, min_periods=min_periods).std()

        # 用0.05填充NaN（对于没有足够历史数据的时期）
        group['rolling_std'] = group['rolling_std'].fillna(0.05)
        group['rolling_mean'] = group['rolling_mean'].fillna(0.0)

        # 识别异常下跌：超过3个标准差且跌幅大于8%
        # 这是保守的阈值，避免误判市场正常波动
        threshold = group['rolling_mean'] - 3 * group['rolling_std']
        threshold = threshold.clip(upper=-0.08)  # 至少8%跌幅才考虑

        # 标记可能的除权日
        group['is_adjustment_day'] = (
                (group['daily_return'] < threshold) &
                (group['daily_return'] < -0.08) &  # 至少下跌8%
                (group['daily_return'] > -0.50)  # 排除极端下跌（>50%）
        )

        # 计算调整因子
        # 后复权逻辑：如果今天除权下跌，那么之前的价格需要上调
        adjustment_factor = 1.0
        cumulative_factors = []

        # 从后往前累积调整因子（后复权的关键）
        for i in range(len(group) - 1, -1, -1):
            if group.loc[i, 'is_adjustment_day']:
                # 调整因子 = 前一日收盘价 / 当日收盘价
                if i > 0:
                    prev_close = group.loc[i - 1, 'close']
                    curr_close = group.loc[i, 'close']
                    if curr_close > 0:  # 避免除以0
                        adjustment_factor *= (prev_close / curr_close)

            cumulative_factors.append(adjustment_factor)

        # 反转因子列表（从最早到最晚）
        cumulative_factors.reverse()
        group['cumulative_factor'] = cumulative_factors

        # 计算复权价格
        group['adj_close'] = group['close'] * group['cumulative_factor']
        group['adj_open'] = group['open'] * group['cumulative_factor']
        group['adj_high'] = group['max'] * group['cumulative_factor']
        group['adj_low'] = group['min'] * group['cumulative_factor']

        # 计算调整比率（用于分析）
        group['adjustment_ratio'] = 1.0
        adj_mask = group['is_adjustment_day']
        if adj_mask.any():
            group.loc[adj_mask, 'adjustment_ratio'] = (
                    group.loc[adj_mask, 'adj_close'].shift(1) /
                    group.loc[adj_mask, 'adj_close']
            ).fillna(1.0)

        all_adjusted.append(group)

    # 合并所有股票
    result_df = pd.concat(all_adjusted, ignore_index=True)

    return result_df


# --------------------------
# 股价数据处理（清理+复权计算）
# --------------------------
def process_stock_data(paths):
    """处理股价数据：脏数据清理 + 复权计算"""
    print("=" * 50)
    print("开始处理股价数据...")
    print("=" * 50)

    if not os.path.exists(paths["stock_raw"]):
        raise FileNotFoundError(f"股价原始数据未找到！请确认文件在：{paths['stock_raw']}")

    raw_path = paths["stock_raw"]
    processed_path = paths["stock_processed"]
    print(f"股价原始数据路径：{raw_path}")
    print(f"股价处理后路径：{processed_path}\n")

    # 1. 数据读取与列名标准化
    df = pd.read_csv(raw_path, encoding="utf-8-sig")

    # 列名映射
    column_mapping = {
        "id": "id",
        "date": "date",
        "stock_id": "stock_id",
        "trading_volume": "trading_volume",
        "trading_money": "trading_money",
        "open": "open",
        "max": "max",
        "min": "min",
        "close": "close",
        "spread": "spread",
        "trading_turnover": "trading_turnover"
    }
    df = df.rename(columns=column_mapping)

    # 转换日期格式
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    print(f"🔍 股价原始数据总行数：{len(df)}")

    # 2. 脏数据清理
    df_clean = df.dropna(subset=["date"]).copy()

    # 删除全零行
    zero_mask = (df_clean["open"] == 0) & (df_clean["max"] == 0) & (df_clean["min"] == 0) & \
                (df_clean["close"] == 0) & (df_clean["trading_volume"] == 0)
    df_clean = df_clean[~zero_mask].copy()

    # 删除价格逻辑矛盾行
    df_clean["price_max"] = df_clean[["open", "close"]].max(axis=1)
    df_clean["price_min"] = df_clean[["open", "close"]].min(axis=1)
    contradict_mask = (df_clean["max"] < df_clean["price_max"]) | (df_clean["min"] > df_clean["price_min"])
    df_clean = df_clean[~contradict_mask].copy()
    df_clean = df_clean.drop(columns=["price_max", "price_min"])

    # 删除异常价格行
    price_abnormal_mask = (df_clean["close"] < 0.1) | (df_clean["close"] > 1000)
    df_clean = df_clean[~price_abnormal_mask].copy()

    # 删除成交量异常行
    volume_zero_mask = (df_clean["trading_volume"] == 0) & (df_clean["close"] > 0)
    df_clean = df_clean[~volume_zero_mask].copy()

    # 删除重复日期行
    df_clean = df_clean.drop_duplicates(subset=["stock_id", "date"], keep="first").copy()

    cleaned_count = len(df_clean)
    deleted_count = len(df) - cleaned_count
    print(f"股价脏数据清理完成：")
    print(f"   - 清理后总行数：{cleaned_count}")
    print(f"   - 删除脏数据行数：{deleted_count}")
    print(f"   - 涉及股票数量：{df_clean['stock_id'].nunique()}只\n")

    # 3. 复权计算
    print("开始计算后复权价格...")

    try:
        df_adjusted = calculate_backward_adjusted_prices(df_clean)
        print(f"复权计算完成，共处理 {df_adjusted['stock_id'].nunique()} 只股票")

        # 统计调整日信息
        adjustment_days = df_adjusted['is_adjustment_day'].sum()
        print(f"  识别出 {adjustment_days} 个潜在除权日")

        if adjustment_days > 0:
            avg_adjustment = df_adjusted[df_adjusted['is_adjustment_day']]['adjustment_ratio'].mean()
            print(f"  平均调整比率: {avg_adjustment:.4f}")

    except Exception as e:
        print(f"复权计算失败: {str(e)}")
        print("使用原始价格作为复权价格")
        df_adjusted = df_clean.copy()
        df_adjusted['adj_close'] = df_adjusted['close']
        df_adjusted['adj_open'] = df_adjusted['open']
        df_adjusted['adj_high'] = df_adjusted['max']
        df_adjusted['adj_low'] = df_adjusted['min']
        df_adjusted['cumulative_factor'] = 1.0
        df_adjusted['is_adjustment_day'] = False
        df_adjusted['adjustment_ratio'] = 1.0
        df_adjusted['daily_return'] = df_adjusted['close'].pct_change()

    # 4. 计算收益率和验证统计
    df_adjusted['adj_return'] = df_adjusted.groupby('stock_id')['adj_close'].pct_change()

    # 5. 保存结果
    final_columns = [
        "id", "date", "stock_id", "trading_volume", "trading_money",
        "open", "max", "min", "close", "spread", "trading_turnover",
        "daily_return", "is_adjustment_day", "adjustment_ratio",
        "cumulative_factor", "adj_open", "adj_high", "adj_low", "adj_close", "adj_return"
    ]

    existing_columns = [col for col in final_columns if col in df_adjusted.columns]
    df_final = df_adjusted[existing_columns].copy()

    df_final.to_csv(processed_path, index=False, encoding="utf-8-sig")

    # 6. 计算统计信息
    print(f"\n复权结果统计：")
    print(f"   - 总行数: {len(df_final)}")
    print(f"   - 股票数量: {df_final['stock_id'].nunique()}")

    if 'adj_return' in df_final.columns:
        return_stats = df_final['adj_return'].describe()
        print(f"   - 复权收益率均值: {return_stats['mean']:.6f}")
        print(f"   - 复权收益率标准差: {return_stats['std']:.6f}")
        print(f"   - 复权收益率范围: [{return_stats['min']:.4f}, {return_stats['max']:.4f}]")

    if 'adj_close' in df_final.columns:
        price_stats = df_final['adj_close'].describe()
        print(f"   - 复权价格范围: [{price_stats['min']:.2f}, {price_stats['max']:.2f}]")

    print(f"\n股价数据处理完成！")
    print(f"股价处理后文件位置：{processed_path}")
    print(f"\n复权列说明：")
    print("1. daily_return: 原始日收益率")
    print("2. is_adjustment_day: 是否为潜在除权日")
    print("3. adjustment_ratio: 单日调整比率")
    print("4. cumulative_factor: 累积复权因子")
    print("5. adj_open/adj_high/adj_low/adj_close: 后复权价格")
    print("6. adj_return: 复权后日收益率")

    return df_final


# --------------------------
# 财务数据处理（保持不变）
# --------------------------
def process_report_data(paths):
    """处理财务数据：脏数据清理（核心字段+业务规则+单位校验）"""
    print("=" * 50)
    print("开始处理财务数据...")
    print("=" * 50)

    if not os.path.exists(paths["report_raw"]):
        raise FileNotFoundError(f"财务原始数据未找到！请确认文件在：{paths['report_raw']}")

    raw_path = paths["report_raw"]
    processed_path = paths["report_processed"]
    print(f"财务原始数据路径：{raw_path}")
    print(f"财务处理后路径：{processed_path}\n")

    df_reports = pd.read_csv(raw_path, encoding="utf-8-sig")

    column_mapping = {
        "id": "id",
        "number": "number",
        "symbol": "symbol",
        "year": "year",
        "period": "period",
        "month": "month",
        "type": "type",
        "key": "key",
        "key_en": "key_en",
        "code": "code",
        "custom_code": "custom_code",
        "date": "date",
        "value": "value",
        "manual_value": "manual_value",
        "original_value": "original_value",
        "unit": "unit",
        "parent_id": "parent_id",
        "created_at": "created_at",
        "updated_at": "updated_at"
    }

    df_reports = df_reports.rename(columns=column_mapping)

    df_reports["year"] = pd.to_numeric(df_reports["year"], errors="coerce")
    df_reports["period"] = pd.to_numeric(df_reports["period"], errors="coerce")
    df_reports["month"] = pd.to_numeric(df_reports["month"], errors="coerce")
    df_reports["unit"] = pd.to_numeric(df_reports["unit"], errors="coerce")
    df_reports["value"] = pd.to_numeric(df_reports["value"], errors="coerce")
    df_reports["manual_value"] = pd.to_numeric(df_reports["manual_value"], errors="coerce")
    df_reports["original_value"] = pd.to_numeric(df_reports["original_value"], errors="coerce")

    print(f"财务原始数据总行数：{len(df_reports)}")

    df_clean = df_reports.copy()
    initial_count = len(df_clean)

    core_fields = ["number", "year", "type", "code", "unit"]
    df_clean = df_clean.dropna(subset=core_fields).copy()
    missing_core_count = initial_count - len(df_clean)
    print(f"删除核心字段缺失行：{missing_core_count} 行")

    period_abnormal_mask = (df_clean["period"].notna()) & (~df_clean["period"].isin([1, 2, 3, 4]))
    df_clean = df_clean[~period_abnormal_mask].copy()
    period_abnormal_count = len(period_abnormal_mask[period_abnormal_mask])
    print(f"删除季度异常行（非1-4）：{period_abnormal_count} 行")

    month_abnormal_mask = (df_clean["month"] != -1) & (~df_clean["month"].isin(range(1, 13)))
    df_clean = df_clean[~month_abnormal_mask].copy()
    month_abnormal_count = len(month_abnormal_mask[month_abnormal_mask])
    print(f"删除月份异常行（非-1/1-12）：{month_abnormal_count} 行")

    valid_report_types = ["balance_sheet", "comprehensive_income_statement", "cash_flow"]
    type_abnormal_mask = ~df_clean["type"].isin(valid_report_types)
    df_clean = df_clean[~type_abnormal_mask].copy()
    type_abnormal_count = len(type_abnormal_mask[type_abnormal_mask])
    print(f"删除报表类型异常行（非指定三类）：{type_abnormal_count} 行")

    print("财务数据单位分布（去重）：")
    all_units = df_clean["unit"].dropna().unique()
    print(f"所有出现的单位值：{sorted(all_units)}")

    valid_units = [1, 1000, 0.01]
    unit_abnormal_mask = ~df_clean["unit"].isin(valid_units)
    unit_abnormal_count = len(df_clean[unit_abnormal_mask])
    df_clean = df_clean[~unit_abnormal_mask].copy()
    print(f"删除单位异常行（非{valid_units}）：{unit_abnormal_count} 行")

    unique_keys = ["number", "year", "period", "month", "type", "code"]
    df_clean = df_clean.drop_duplicates(subset=unique_keys, keep="first").copy()
    duplicate_count = initial_count - len(
        df_clean) - missing_core_count - period_abnormal_count - month_abnormal_count - type_abnormal_count - unit_abnormal_count
    print(f"删除重复行（按唯一约束）：{duplicate_count} 行")

    value_fields = ["value", "manual_value", "original_value"]
    for field in value_fields:
        if field in df_clean.columns:
            extreme_mask = df_clean[field].abs() > 1e12
            df_clean = df_clean[~extreme_mask].copy()
            extreme_count = len(extreme_mask[extreme_mask])
            print(f"删除{field}极端值行（>1e12）：{extreme_count} 行")

    df_clean["date"] = pd.to_datetime(df_clean["date"], errors="coerce")
    invalid_date_count = df_clean["date"].isna().sum() - df_reports["date"].isna().sum()
    print(f"无效日期标记为空：{invalid_date_count} 行")

    final_count = len(df_clean)
    deleted_total = initial_count - final_count
    print(f"\n财务数据清理完成：")
    print(f"   - 原始总行数：{initial_count}")
    print(f"   - 清理后总行数：{final_count}")
    print(f"   - 累计删除脏数据：{deleted_total} 行")

    df_clean.to_csv(processed_path, index=False, encoding="utf-8-sig")
    print(f"\n财务数据处理完成！")
    print(f"财务处理后文件位置：{processed_path}\n")


# --------------------------
# 增强验证函数
# --------------------------
def validate_adjustment_results(processed_path):
    """验证复权计算结果"""
    try:
        df = pd.read_csv(processed_path)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        print("=" * 50)
        print("验证复权计算结果...")
        print("=" * 50)

        print(f"数据总行数: {len(df)}")
        print(f"股票数量: {df['stock_id'].nunique()}")

        if "adj_close" in df.columns:
            nan_count = df["adj_close"].isna().sum()
            print(f"复权价格NaN值数量: {nan_count}")

            df["adj_return"] = df.groupby("stock_id")["adj_close"].pct_change()
            extreme_returns = (df["adj_return"].abs() > 0.2).sum()
            print(f"复权后收益率绝对值>20%的天数: {extreme_returns}")

            if "is_adjustment_day" in df.columns:
                adjustment_data = df[df["is_adjustment_day"]]
                if len(adjustment_data) > 0:
                    print(f"\n调整日统计:")
                    print(f"  总调整日数: {len(adjustment_data)}")
                    print(f"  平均调整比率: {adjustment_data['adjustment_ratio'].mean():.4f}")
                    print(
                        f"  调整比率范围: {adjustment_data['adjustment_ratio'].min():.4f} - {adjustment_data['adjustment_ratio'].max():.4f}")

            sample_stocks = df["stock_id"].unique()[:3] if len(df["stock_id"].unique()) >= 3 else df[
                "stock_id"].unique()

            for stock_id in sample_stocks:
                stock_data = df[df["stock_id"] == stock_id].copy()
                stock_data = stock_data.sort_values("date")

                if len(stock_data) > 1:
                    raw_returns = stock_data["close"].pct_change()
                    adj_returns = stock_data["adj_close"].pct_change()

                    print(f"\n股票 {stock_id} 的复权验证:")
                    print(f"  数据天数: {len(stock_data)}")
                    print(f"  原始价格范围: {stock_data['close'].min():.2f} - {stock_data['close'].max():.2f}")
                    print(f"  复权价格范围: {stock_data['adj_close'].min():.2f} - {stock_data['adj_close'].max():.2f}")
                    print(f"  原始收益率标准差: {raw_returns.std():.6f}")
                    print(f"  复权收益率标准差: {adj_returns.std():.6f}")

                    if "is_adjustment_day" in stock_data.columns:
                        adjustment_days = stock_data["is_adjustment_day"].sum()
                        if adjustment_days > 0:
                            adj_dates = stock_data[stock_data["is_adjustment_day"]]["date"]
                            print(f"  调整日数量: {adjustment_days}")
                            if len(adj_dates) > 0:
                                dates_str = ', '.join([d.strftime('%Y-%m-%d') for d in adj_dates[:3]])
                                if len(adj_dates) > 3:
                                    dates_str += f" ... (共{len(adj_dates)}个)"
                                print(f"  调整日期: {dates_str}")

        return True
    except Exception as e:
        print(f"验证失败: {str(e)}")
        return False


# --------------------------
# 主函数：统一执行所有数据处理
# --------------------------
def main():
    """主函数：依次处理股价数据和财务数据"""
    try:
        paths = get_project_paths()

        print("=" * 60)
        print("STARKTECH 股票数据处理系统")
        print("=" * 60)

        # 处理股价数据
        stock_df = process_stock_data(paths)

        # 验证股价数据
        print("\n" + "=" * 50)
        print("开始验证股价数据...")
        validation_result = validate_adjustment_results(paths["stock_processed"])
        if validation_result:
            print("股价数据验证通过")
        else:
            print("股价数据验证发现问题")

        # 处理财务数据
        print("\n" + "=" * 50)
        process_report_data(paths)

        print("=" * 60)
        print("所有数据处理完成！")
        print(f"处理后文件均保存在：{os.path.dirname(paths['stock_processed'])}")
        print("=" * 60)

        # 额外统计信息
        if 'stock_df' in locals():
            print("\n最终数据概览:")
            print(f"股票数量: {stock_df['stock_id'].nunique()}")
            print(f"时间范围: {stock_df['date'].min().date()} 到 {stock_df['date'].max().date()}")
            print(f"总交易日数: {stock_df['date'].nunique()}")

            # 检查复权效果
            if 'adj_return' in stock_df.columns:
                extreme_up = (stock_df['adj_return'] > 0.1).sum()
                extreme_down = (stock_df['adj_return'] < -0.1).sum()
                print(f"大幅上涨(>10%)天数: {extreme_up}")
                print(f"大幅下跌(<-10%)天数: {extreme_down}")

    except Exception as e:
        print(f"\n数据处理出错：{str(e)}")
        import traceback
        traceback.print_exc()
        raise


# --------------------------
# 执行主函数
# --------------------------
if __name__ == "__main__":
    main()