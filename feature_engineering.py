# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Liu Yanwei / 刘彦巍

# feature_engineering.py

import re
import ruptures as rpt
import numpy as np
import pandas as pd

import config
import analysis_engine
import logging
logger = logging.getLogger(__name__)


def _calculate_normalized_features(df: pd.DataFrame, ref_ranges: dict) -> pd.DataFrame:
    """
    基于医学参考范围计算标准化特征 (_norm_ratio, _norm_pos)。
    """
    logger.debug("开始计算标准化特征 (_norm_ratio, _norm_pos)...")
    df_enhanced = df.copy()

    indicators_to_normalize = [
        ind for ind in ref_ranges.keys() if ind in df_enhanced.columns
    ]
    logger.debug(f"找到 {len(indicators_to_normalize)} 个指标进行标准化。")

    # 使用字典收集所有新列
    norm_features_dict = {}

    for col in indicators_to_normalize:
        if config.DERIVED_FEATURE_PATTERN.search(col):
            continue

        lower, upper = ref_ranges.get(col, (None, None))

        # 计算 _norm_ratio
        if pd.notna(upper) and upper > 0:
            norm_features_dict[f'{col}_norm_ratio'] = df_enhanced[col] / upper

        # 计算 _norm_pos
        if pd.notna(lower) and pd.notna(upper) and (upper - lower) > 1e-6:
            norm_features_dict[f'{col}_norm_pos'] = (df_enhanced[col] - lower) / (upper - lower)

    # 一次性创建 DataFrame 并合并
    if norm_features_dict:
        norm_features_df = pd.DataFrame(norm_features_dict, index=df_enhanced.index)
        df_enhanced = pd.concat([df_enhanced, norm_features_df], axis=1)

    logger.debug("标准化特征计算完毕。")
    return df_enhanced


def _calculate_dynamic_buffer_factors(df: pd.DataFrame, markers: list, ref_ranges: dict) -> pd.Series:
    """
    计算个性化的“波动缓冲区因子”。
    """
    logger.debug("开始计算动态波动缓冲区因子...")

    # 确保在 groupby 计算时，原始 df 包含 'phase'
    if 'phase' not in df.columns:
        logger.warning("缓冲区因子: DataFrame 缺少 'phase' 列，使用后备值。")
        return pd.Series(config.FALLBACK_BUFFER_FACTOR, index=markers)

    stable_mask = df.get('phase', pd.Series(dtype=str)).eq('稳定监控期')
    stable_phase_data = df[stable_mask]
    logger.debug(f"用于计算缓冲区因子的稳定期数据点: {len(stable_phase_data)}")

    volatility_factors = pd.Series(index=markers, dtype=float)

    for marker in markers:
        if marker not in stable_phase_data.columns:
            logger.warning(f"缓冲区因子: 指标 '{marker}' 在稳定期无数据，使用后备值 {config.FALLBACK_BUFFER_FACTOR}。")
            volatility_factors[marker] = config.FALLBACK_BUFFER_FACTOR
            continue

        stable_values = stable_phase_data[marker].dropna()

        if len(stable_values) < config.MIN_SAMPLES_FOR_CV:
            logger.warning(f"缓冲区因子: 指标 '{marker}' 稳定期数据点不足 ({len(stable_values)} < {config.MIN_SAMPLES_FOR_CV})，使用后备值 {config.FALLBACK_BUFFER_FACTOR}。")
            volatility_factors[marker] = config.FALLBACK_BUFFER_FACTOR
        else:
            mean = stable_values.mean()
            std = stable_values.std()

            if mean > 1e-9:
                cv = std / mean
                dynamic_factor = 1.0 + (config.DYNAMIC_BUFFER_CV_MULTIPLIER * cv)
                clipped_factor = np.clip(dynamic_factor, 1.05, 1.5) # 安全约束
                volatility_factors[marker] = clipped_factor
                logger.debug(f"  指标 '{marker}': CV={cv:.3f}, 动态因子={clipped_factor:.3f}")
            else:
                logger.warning(f"缓冲区因子: 指标 '{marker}' 稳定期均值接近零，使用后备值 {config.FALLBACK_BUFFER_FACTOR}。")
                volatility_factors[marker] = config.FALLBACK_BUFFER_FACTOR

    logger.debug("动态波动缓冲区因子计算完毕。")
    return volatility_factors


def _calculate_dynamic_growth_thresholds(df: pd.DataFrame, markers: list) -> pd.Series:
    """
    计算个性化的“陡峭增长率阈值”。
    """
    logger.debug("开始计算动态增长率阈值...")

    # --- 使用 'phase' 列 ---
    if 'phase' not in df.columns:
        logger.warning("增长率阈值: DataFrame 中缺少 'phase' 列，所有指标将使用后备阈值。")
        return pd.Series(config.FALLBACK_GROWTH_THRESHOLD, index=markers)

    stable_phase_data = df[df['phase'] == '稳定监控期']
    logger.debug(f"用于计算增长率阈值的稳定期数据点: {len(stable_phase_data)}")

    growth_thresholds = pd.Series(index=markers, dtype=float)

    for marker in markers:
        if marker not in stable_phase_data.columns:
            logger.warning(f"增长率阈值: 指标 '{marker}' 在稳定期无数据，使用后备值 {config.FALLBACK_GROWTH_THRESHOLD}。")
            growth_thresholds[marker] = config.FALLBACK_GROWTH_THRESHOLD
            continue

        stable_values = stable_phase_data[marker].dropna()

        if len(stable_values) < config.MIN_SAMPLES_FOR_CV:
            logger.warning(f"增长率阈值: 指标 '{marker}' 稳定期数据点不足 ({len(stable_values)} < {config.MIN_SAMPLES_FOR_CV})，使用后备值 {config.FALLBACK_GROWTH_THRESHOLD}。")
            growth_thresholds[marker] = config.FALLBACK_GROWTH_THRESHOLD
        else:
            stable_growth_rates = stable_values.pct_change().dropna()
            positive_growth_rates = stable_growth_rates[stable_growth_rates > 0]

            stable_absolute_increases = stable_values.diff().dropna()
            positive_absolute_increases = stable_absolute_increases[stable_absolute_increases > 0]

            if len(positive_growth_rates) < config.MIN_SAMPLES_FOR_CV:
                 logger.warning(f"增长率阈值: 指标 '{marker}' 稳定期正增长数据点不足 ({len(positive_growth_rates)} < {config.MIN_SAMPLES_FOR_CV})，使用后备值 {config.FALLBACK_GROWTH_THRESHOLD}。")
                 growth_thresholds[marker] = config.FALLBACK_GROWTH_THRESHOLD
            else:
                relative_threshold = positive_growth_rates.quantile(0.95)
                absolute_threshold = config.FALLBACK_GROWTH_THRESHOLD # Default

                if len(positive_absolute_increases) >= config.MIN_SAMPLES_FOR_CV:
                    abs_increase_percentile = positive_absolute_increases.quantile(0.95)
                    median_val = stable_values.median()
                    if median_val > 1e-6:
                        absolute_threshold = abs_increase_percentile / median_val

                dynamic_threshold = min(relative_threshold, absolute_threshold)
                clipped_threshold = np.clip(dynamic_threshold, 0.2, 1.5) # 安全护栏
                growth_thresholds[marker] = clipped_threshold
                logger.debug(f"  指标 '{marker}': 相对阈值={relative_threshold:.3f}, 绝对等效阈值={absolute_threshold:.3f}, 最终阈值={clipped_threshold:.3f}")

    logger.debug("动态增长率阈值计算完毕。")
    return growth_thresholds


def _calculate_dynamic_decay_thresholds(df: pd.DataFrame, markers: list) -> pd.Series:
    """
    计算个性化的“陡峭下降率阈值”。
    """
    logger.debug("开始计算动态下降率阈值...")

    # --- 使用 'phase' 列 ---
    if 'phase' not in df.columns:
        logger.warning("下降率阈值: DataFrame 中缺少 'phase' 列，所有指标将使用后备阈值。")
        return pd.Series(-config.FALLBACK_GROWTH_THRESHOLD, index=markers) # 注意是负值

    stable_phase_data = df[df['phase'] == '稳定监控期']
    logger.debug(f"用于计算下降率阈值的稳定期数据点: {len(stable_phase_data)}")

    decay_thresholds = pd.Series(index=markers, dtype=float)

    for marker in markers:
        if marker not in stable_phase_data.columns:
            logger.warning(f"下降率阈值: 指标 '{marker}' 在稳定期无数据，使用后备值 {-config.FALLBACK_GROWTH_THRESHOLD}。")
            decay_thresholds[marker] = -config.FALLBACK_GROWTH_THRESHOLD
            continue

        stable_values = stable_phase_data[marker].dropna()

        if len(stable_values) < config.MIN_SAMPLES_FOR_CV:
            logger.warning(f"下降率阈值: 指标 '{marker}' 稳定期数据点不足 ({len(stable_values)} < {config.MIN_SAMPLES_FOR_CV})，使用后备值 {-config.FALLBACK_GROWTH_THRESHOLD}。")
            decay_thresholds[marker] = -config.FALLBACK_GROWTH_THRESHOLD
        else:
            stable_decay_rates = stable_values.pct_change().dropna()
            negative_decay_rates = stable_decay_rates[stable_decay_rates < 0]

            if len(negative_decay_rates) < config.MIN_SAMPLES_FOR_CV:
                logger.warning(f"下降率阈值: 指标 '{marker}' 稳定期负增长数据点不足 ({len(negative_decay_rates)} < {config.MIN_SAMPLES_FOR_CV})，使用后备值 {-config.FALLBACK_GROWTH_THRESHOLD}。")
                decay_thresholds[marker] = -config.FALLBACK_GROWTH_THRESHOLD
            else:
                dynamic_threshold = negative_decay_rates.quantile(0.05) # 5%分位数
                clipped_threshold = np.clip(dynamic_threshold, -0.8, -0.15) # 安全护栏
                decay_thresholds[marker] = clipped_threshold
                logger.debug(f"  指标 '{marker}': 动态阈值={clipped_threshold:.3f}")

    logger.debug("动态下降率阈值计算完毕。")
    return decay_thresholds



def _calculate_doubling_time_features(df: pd.DataFrame, markers_to_calculate: list, ref_ranges: dict) -> pd.DataFrame:
    """
    【V4.0.3 - phase 修正版】
    为“高为异常”指标计算倍增时间(DT)和半衰期(HT)。
    """
    logger.debug(f"开始为 {len(markers_to_calculate)} 个 [高为异常] 指标计算 DT/HT (Phase-Aware)...")
    df_enhanced = df.copy()

    if not markers_to_calculate:
        return df_enhanced

    valid_markers = [
        m for m in markers_to_calculate
        if m in df_enhanced.columns and m in ref_ranges
    ]
    if not valid_markers:
        logger.warning("DT/HT 计算: 没有有效的 [高为异常] 指标可供处理。")
        return df_enhanced
    logger.debug(f"  有效指标: {valid_markers}")

    # --- 数据准备 (已修正) ---
    # 【修正】按 'phase' 分组计算 time_diff
    time_diff_days = df_enhanced.groupby('phase')['report_date'].diff().dt.days.fillna(0).values
    
    v2_df = df_enhanced[valid_markers]
    # 【修正】按 'phase' 分组计算 shift
    v1_df = df_enhanced.groupby('phase')[valid_markers].shift(1)
    
    upper_refs = pd.Series({marker: ref_ranges.get(marker, (None, None))[1] for marker in valid_markers})
    has_upper_ref_mask = upper_refs.notna().values

    # --- 结果列初始化 ---
    dt_cols = [f'{col}_DT' for col in valid_markers]
    ht_cols = [f'{col}_HT' for col in valid_markers]
    df_enhanced[dt_cols] = 0.0
    df_enhanced[ht_cols] = 0.0

    # --- 调用已修改的动态阈值函数 ---
    logger.debug("  正在调用动态阈值计算函数 (需要 df 包含 'phase' 列)...")
    dynamic_buffer_factors = _calculate_dynamic_buffer_factors(df_enhanced, valid_markers, ref_ranges) # 依赖 phase
    buffered_uln = upper_refs * dynamic_buffer_factors
    dynamic_growth_thresholds = _calculate_dynamic_growth_thresholds(df_enhanced, valid_markers) # 依赖 phase

    # --- 触发条件定义 (依赖于上面动态计算的结果) ---
    v1_is_elevated = (v1_df.values > buffered_uln.values) & has_upper_ref_mask[:,None].T # Add mask check
    v2_is_elevated = (v2_df.values > buffered_uln.values) & has_upper_ref_mask[:,None].T # Add mask check

    from_zero_condition = (
        (v1_df.values <= 0) &
        v2_is_elevated &
        (time_diff_days[:, None] > 0) &
        has_upper_ref_mask[:,None].T
    )

    standard_dt_condition = (
        (v1_is_elevated | (~v1_is_elevated & v2_is_elevated)) &
        (v2_df.values > v1_df.values) &
        (time_diff_days[:, None] > 0) &
        has_upper_ref_mask[:,None].T
    )

    growth_rate = np.divide(
        v2_df.values - v1_df.values,
        np.where(v1_df.values > config.EPSILON, v1_df.values, config.EPSILON),
        out=np.zeros_like(v2_df.values),
        where=v1_df.values > config.EPSILON
    )

    rapid_growth_within_normal_range_condition = (
        (~v2_is_elevated) &
        (growth_rate > dynamic_growth_thresholds.values) &
        (time_diff_days[:, None] > 0) &
        has_upper_ref_mask[:,None].T
    )

    dt_condition = standard_dt_condition | from_zero_condition | rapid_growth_within_normal_range_condition

    ht_condition = (
        v1_is_elevated &
        (v2_df.values < v1_df.values) &
        (v2_df.values > 0) &
        (time_diff_days[:, None] > 0) &
        has_upper_ref_mask[:,None].T
    )

    # --- 核心计算 ---
    v1_safe = np.where(v1_df.values <= 0, config.EPSILON, v1_df.values)
    time_diff_safe = np.where(time_diff_days[:, None] == 0, config.EPSILON, time_diff_days[:, None])

    with np.errstate(divide='ignore', invalid='ignore'):
        log_ratio = np.log(np.maximum(v2_df.values, config.EPSILON) / v1_safe) # Add max(EPSILON) for safety
        daily_log_rate = log_ratio / time_diff_safe
        doubling_time = np.log(2) / daily_log_rate
        halving_time = np.abs(np.log(2) / daily_log_rate)

    # --- 清理与合理性检查 ---
    doubling_time = np.nan_to_num(doubling_time, nan=0.0, posinf=0.0, neginf=0.0)
    halving_time = np.nan_to_num(halving_time, nan=0.0, posinf=0.0, neginf=0.0)

    extreme_dt_mask = (doubling_time > 0) & (doubling_time < config.MIN_PLAUSIBLE_DT)
    doubling_time[extreme_dt_mask] = config.MIN_PLAUSIBLE_DT
    extreme_ht_mask = (halving_time > 0) & (halving_time < config.MIN_PLAUSIBLE_HT)
    halving_time[extreme_ht_mask] = config.MIN_PLAUSIBLE_HT

    doubling_time = np.clip(doubling_time, a_min=None, a_max=9999)
    halving_time = np.clip(halving_time, a_min=None, a_max=9999)

    # --- 安全赋值 ---
    results_dt_df = pd.DataFrame(np.where(dt_condition, doubling_time, 0), index=v2_df.index, columns=dt_cols)
    results_ht_df = pd.DataFrame(np.where(ht_condition, halving_time, 0), index=v2_df.index, columns=ht_cols)

    df_enhanced.update(results_dt_df)
    df_enhanced.update(results_ht_df)

    logger.debug(f"[高为异常] 指标 DT/HT 计算完毕 (Phase-Aware)。")
    return df_enhanced


def _calculate_decay_time_features(df: pd.DataFrame, markers_to_calculate: list, ref_ranges: dict) -> pd.DataFrame:
    """
    【V4.0.3 - phase 修正版】
    为“低为异常”指标计算半衰期(HT)和恢复时间(RT)。
    """
    logger.debug(f"开始为 {len(markers_to_calculate)} 个 [低为异常] 指标计算 HT/RT (Phase-Aware)...")
    df_enhanced = df.copy()

    if not markers_to_calculate:
        return df_enhanced

    valid_markers = [
        m for m in markers_to_calculate
        if m in df_enhanced.columns and m in ref_ranges
    ]
    if not valid_markers:
        logger.warning("HT/RT 计算: 没有有效的 [低为异常] 指标可供处理。")
        return df_enhanced
    logger.debug(f"  有效指标: {valid_markers}")

    # --- 数据准备 (已修正) ---
    # 【修正】按 'phase' 分组计算 time_diff
    time_diff_days = df_enhanced.groupby('phase')['report_date'].diff().dt.days.fillna(0).values
    
    v2_df = df_enhanced[valid_markers]
    # 【修正】按 'phase' 分组计算 shift
    v1_df = df_enhanced.groupby('phase')[valid_markers].shift(1)
    
    lower_refs = pd.Series({marker: ref_ranges.get(marker, (None, None))[0] for marker in valid_markers})
    has_lower_ref_mask = lower_refs.notna().values
    upper_refs = pd.Series({marker: ref_ranges.get(marker, (None, None))[1] for marker in valid_markers}) # For recovery_from_high
    has_upper_ref_mask = upper_refs.notna().values # For recovery_from_high

    # --- 结果列初始化 ---
    ht_cols = [f'{col}_HT' for col in valid_markers]
    rt_cols = [f'{col}_RT' for col in valid_markers]
    df_enhanced[ht_cols] = 0.0
    df_enhanced[rt_cols] = 0.0

    # --- 调用已修改的动态阈值函数 ---
    logger.debug("  正在调用动态阈值计算函数 (需要 df 包含 'phase' 列)...")
    dynamic_buffer_factors = _calculate_dynamic_buffer_factors(df_enhanced, valid_markers, ref_ranges) # 依赖 phase
    buffered_lln = lower_refs / dynamic_buffer_factors # 注意是除法
    dynamic_decay_thresholds = _calculate_dynamic_decay_thresholds(df_enhanced, valid_markers) # 依赖 phase

    # --- 触发条件定义 (基于 lower_refs 和 decay) ---
    v1_is_depressed = (v1_df.values < buffered_lln.values) & has_lower_ref_mask[:,None].T

    standard_ht_condition = (
        v1_is_depressed &
        (v2_df.values < v1_df.values) &
        (time_diff_days[:, np.newaxis] > 0) &
        has_lower_ref_mask[:,None].T
    )

    change_rate = np.divide(
        v2_df.values - v1_df.values,
        np.where(v1_df.values > config.EPSILON, v1_df.values, config.EPSILON),
        out=np.zeros_like(v2_df.values),
        where=v1_df.values > config.EPSILON
    )

    rapid_decay_within_normal_range_condition = (
        (~v1_is_depressed) &
        (change_rate < dynamic_decay_thresholds.values) & # 使用下降阈值
        (time_diff_days[:, np.newaxis] > 0) &
        has_lower_ref_mask[:,None].T
    )

    ht_condition = standard_ht_condition | rapid_decay_within_normal_range_condition

    # RT 条件
    recovery_from_low = (
        v1_is_depressed &
        (v2_df.values > v1_df.values) &
        (time_diff_days[:, np.newaxis] > 0) &
        has_lower_ref_mask[:,None].T
    )
    v1_was_elevated = (v1_df.values > upper_refs.values) & has_upper_ref_mask[:,None].T
    recovery_from_high = (
        v1_was_elevated &
        (v2_df.values < v1_df.values) &
        (~v1_is_depressed) & 
        (time_diff_days[:, np.newaxis] > 0) &
        has_upper_ref_mask[:,None].T
    )
    rt_condition = recovery_from_low | recovery_from_high

    # --- 核心计算 ---
    v1_safe = np.where(v1_df.values <= 0, config.EPSILON, v1_df.values)
    time_diff_safe = np.where(time_diff_days[:, np.newaxis] == 0, config.EPSILON, time_diff_days[:, np.newaxis])

    with np.errstate(divide='ignore', invalid='ignore'):
        log_ratio = np.log(np.maximum(v2_df.values, config.EPSILON) / v1_safe) # Add max(EPSILON)
        daily_log_rate = log_ratio / time_diff_safe
        halving_time = np.abs(np.log(2) / daily_log_rate)
        recovery_time = np.log(2) / daily_log_rate 

    # --- 清理与合理性检查 ---
    halving_time = np.nan_to_num(halving_time, nan=0.0, posinf=0.0, neginf=0.0)
    recovery_time = np.nan_to_num(recovery_time, nan=0.0, posinf=0.0, neginf=0.0)

    extreme_ht_mask = (halving_time > 0) & (halving_time < config.MIN_PLAUSIBLE_HT)
    halving_time[extreme_ht_mask] = config.MIN_PLAUSIBLE_HT
    extreme_rt_mask = (recovery_time > 0) & (recovery_time < config.MIN_PLAUSIBLE_RT)
    recovery_time[extreme_rt_mask] = config.MIN_PLAUSIBLE_RT

    halving_time = np.clip(halving_time, a_min=None, a_max=9999)
    recovery_time = np.clip(recovery_time, a_min=None, a_max=9999)

    # --- 安全赋值 ---
    final_rt_values = np.where(
        recovery_from_high, 
        halving_time, 
        recovery_time 
    )

    results_ht_df = pd.DataFrame(np.where(ht_condition, halving_time, 0), index=v2_df.index, columns=ht_cols)
    results_rt_df = pd.DataFrame(np.where(rt_condition, final_rt_values, 0), index=v2_df.index, columns=rt_cols)

    df_enhanced.update(results_ht_df)
    df_enhanced.update(results_rt_df)

    logger.debug(f"[低为异常] 指标 HT/RT 计算完毕 (Phase-Aware)。")
    return df_enhanced


def _calculate_exceedance_duration_features(df: pd.DataFrame, ref_ranges: dict) -> pd.DataFrame:
    """
    【V4.0.3 - phase 修正版】计算指标持续超出正常范围（ULN/LLN）的天数。
    
    在每个 phase 内部独立计算。
    
    为每个数值列生成两个新特征：
    - _days_since_first_high: 持续高于正常上限(ULN)的天数。
    - _days_since_first_low: 持续低于正常下限(LLN)的天数。
    如果指标恢复正常，计数器会重置为0。缺失值不会中断计数。
    """
    logger.debug("  (辅助函数) _calculate_exceedance_duration_features (Phase-Aware): 开始计算...")
    df_enhanced = df.copy()
    
    lab_indicator_columns = [
        col for col in ref_ranges.keys()
        if col in df_enhanced.columns and not config.DERIVED_FEATURE_PATTERN.search(col)
    ]
    logger.debug(f"    > 找到 {len(lab_indicator_columns)} 个带参考范围的指标用于计算。")

    if 'report_date' not in df_enhanced.columns:
        logger.error("  > _calculate_exceedance_duration_features: 致命错误: 'report_date' 列不存在。")
        return df 
    
    # 按 'phase' 分组处理
    all_high_cols_data = []
    all_low_cols_data = []

    for phase, group_df in df_enhanced.groupby('phase'):
        logger.debug(f"    > 正在处理 Phase '{phase}' (共 {len(group_df)} 行)...")
        
        # 提取当前组的日期
        dates_series = group_df['report_date']
        
        # 内部辅助函数，现在在 group 内部工作
        def _calculate_duration_in_group(value_series: pd.Series, date_series: pd.Series, ref_value: float, comparison_op) -> pd.Series:
            durations = [0] * len(value_series)
            first_exceed_date = None
            for i, (idx, current_date) in enumerate(date_series.items()):
                current_value = value_series.loc[idx] 

                if pd.notna(current_value):
                    if comparison_op(current_value, ref_value):
                        if first_exceed_date is None:
                            first_exceed_date = current_date
                        duration = (current_date - first_exceed_date).days
                        durations[i] = duration
                    else:
                        first_exceed_date = None # 恢复正常，重置
                else: 
                    if first_exceed_date is not None:
                        duration = (current_date - first_exceed_date).days
                        durations[i] = duration
            # 返回一个以原始索引(idx)为索引的Series
            return pd.Series(durations, index=date_series.index, dtype=float)

        for col in lab_indicator_columns:
            if col not in group_df.columns:
                continue
                
            lower_ref, upper_ref = ref_ranges.get(col, (None, None))
            
            # 计算持续高于上限(ULN)的天数
            if pd.notna(upper_ref):
                high_col_name = f'{col}_days_since_first_high'
                duration_series = _calculate_duration_in_group(group_df[col], dates_series, upper_ref, np.greater)
                duration_series.name = high_col_name
                all_high_cols_data.append(duration_series)

            # 计算持续低于下限(LLN)的天数
            if pd.notna(lower_ref):
                low_col_name = f'{col}_days_since_first_low'
                duration_series = _calculate_duration_in_group(group_df[col], dates_series, lower_ref, np.less)
                duration_series.name = low_col_name
                all_low_cols_data.append(duration_series)

    # 在循环外合并所有结果
    if all_high_cols_data:
        high_df = pd.concat(all_high_cols_data, axis=1)
        df_enhanced = df_enhanced.join(high_df) # 按索引合并
    if all_low_cols_data:
        low_df = pd.concat(all_low_cols_data, axis=1)
        df_enhanced = df_enhanced.join(low_df) # 按索引合并

    logger.debug("  (辅助函数) _calculate_exceedance_duration_features (Phase-Aware): 计算完毕。")
    return df_enhanced



def _calculate_changepoint_features(df: pd.DataFrame, dynamic_indicators: list) -> pd.DataFrame:
    """
    【V4.0.3 - phase 修正版】使用Ruptures库检测趋势拐点。
    
    在每个 phase 内部独立计算。
    
    生成两个特征：
    1. _days_since_cp: 距离上一个拐点的天数（趋势已持续多久）。
    2. _is_changepoint: 是否为拐点（趋势改变的瞬间）。
    """
    logger.info("  (辅助函数) _calculate_changepoint_features (Phase-Aware): 开始计算...")
    df_enhanced = df.copy()
    
    if 'report_date' not in df_enhanced.columns:
        logger.error("  > _calculate_changepoint_features: 致命错误: 'report_date' 列不存在。")
        return df 

    valid_indicators = [ind for ind in dynamic_indicators if ind in df_enhanced.columns and df_enhanced[ind].notna().sum() >= 5]
    logger.debug(f"    > 共有 {len(valid_indicators)} 个有效指标进行拐点检测。")
    
    all_cp_cols_data = [] # 存储所有计算结果

    for indicator in valid_indicators:
        days_feature_name = f'{indicator}_days_since_cp'
        is_cp_feature_name = f'{indicator}_is_changepoint'
        
        # 初始化结果列 (Series)
        days_since_cp_full = pd.Series(0.0, index=df_enhanced.index, name=days_feature_name)
        is_changepoint_full = pd.Series(0.0, index=df_enhanced.index, name=is_cp_feature_name)

        # 按 'phase' 分组处理
        for phase, group_df in df_enhanced.groupby('phase'):
            
            # --- 同时获取指标值和对应的日期，并丢弃NaN ---
            full_series_data = group_df[[indicator, 'report_date']].dropna()
            logger.debug(f"    > Phase '{phase}': 检测到 {len(full_series_data)} 个有效数据点")

            if len(full_series_data) < 3:
                logger.warning(f"    > 跳过 {indicator} (Phase: {phase}) 拐点检测：数据点不足 ({len(full_series_data)} < 3)。")
                continue
                
            series = full_series_data[indicator]
            series_dates = full_series_data['report_date']
                
            try:
                algo = rpt.Pelt(model="rbf").fit(series.values)
                result_indices = algo.predict(pen=np.log(len(series)))

                changepoint_dates = [series_dates.iloc[i-1] for i in result_indices if i < len(series)]
                logger.info(f"    > 指标 {indicator} (Phase: {phase}) 成功，发现 {len(changepoint_dates)} 个拐点。")
                
                if changepoint_dates:
                    changepoint_indices = series_dates[series_dates.isin(changepoint_dates)].index
                    is_changepoint_full.loc[changepoint_indices] = 1.0 # 在完整Series上赋值

                # --- 计算自上一个拐点以来的天数 ---
                all_segment_start_dates = sorted(list(set([series_dates.iloc[0]] + changepoint_dates)))
                
                segment_indices = np.searchsorted(all_segment_start_dates, series_dates, side='right') - 1
                
                start_dates_array = np.array(all_segment_start_dates)
                last_cp_date_values = start_dates_array[segment_indices]
                
                last_cp_date_series = pd.Series(last_cp_date_values, index=series.index, dtype='datetime64[ns]')
                
                days_since_cp = (series_dates - last_cp_date_series).dt.days
                
                # 将结果合并回【完整】Series
                days_since_cp_full.loc[series.index] = days_since_cp.fillna(0)

            except Exception as e:
                logger.error(
                    f"使用 Ruptures 计算拐点特征在指标'{indicator}' (Phase: {phase}) 上失败", 
                    exc_info=True
                )
        
        # 将这个指标的两个完整结果列添加到列表中
        all_cp_cols_data.append(days_since_cp_full)
        all_cp_cols_data.append(is_changepoint_full)

    # 在循环外合并所有结果
    if all_cp_cols_data:
        cp_df = pd.concat(all_cp_cols_data, axis=1)
        df_enhanced = df_enhanced.join(cp_df) # 按索引合并

    logger.info("  (辅助函数) _calculate_changepoint_features (Phase-Aware): 计算完毕。")        
    return df_enhanced


def _calculate_transition_shock_features(df: pd.DataFrame, numeric_columns: list) -> pd.DataFrame:
    """
    【V4.0.3 - 新增辅助函数 - 性能优化版】
    计算跨越 Phase 边界的“过渡冲击”特征。
    
    这些特征专用于捕捉 phase 变化瞬间的剧烈变动，
    例如从“治疗期”到“稳定期”的指标大幅下降。
    """
    logger.debug("  (辅助函数) _calculate_transition_shock_features (V-Optimized): 开始计算...")
    
    # 1. 识别边界点 (一次性计算)
    # (假设 df 已经按 report_date 排序)
    last_phase = df['phase'].shift(1)
    
    # is_first_row 仅为 True (在索引0处)
    is_first_row_mask = last_phase.isna() 
    # is_transition_point 为 True (在每个 phase 改变的行)
    is_transition_point_mask = (df['phase'] != last_phase) & ~is_first_row_mask
    
    if not is_transition_point_mask.any():
        logger.debug("    > 未检测到 Phase 边界，跳过过渡特征计算。")
        return df # 无需计算

    logger.debug(f"    > 检测到 {is_transition_point_mask.sum()} 个 Phase 边界点。")

    # 2. 计算跨界 (Inter-Phase) 变化率 (一次性计算)
    # 这里的 diff 是不带 groupby 的，是故意的
    time_diff_days_inter = df['report_date'].diff().dt.days
    
    shock_features_to_add = {} # 存储新列

    for col in numeric_columns:
        # 只计算原始指标，不计算衍生特征
        if config.DERIVED_FEATURE_PATTERN.search(col):
            continue
            
        shock_feature_name = f'{col}_transition_shock'
        
        # 跨界 diff
        value_diff_inter = df[col].diff()
        
        # 计算冲击率
        shock_rate = np.where(
            time_diff_days_inter > 0,
            value_diff_inter / time_diff_days_inter,
            0
        )
        
        # 3. 仅在边界点赋值 (向量化)
        # 只有在 is_transition_point_mask 为 True 的行，才保留 shock_rate，否则为 0
        shock_features_to_add[shock_feature_name] = np.where(
            is_transition_point_mask,
            shock_rate,
            0
        )
    
    # 4. 一次性将所有新列合并回 DataFrame
    if shock_features_to_add:
        shock_df = pd.DataFrame(shock_features_to_add, index=df.index)
        df = pd.concat([df, shock_df], axis=1)

    logger.debug("  (辅助函数) _calculate_transition_shock_features: 计算完毕。")
    return df


def add_change_rate_features(df: pd.DataFrame, ref_ranges: dict) -> pd.DataFrame:
    """
    【特征工程-临床智能增强最终版 V4.0.3 - "双特征"策略版】
    
    本函数是整个预警系统的核心特征生成器。
    
    V4.0.3 核心修正: "双特征"策略
    
    1. "期内特征" (Intra-Phase Features):
       - 包括: _rate, _accel, _DT, _HT, _RT, _days_since_...
       - 计算方式: **严格使用 `groupby('phase')`**
       - 目的: 建立统计纯净的基线，用于 IsolationForest 模型训练
       - 结果: 阶段边界点的特征值为 0，防止模型污染
       
    2. "过渡特征" (Transition Features):
       - 包括: _transition_shock
       - 计算方式: **不使用 `groupby('phase')`**，仅在 phase 边界点计算
       - 目的: 捕捉跨阶段的临床事件（如“治疗反应”），专供 RiskEngine 使用
       - 结果: 边界点的特征值反映了剧烈变化，非边界点为 0
    """
    logger.info("===== 特征工程 (V4.0.3 - '双特征'策略版) 开始执行 =====")

    # --- 1. 初始条件检查 ---
    if df.shape[0] < 2:
        logger.warning("输入数据少于2行，无法计算时序特征，直接返回。")
        return df.copy()

    # --- 2. 数据准备与排序 ---
    df_enhanced = df.copy()
    
    df_enhanced = df_enhanced.reset_index()
    df_enhanced['original_order'] = df_enhanced.index

    # 关键：所有计算（包括 groupby）都依赖于按日期排序
    df_enhanced = df_enhanced.sort_values(by=['report_date'])
    
    # ========================================
    # V4.0 自适应特征复杂度 - 步骤1: 计算组大小并记录
    # ========================================
    MIN_SAMPLES_FOR_ADV_FEATURES = 10
    logger.info(f"[V4.0] 启用自适应特征复杂度 (阈值 = {MIN_SAMPLES_FOR_ADV_FEATURES} 样本)")
    
    phase_sample_counts = df_enhanced.groupby('phase').size()
    for phase_name, count in phase_sample_counts.items():
        if count < MIN_SAMPLES_FOR_ADV_FEATURES:
            logger.warning(
                f"  ⚠️ Phase '{phase_name}' 样本量不足 ({count} < {MIN_SAMPLES_FOR_ADV_FEATURES})，"
                f"其高阶特征(加速度、拐点等)将被屏蔽"
            )
        else:
            logger.info(f"  ✅ Phase '{phase_name}' 样本量充足 ({count} ≥ {MIN_SAMPLES_FOR_ADV_FEATURES})")
    # ========================================
    
    numeric_cols = df_enhanced.select_dtypes(include=np.number).columns
    columns_to_exclude = ['id', 'original_order']
    numeric_columns = numeric_cols.drop(columns_to_exclude, errors='ignore').tolist()
    logger.debug(f"步骤 2: 数据准备完毕。找到 {len(numeric_columns)} 个数值列用于计算。")
    
    # --- 3. 动态特征计算（所有依赖时序的功能） ---
    
    # 【性能优化】一次性计算 [期内] time_diff，供 3a 和 3b 使用
    time_diff_days_internal = df_enhanced.groupby('phase')['report_date'].diff().dt.days
    

    # 3a. 计算 [期内] 一阶变化率 (速度) - 【基础特征】
    logger.debug("步骤 3a: (优化) 批量计算 [期内] 一阶变化率 (速度)...")
    
    # 【修改开始】使用字典收集，最后一次性合并
    rate_features_dict = {}
    rate_columns = []
    
    for col in numeric_columns:
        # 按 phase 分组计算
        value_diff_internal = df_enhanced.groupby('phase')[col].diff()
        rate_col_name = f"{col}_rate"
        rate_columns.append(rate_col_name)

        # 先存入字典，而不是直接赋值给 DataFrame
        rate_features_dict[rate_col_name] = np.where(
            time_diff_days_internal > 0, 
            value_diff_internal / time_diff_days_internal, 
            0
        )
    
    # 一次性创建 DataFrame 并合并
    rate_features_df = pd.DataFrame(rate_features_dict, index=df_enhanced.index)
    df_enhanced = pd.concat([df_enhanced, rate_features_df], axis=1)

    # 3b. 计算 [期内] 二阶变化率 (加速度) - 【高阶特征】
    logger.debug("步骤 3b: (优化) 批量计算 [期内] 二阶变化率 (加速度)...")
    
    # 【修改开始】使用字典收集，最后一次性合并
    accel_features_dict = {}
    accel_columns = []
    
    for rate_col in rate_columns:
        # 【修正】按 phase 分组计算
        rate_diff_internal = df_enhanced.groupby('phase')[rate_col].diff()
        accel_col_name = f"{rate_col}_accel"
        accel_columns.append(accel_col_name)
        
        # 先存入字典
        accel_features_dict[accel_col_name] = np.where(
            time_diff_days_internal > 0, 
            rate_diff_internal / time_diff_days_internal, 
            0
        )
    
    # 一次性创建 DataFrame 并合并
    accel_features_df = pd.DataFrame(accel_features_dict, index=df_enhanced.index)
    df_enhanced = pd.concat([df_enhanced, accel_features_df], axis=1)

    # 3c.1: 计算 [期内] 持续超限天数 - 【高阶特征】
    logger.debug("步骤 3c.1: (修正) 计算 [期内] 持续超限天数...")
    df_enhanced = _calculate_exceedance_duration_features(df_enhanced, ref_ranges)

    # 3c.2: 计算 [期内] 趋势拐点 - 【高阶特征】
    logger.debug("步骤 3c.2: (修正) 计算 [期内] 趋势拐点...")
    original_indicators = [
        col for col in df_enhanced.columns if not config.DERIVED_FEATURE_PATTERN.search(col)
        and df_enhanced[col].dtype in ['float64', 'int64']
        and col not in ['id', 'patient_id', 'original_order']
    ]
    variances = df_enhanced[original_indicators].var().dropna()
    valid_indicators_for_variance = variances[df_enhanced[original_indicators].notna().sum() >= 5]
    top_dynamic_indicators = valid_indicators_for_variance.sort_values(ascending=False).head(6).index.tolist()

    if top_dynamic_indicators:
        logger.info(f"自动选择的动态指标进行拐点分析: {top_dynamic_indicators}")
        df_enhanced = _calculate_changepoint_features(df_enhanced, top_dynamic_indicators)
    else:
        logger.warning("未找到足够的动态指标来进行拐点分析。")


    # --- 4. 边界处理 (已废弃) ---
    # logger.debug("步骤 4: 边界处理 (已废弃，groupby 已自动处理)")
    # (由于步骤 3a/3b/3c 均已使用 groupby('phase')，边界点自动为 0 或 NaN，无需手动重置)
            
    # --- 5. 临床综合指标计算 - 【基础特征】 ---
    logger.debug("步骤 5: 开始计算临床综合指标...")
    # 5a. 中性粒细胞与淋巴细胞比率 (NLR)
    if '中性粒细胞绝对数' in df_enhanced.columns and '淋巴细胞绝对数' in df_enhanced.columns:
        with np.errstate(divide='ignore', invalid='ignore'):
            nlr_values = np.divide(df_enhanced['中性粒细胞绝对数'], df_enhanced['淋巴细胞绝对数'])
        df_enhanced['NLR'] = np.nan_to_num(nlr_values, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        logger.warning("跳过 NLR 计算：缺少 '中性粒细胞绝对数' 或 '淋巴细胞绝对数' 列。")

    # 5b. 血小板与淋巴细胞比率 (PLR)
    if '血小板计数' in df_enhanced.columns and '淋巴细胞绝对数' in df_enhanced.columns:
        with np.errstate(divide='ignore', invalid='ignore'):
            plr_values = np.divide(df_enhanced['血小板计数'], df_enhanced['淋巴细胞绝对数'])
        df_enhanced['PLR'] = np.nan_to_num(plr_values, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        logger.warning("跳过 PLR 计算：缺少 '血小板计数' 或 '淋巴细胞绝对数' 列。")

    # 5c. 淋巴细胞与单核细胞比值 (LMR)
    if '淋巴细胞绝对数' in df_enhanced.columns and '单核细胞绝对数' in df_enhanced.columns:
        with np.errstate(divide='ignore', invalid='ignore'):
            lmr_values = np.divide(df_enhanced['淋巴细胞绝对数'], df_enhanced['单核细胞绝对数'])
        df_enhanced['LMR'] = np.nan_to_num(lmr_values, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        logger.warning("跳过 LMR 计算：缺少 '淋巴细胞绝对数' 或 '单核细胞绝对数' 列。")

    # 5d. 全身免疫炎症指数 (SII)
    if all(col in df_enhanced.columns for col in ['血小板计数', '中性粒细胞绝对数', '淋巴细胞绝对数']):
        with np.errstate(divide='ignore', invalid='ignore'):
            nlr_values_for_sii = np.divide(df_enhanced['中性粒细胞绝对数'], df_enhanced['淋巴细胞绝对数'])
            sii_values = df_enhanced['血小板计数'] * nlr_values_for_sii
        df_enhanced['SII'] = np.nan_to_num(sii_values, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        logger.warning("跳过 SII 计算：缺少 '血小板计数', '中性粒细胞绝对数' 或 '淋巴细胞绝对数' 列。")

    # 5e. 全身炎症反应指数 (SIRI)
    if all(col in df_enhanced.columns for col in ['中性粒细胞绝对数', '单核细胞绝对数', '淋巴细胞绝对数']):
        with np.errstate(divide='ignore', invalid='ignore'):
            siri_values = np.divide(df_enhanced['中性粒细胞绝对数'] * df_enhanced['单核细胞绝对数'], df_enhanced['淋巴细胞绝对数'])
        df_enhanced['SIRI'] = np.nan_to_num(siri_values, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        logger.warning("跳过 SIRI 计算：缺少 '中性粒细胞绝对数', '单核细胞绝对数' 或 '淋巴细胞绝对数' 列。")

    # --- 6. [期内] 指数加权移动平均 (EWMA) - 【基础特征】 --
    span = 3 
    logger.debug(f"步骤 6: (修正) 计算 [期内] EWMA (span={span})...")
    ewma_cols = [col for col in numeric_columns if '_rate' not in col and '_accel' not in col]
    for col in ewma_cols:
        df_enhanced[f'{col}_ewma'] = df_enhanced.groupby('phase')[col]\
            .transform(lambda x: x.ewm(span=span, adjust=False).mean())

    # --- 7. 标准化特征 - 【基础特征】 ---
    logger.debug("步骤 7: 开始计算基于医学参考范围的标准化特征...")
    df_enhanced = _calculate_normalized_features(df_enhanced, ref_ranges)

    # --- 8. [期内] DT/HT/RT - 【基线依赖特征】 ---
    logger.debug("步骤 8: (修正) 计算 [期内] 高级时序特征 (倍增/半衰期)...")
    
    high_is_bad_markers = []
    low_is_bad_markers = []
    for category_items in config.LAB_REPORT_CONFIG.values():
        for item in category_items:
            behavior = item.get("behavior", "neutral")
            if behavior == "high_is_bad":
                high_is_bad_markers.append(item["name"])
            elif behavior == "low_is_bad":
                low_is_bad_markers.append(item["name"])
    
    # 调用已修正的 groupby 辅助函数
    df_enhanced = _calculate_doubling_time_features(df_enhanced, high_is_bad_markers, ref_ranges)
    df_enhanced = _calculate_decay_time_features(df_enhanced, low_is_bad_markers, ref_ranges)

    # --- 8b. 计算 "过渡特征" ---
    logger.debug("步骤 8b: (新增) 计算 [过渡] 特征 (_transition_shock)...")
    # (此函数内部不使用 groupby)
    df_enhanced = _calculate_transition_shock_features(df_enhanced, numeric_columns)

    # --- 9. 自适应特征复杂度 - 步骤2: 屏蔽 [期内] 高阶特征 ---
    logger.debug("步骤 9: 正在应用自适应特征复杂度屏蔽...")
    
    current_group_sizes = df_enhanced.groupby('phase')['phase'].transform('size')
    
    # duration_cp_columns 也需要在辅助函数中修正
    duration_cp_columns = [
        col for col in df_enhanced.columns 
        if col.endswith('_days_since_first_high') or \
        col.endswith('_days_since_first_low') or \
        col.endswith('_days_since_cp') or \
        col.endswith('_is_changepoint')
    ]
    
    # 仅屏蔽 [期内] 高阶特征
    advanced_feature_columns_to_mask = [
        col for col in dict.fromkeys(accel_columns + duration_cp_columns)
        if col in df_enhanced.columns
    ]
    
    insufficient_samples_mask = (current_group_sizes < MIN_SAMPLES_FOR_ADV_FEATURES)
    
    if insufficient_samples_mask.any() and advanced_feature_columns_to_mask:
        df_enhanced.loc[insufficient_samples_mask, advanced_feature_columns_to_mask] = 0
        num_rows_masked = insufficient_samples_mask.sum()
        affected_phases = df_enhanced.loc[insufficient_samples_mask, 'phase'].unique()
        logger.warning(
            f"⚠️ 已屏蔽 {num_rows_masked} 行的 [期内] 高阶特征 "
            f"(不包括 _transition_shock)。(受影响: {', '.join(affected_phases)})"
        )
    elif insufficient_samples_mask.any():
        logger.debug("样本量不足，但未找到需要屏蔽的高阶特征列。")
    else:
        logger.info("✅ [V4.0] 所有phase组样本量充足，未触发高阶特征屏蔽")
    
    # --- 10. 恢复顺序与数据清理 ---
    df_enhanced = df_enhanced.sort_values(by='original_order').drop(columns='original_order')
    df_enhanced = df_enhanced.set_index('report_date')
    logger.debug("步骤 10: 恢复数据原始顺序并重置索引。")

    # --- 11.  智能填充策略 ---
    logger.debug("步骤 11: 执行智能填充策略 (仅填充衍生特征)...")

    if df_enhanced.columns.duplicated().any():
        duplicated_columns = df_enhanced.columns[df_enhanced.columns.duplicated()].unique().tolist()
        logger.warning(
            f"检测到重复特征列，保留首次出现的列并丢弃重复项: {duplicated_columns[:10]}"
        )
        df_enhanced = df_enhanced.loc[:, ~df_enhanced.columns.duplicated()]
    
    original_indicator_columns = [
        col for col in df_enhanced.columns 
        if not config.DERIVED_FEATURE_PATTERN.search(col)
        and col not in ['report_uuid', 'phase', 'id', 'patient_id']
    ]
    
    derived_feature_columns = [
        col for col in df_enhanced.columns 
        if config.DERIVED_FEATURE_PATTERN.search(col)
    ]
    
    if derived_feature_columns:
        df_enhanced[derived_feature_columns] = df_enhanced[derived_feature_columns].fillna(0)
        logger.info(f"已填充 {len(derived_feature_columns)} 个衍生特征的缺失值（填充为0）。")
    
    original_na_count = df_enhanced[original_indicator_columns].isna().sum().sum()
    if original_na_count > 0:
        logger.info(f"原始指标保留 {original_na_count} 个缺失值（不填充）")

    logger.info("===== 特征工程 (V4.0.3 - '双特征'策略版) 执行完毕 =====\n")
    return df_enhanced
