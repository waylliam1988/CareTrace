# explainability.py

import shap
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['mathtext.default'] = 'regular'

import logging
logger = logging.getLogger(__name__)


def set_chinese_font_for_matplotlib():
    """
    为 Matplotlib 设置一个支持中文的字体，以解决中文显示为方块的问题。
    """
    try:
        plt.rcParams['font.sans-serif'] = [
            'Microsoft YaHei', # 优先使用微软雅黑
            'SimHei',          # 其次使用黑体
            'Arial Unicode MS',# 备用
            'sans-serif'       # 默认
        ]
        # 解决设置字体后，坐标轴负号'-'显示为方块的问题
        plt.rcParams['axes.unicode_minus'] = False
        logger.debug("中文字体及负号显示已配置。")
    except Exception as e:
        logger.error(f"设置中文字体失败: {e}", exc_info=True)
        logger.error("请确保您的系统安装了 'Microsoft YaHei' (微软雅黑) 或 'SimHei' (黑体) 字体。")

@st.cache_data
def get_shap_explanation(
    _model, 
    background_data: pd.DataFrame, 
    weighted_data_point: pd.DataFrame,
    compute_interactions: bool = False
) -> dict:
    """
    为单个数据点生成完整的SHAP分析结果（V5.3 - 过滤污染特征）
    
    :param compute_interactions: 是否计算交互值（会牺牲准确性）
    """
    logger.info(f"开始生成 SHAP 解释（交互值: {compute_interactions}）...")
    logger.debug(f"  > 背景数据 shape: {background_data.shape}")
    logger.debug(f"  > 预测数据点 shape: {weighted_data_point.shape}")
    
    try:
        # 过滤污染特征
        transition_cols = [
            col for col in background_data.columns 
            if '_transition_shock' in col
        ]
        
        if transition_cols:
            logger.warning(
                f"⚠️ 检测到 {len(transition_cols)} 个污染特征（已自动过滤）：\n"
                f"  {', '.join(transition_cols[:3])}{'...' if len(transition_cols) > 3 else ''}\n"
                f"  【原因】这些特征包含跨阶段差分值，会导致 SHAP 归因错误\n"
                f"  【处理提示】在 controller 层调用前过滤这些特征"
            )
            
            # 过滤背景数据
            clean_background = background_data.drop(columns=transition_cols)
            # 过滤预测点（使用 .loc 避免警告）
            clean_data_point = weighted_data_point.loc[
                :, 
                weighted_data_point.columns.difference(transition_cols)
            ]
            
            logger.info(
                f"✅ SHAP 特征清洗完成：\n"
                f"  - 原始特征: {background_data.shape[1]}\n"
                f"  - 清洗后: {clean_background.shape[1]}\n"
                f"  - 已移除: {len(transition_cols)} 个跨界特征"
            )
        else:
            clean_background = background_data
            clean_data_point = weighted_data_point
            logger.debug("  > 未检测到污染特征，使用原始数据")
        
        # 特征严格对齐（使用清洗后的数据）
        aligned_data_point = clean_data_point.reindex(
            columns=clean_background.columns, 
            fill_value=0
        )
        
        if aligned_data_point.shape[1] != clean_background.shape[1]:
            logger.error(
                f"❌ 特征对齐失败！background: {clean_background.shape[1]} 列, "
                f"data_point: {aligned_data_point.shape[1]} 列"
            )
            return {}
        
        logger.debug(f"  > 特征对齐成功: {aligned_data_point.shape}")
        
        # 创建 SHAP 解释器（使用清洗后的背景数据）
        explainer = shap.TreeExplainer(
            _model, 
            clean_background,  # ← 使用清洗后的数据
            feature_perturbation="interventional",
            model_output="raw"
        )
        
        logger.debug("  > TreeExplainer 创建成功")
        
        # 计算 SHAP 值
        logger.debug("  > 正在计算 SHAP 值...")
        shap_values_obj = explainer(
            aligned_data_point,
            check_additivity=False
        )
        logger.debug("  > SHAP values 计算成功")
        
        # 按需计算交互值
        shap_interaction_values = None
        if compute_interactions:
            logger.debug("  > 正在计算 SHAP 交互值...")
            try:
                shap_interaction_values = explainer.shap_interaction_values(aligned_data_point)
                logger.debug("  > SHAP 交互值计算成功")
            except Exception as interact_e:
                logger.warning(f"⚠️ SHAP 交互值计算失败: {interact_e}")
        
        logger.info("✅ SHAP 解释生成完毕")
        return {
            "explainer": explainer,
            "shap_values_obj": shap_values_obj,
            "shap_interaction_values": shap_interaction_values
        }
        
    except Exception as e:
        logger.error(f"❌ SHAP 解释生成失败: {e}", exc_info=True)
        return {}


def plot_dependence(shap_interaction_values, weighted_data_point: pd.DataFrame, feature1: str, feature2: str):
    """
    为交互作用最强的两个特征生成并返回一个依赖图的matplotlib图形对象。
    """
    logger.info(f"正在绘制 SHAP 依赖图: {feature1} vs {feature2}")
    try:
        set_chinese_font_for_matplotlib() 
        fig, ax = plt.subplots()
        shap.dependence_plot(
            (feature1, feature2),
            shap_interaction_values,
            weighted_data_point,
            ax=ax,
            show=False
        )
        fig.tight_layout() # <-- 修改这里：在 fig 对象上调用
        logger.debug("SHAP 依赖图绘制成功。")
        return fig
    except Exception as e:
        logger.error(f"❌ SHAP 依赖图绘制失败: {e}", exc_info=True)
        # 返回一个空图表
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f'依赖图 ({feature1} / {feature2})\n渲染失败', 
                ha='center', va='center', fontsize=12, color='red')
        return fig


def plot_waterfall(shap_values, max_display: int = 15):
    """
    绘制 SHAP 贡献图 (V5 - 使用 Bar 替代 Waterfall)
    
    参数:
        shap_values: SHAP Explanation 对象 (通常是单一样本)
        max_display: 显示的最大特征数
    
    返回:
        matplotlib.figure.Figure: 贡献图对象
    """
    logger.info("正在绘制 SHAP 贡献图 (bar plot)...")
    try:
        # 1. 设置字体
        set_chinese_font_for_matplotlib()
        
        # 2. 使用 SHAP 的 bar plot（不传递 ax，让 SHAP 自己创建）
        shap.plots.bar(shap_values, max_display=max_display, show=False)
        
        # 3. 获取 SHAP 创建的 figure
        fig = plt.gcf()
        
        # 4. 调整布局
        fig.tight_layout()

        logger.debug("SHAP 贡献图 (bar plot) 绘制成功。")
        return fig
        
    except Exception as e:
        # 备用方案：手动创建简化版条形图
        logger.warning(f"⚠️ SHAP bar plot 渲染失败: {str(e)}。尝试手动绘制...", exc_info=True)
        
        try:
            # 提取 SHAP 值并手动绘制
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 获取特征名和 SHAP 值
            if hasattr(shap_values, 'values') and hasattr(shap_values, 'feature_names'):
                values = shap_values.values.flatten()
                features = shap_values.feature_names
                
                # 按绝对值排序
                abs_values = np.abs(values)
                sorted_idx = np.argsort(abs_values)[::-1][:max_display]
                
                # 绘制条形图
                colors = ['#ff0051' if v > 0 else '#008bfb' for v in values[sorted_idx]]
                y_pos = np.arange(len(sorted_idx))
                
                ax.barh(y_pos, values[sorted_idx], color=colors)
                ax.set_yticks(y_pos)
                ax.set_yticklabels([features[i] for i in sorted_idx])
                ax.set_xlabel('SHAP 值（对输出的影响）')
                ax.set_title('指标影响力分析')
                ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                
                fig.tight_layout()
                logger.info("✅ 手动绘制 SHAP 条形图成功")
                return fig
            else:
                raise ValueError("无法提取 SHAP 值")
                
        except Exception as fallback_e:
            logger.error(f"❌ 备用绘图也失败: {fallback_e}", exc_info=True)
            # 最终备用：错误提示图
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, '图表渲染失败\n请检查 SHAP 库版本', 
                    ha='center', va='center', fontsize=14, color='red')
            ax.set_xticks([])
            ax.set_yticks([])
            return fig


def analyze_feature_correlations(
    patient_data: pd.DataFrame,
    original_lab_indicators: list,
    top_n: int = 5
) -> pd.DataFrame:
    """分析基础化验指标之间的相关性"""
    logger.info("开始分析指标相关性...")
    
    try:
        # 1. 筛选出存在的基础指标列
        available_indicators = [
            col for col in original_lab_indicators 
            if col in patient_data.columns
        ]
        
        if len(available_indicators) < 2:
            logger.warning("可分析的基础指标不足（需要 ≥2 个）")
            return pd.DataFrame()
        
        # ✅ 【关键修改】过滤出每个指标至少有 10 个有效值的列
        valid_indicators = []
        for col in available_indicators:
            non_null_count = patient_data[col].notna().sum()
            if non_null_count >= 10:  # 提高到 10 个有效值
                valid_indicators.append(col)
            else:
                logger.debug(f"  > 跳过 {col}（仅 {non_null_count} 个有效值）")
        
        if len(valid_indicators) < 2:
            logger.warning(f"⚠️ 有效指标不足（需要至少 2 个，当前 {len(valid_indicators)} 个）")
            return pd.DataFrame()
        
        logger.debug(f"  > 共 {len(valid_indicators)} 个指标通过筛选（≥10 个有效值）")
        
        # 2. 计算相关性矩阵
        corr_matrix = patient_data[valid_indicators].corr()
        
        # ✅ 【新增】记录相关性矩阵的统计信息
        logger.debug(f"  > 相关性矩阵 shape: {corr_matrix.shape}")
        logger.debug(f"  > 相关性矩阵非 NaN 元素: {corr_matrix.notna().sum().sum()}")
        
        # 3. 提取上三角（避免重复）
        corr_flat = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        ).stack()
        
        # ✅ 【新增】过滤掉完全相关（abs = 1.0）和无效值
        corr_flat = corr_flat[
            (corr_flat.abs() < 0.9999) &  # 排除几乎完全相关的
            (corr_flat.notna())           # 排除 NaN
        ]
        
        if corr_flat.empty:
            logger.warning("⚠️ 过滤后无有效相关性数据（可能数据重复或配对不足）")
            return pd.DataFrame()
        
        # 4. 找到相关性最强的前 N 对
        top_corr = corr_flat.abs().nlargest(top_n)
        
        if top_corr.empty:
            logger.warning("未找到显著相关的指标对")
            return pd.DataFrame()
        
        # 5. 构建结果 DataFrame（保持原有逻辑）
        results = []
        for (feat1, feat2), abs_corr in top_corr.items():
            actual_corr = corr_matrix.loc[feat1, feat2]
            
            # ✅ 【新增】记录配对数量
            valid_pairs = patient_data[[feat1, feat2]].dropna()
            n_pairs = len(valid_pairs)
            
            logger.debug(
                f"  > {feat1} <-> {feat2}: "
                f"相关性={actual_corr:.3f}, 配对数={n_pairs}"
            )
            
            # 判断相关性强度
            if abs_corr > 0.7:
                strength = "强"
                emoji = "🔴"
            elif abs_corr > 0.4:
                strength = "中等"
                emoji = "🟡"
            else:
                strength = "弱"
                emoji = "🟢"
            
            results.append({
                'feature1': feat1,
                'feature2': feat2,
                'correlation': actual_corr,
                'abs_correlation': abs_corr,
                'strength': strength,
                'emoji': emoji,
                'direction': "正相关 ↗️" if actual_corr > 0 else "负相关 ↘️",
                'n_pairs': n_pairs  # ✅ 新增字段
            })
        
        results_df = pd.DataFrame(results)
        logger.info(f"✅ 找到 {len(results_df)} 对显著相关的指标")
        
        return results_df
        
    except Exception as e:
        logger.error(f"❌ 相关性分析失败: {e}", exc_info=True)
        return pd.DataFrame()


def get_top_shap_features(
    shap_values_obj,
    original_lab_indicators: list,
    top_n: int = 10
) -> pd.DataFrame:
    """
    提取 SHAP 主效应最强的基础指标
    
    参数:
        shap_values_obj: SHAP Explanation 对象
        original_lab_indicators: 基础指标名称列表
        top_n: 返回前 N 个最重要的特征
    
    返回:
        包含 (feature, impact) 的 DataFrame
    """
    logger.info("开始提取主效应最强的基础指标...")
    
    try:
        if shap_values_obj is None:
            logger.warning("SHAP 值对象为 None")
            return pd.DataFrame()
        
        # 1. 提取 SHAP 值
        shap_values_df = pd.DataFrame(
            shap_values_obj.values,
            columns=shap_values_obj.feature_names
        )
        
        # 2. 只保留基础指标
        available_base_features = [
            col for col in original_lab_indicators 
            if col in shap_values_df.columns
        ]
        
        if not available_base_features:
            logger.warning("没有可用的基础指标")
            return pd.DataFrame()
        
        # 3. 计算绝对 SHAP 值并排序
        base_shap = shap_values_df[available_base_features].abs().iloc[0]
        top_features = base_shap.nlargest(top_n)
        
        # 4. 构建结果 DataFrame
        results_df = pd.DataFrame({
            'feature': top_features.index,
            'impact': top_features.values
        }).reset_index(drop=True)
        
        logger.info(f"✅ 提取了 {len(results_df)} 个最重要的基础指标")
        
        return results_df
        
    except Exception as e:
        logger.error(f"❌ 主效应提取失败: {e}", exc_info=True)
        return pd.DataFrame()



def find_top_interaction_features(shap_interaction_values, feature_names: list) -> tuple:
    """
    从SHAP交互矩阵中，找出交互效应绝对值最大的特征对。

    :param shap_interaction_values: SHAP交互值矩阵 (通常是三维，我们取第一个样本)。
    :param feature_names: 特征名称列表。
    :return: 一个元组，包含 (feature1, feature2, interaction_value)。
    """
    logger.debug("正在查找顶级 SHAP 交互特征...")
    try:
        interaction_matrix = shap_interaction_values[0]
        # 将对角线（自身与自身的交互）清零，我们只关心不同特征间的交互
        np.fill_diagonal(interaction_matrix, 0)
        
        # 创建上三角掩码（避免重复计算 A-B 和 B-A）
        # 同时确保对角线被排除
        upper_triangle_mask = np.triu(np.ones_like(interaction_matrix, dtype=bool), k=1)
        
        # 只在上三角区域寻找最大值
        masked_matrix = np.where(upper_triangle_mask, np.abs(interaction_matrix), -np.inf)
        max_idx = np.unravel_index(np.argmax(masked_matrix), masked_matrix.shape)
        
        # 验证：确保不是对角线（防御性编程）
        if max_idx[0] == max_idx[1]:
            logger.warning("⚠️ 检测到对角线交互（不应发生），返回安全回退值")
            return "N/A", "N/A", 0.0
        
        # 验证：检查是否找到了有效的交互（不是-inf）
        if masked_matrix[max_idx] == -np.inf:
            logger.warning("⚠️ 未找到有效的特征交互（所有非对角线值可能都为0）")
            return "N/A", "N/A", 0.0
        
        feature1 = feature_names[max_idx[0]]
        feature2 = feature_names[max_idx[1]]
        interaction_value = interaction_matrix[max_idx]
        
        logger.debug(f"  > 找到顶级交互: {feature1} <-> {feature2} (Value: {interaction_value:.4f})")
        
        # 最终检查：确保两个特征确实不同
        if feature1 == feature2:
            logger.error(f"❌ 严重错误：找到了相同的特征对 ({feature1})，这不应发生！")
            return "N/A", "N/A", 0.0
        
        return feature1, feature2, interaction_value
        
    except Exception as e:
        logger.error(f"❌ 查找顶级 SHAP 交互特征失败: {e}", exc_info=True)
        # 返回一个安全的回退值
        return "N/A", "N/A", 0.0
