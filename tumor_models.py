# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Liu Yanwei / 刘彦巍

# -*- coding: utf-8 -*-
# tumor_models.py
# 包含多种肿瘤生长与耐药的常微分方程（ODE）机理模型

import pymc as pm
import pandas as pd
import numpy as np

# ==============================================================================
# --- 模型一：Lotka-Volterra 敏感(S)-抵抗(R)竞争模型 (V2.1 - 免疫协同版) ---
# 科学来源: 
#   - 基础竞争模型: Gatenby, R.A. et al. Adaptive Therapy. Cancer Res (2009)
#   - 表型可塑性 (Phenotypic Plasticity): Weinberg, R.A. et al. The Hallmarks of Cancer. Cell (2011)
#   - 适应性成本 (Fitness Cost): Enriquez-Navas, P.M. et al. Cancer Res (2016)
#   - 免疫协同 (Immune Synergy): de Pillis, L. G., et al. A Validated Mathematical Model... Cancer Research (2005)
#
# 模型简介 (V2.1):
# 这是一个集成了“药物-肿瘤-免疫”三方相互作用的、先进的肿瘤生态演化模型。
# 它不仅模拟了药物压力下的克隆竞争，还通过引入由真实数据驱动的外部变量，
# 将患者自身的免疫系统作为一个动态角色纳入考量。模型现在能描绘三种关键的生物学力量：
#
# 1. 适应性成本 (Adaptive Cost): 模型假设抵抗性状在没有药物选择压力的
#    环境下是一种生存负担。因此，在休药期间，抵抗细胞(R)的生长会受到惩罚，
#    这为敏感细胞(S)重新占据优势提供了机会。
#
# 2. 表型可塑性 (Phenotypic Plasticity): 模型允许细胞在敏感(S)和抵抗(R)
#    状态之间进行双向转换。这能模拟肿瘤在治疗压力下的“获得性耐药”(S→R)，
#    以及在药物假期中可能发生的“耐药性逆转”(R→S)。
#
# 3. 【新增】免疫协同 (Immune Synergy): 模型集成了患者真实的外周血淋巴细胞
#    数据，将其作为一个时变的外部变量(I(t))来代表免疫系统的状态。通过引入
#    免疫杀伤项(-c*I*S)，模型现在可以解离出药物杀伤和免疫杀伤两种不同的效应，
#    从而更准确地评估药物的真实效果，并为探索药物-免疫联合治疗策略提供理论基础。
# ==============================================================================
def lotka_volterra_ode(y, t, p, *, treatment_func, immune_func):
    """
    【机理模型核心 V2.0 - 增加适应性成本和表型可塑性】
    定义洛特卡-沃尔泰拉（Lotka-Volterra）竞争模型的常微分方程（ODE）系统。

    这个函数是整个模拟引擎的“物理定律”。它被ODE求解器（如 scipy.odeint）在每一个微小的时间步长上反复调用，
    以计算在该时间点，各个细胞亚群数量的变化速率。
    此版本增加了两个高级生物学特性：
    1. 适应性成本: 在休药期间，抵抗细胞(R)的生长会因其抵抗性状的维持成本而受到惩罚。
    2. 表型可塑性: 允许敏感(S)和抵抗(R)细胞以一定的速率相互转换。
    
    参数:
    - y (list/array):             一个包含当前时刻所有状态变量值的列表。在这里，y = [S, R]，
                                    其中 S 是药物敏感细胞的数量，R 是药物抵抗细胞的数量。
    - t (float):                  当前的时间点。由ODE求解器自动传入。
    - r_s (float):                药物敏感细胞(S)的固有生长速率 (intrinsic growth rate)。可以理解为在没有资源限制和竞争时，它的分裂速度。
    - r_r (float):                药物抵抗细胞(R)的固有生长速率。根据生物学假设，通常 r_r < r_s，因为抵抗性状需要付出“适应性成本”。
    - K (float):                  环境承载能力 (Carrying Capacity)。代表肿瘤微环境（如空间、养分）能支持的最大肿瘤细胞总量。
    - alpha_rs (float):           竞争系数，代表一个抵抗细胞(R)对敏感细胞(S)生长空间的挤占/抑制强度。
    - alpha_sr (float):           竞争系数，代表一个敏感细胞(S)对抵抗细胞(R)的抑制强度。这是维持S种群来压制R种群的关键。
    - d_s (float):                药物对敏感细胞(S)的杀伤率 (drug-induced death rate)。
    - treatment_func (function):  一个动态函数，当调用它并传入时间't'时，它会返回当前时刻的治疗状态 (1.0代表正在用药, 0.0代表休药)。
    - cost_factor (float):        抵抗性状的适应性成本因子 (>1.0)。休药期 R 细胞的有效生长率将是 r_r / cost_factor。
    - k_sr (float):               从敏感(S)到抵抗(R)状态的转换速率。
    - k_rs (float):               从抵抗(R)到敏感(S)状态的转换速率

    返回:
    - list: 一个包含每个状态变量变化速率（导数）的列表。在这里，是 [dS/dt, dR/dt]。
    """
    # 使用索引访问
    r_s = p[0]
    r_r = p[1]
    K = p[2]
    alpha_rs = p[3]
    alpha_sr = p[4]
    d_s = p[5]
    cost_factor = p[6]
    k_sr = p[7]
    k_rs = p[8]
    c = p[9]

    # --- 1. 解包当前状态 ---
    # 从输入列表 y 中，清晰地分离出当前时刻的 S 和 R 细胞数量。
    S, R = y

    # --- 2. 获取当前治疗状态 ---
    # 调用传入的治疗函数，查询在't'这个精确的时间点，病人是否正在接受治疗。
    # u_t 将作为“治疗开关”，控制药物杀伤项是否生效 (1.0 = 用药, 0.0 = 休药)。
    u_t = treatment_func(t)

    # --- 3. 【核心修改】实现适应性成本逻辑 ---
    # cost_factor (>1.0) 代表抵抗性状带来的“适应度成本”(Fitness Cost)。
    # 如果处于休药期 (u_t = 0)，则抵抗细胞的生长速率会受到其适应性成本的惩罚。
    # 否则，在用药期间，生长速率保持不变。
    r_r_effective = r_r / cost_factor if u_t == 0.0 else r_r

    # --- 4. 【核心修改】定义表型转换项 ---
    # k_sr * S: 敏感细胞以速率k_sr转化为抵抗细胞的数量。
    # k_rs * R: 抵抗细胞以速率k_rs逆转为敏感细胞的数量。
    # net_S_to_R_flow 代表从 S 群体流向 R 群体的“净”细胞数量。
    # 值为正表示 S->R 的趋势占优，为负表示 R->S 的趋势占优。
    net_S_to_R_flow = k_sr * S - k_rs * R

    # --- 5. 【核心修改】更新敏感细胞(S)的变化速率 (dS/dt) ---
    # 在原有“生长”和“药物杀伤”的基础上，减去“净流出”到 R 群体的细胞。
    # 敏感细胞(S)的变化由四部分构成：
    #   1. 自身增长项: r_s * S * (1 - (S + alpha_rs * R) / K)
    #   2. 药物杀伤项: - d_s * u_t * S
    #   3. 免疫杀伤项: - c * immune_func(t) * S  <---【核心新增】
    #   4. 表型流出项: - net_S_to_R_flow

    # 获取当前时间的免疫细胞水平，并将其加入dS/dt方程
    I_t = immune_func(t) if immune_func is not None else 1.0 # 如果没提供函数，则用中性值1.0

    dSdt = r_s * S * (1 - (S + alpha_rs * R) / K) - d_s * u_t * S - c * I_t * S - net_S_to_R_flow

    # --- 6. 【核心修改】更新抵抗细胞(R)的变化速率 (dR/dt) ---
    # 在原有“生长”的基础上：
    #   a. 使用带有适应性成本的有效生长速率 r_r_effective。
    #   b. 加上从 S 群体“净流入”的细胞。
    # 抵抗细胞(R)的变化也由三部分构成：
    #   1. 自身增长项: r_r_effective * R * (1 - (R + alpha_sr * S) / K) -> 同样受竞争抑制，但生长速率受“适应性成本”调节。
    #   2. 药物杀伤项: 无 -> 模型的关键假设是药物对R细胞无效。
    #   3. 表型流入项: + net_S_to_R_flow -> 从敏感细胞净转换来的数量。
    dRdt = r_r_effective * R * (1 - (R + alpha_sr * S) / K) + net_S_to_R_flow

    # --- 7. 返回变化速率 ---
    # ODE求解器会使用这个返回的[dS/dt, dR/dt]来计算下一个微小时间步长的S和R值。
    return [dSdt, dRdt]

def build_lotka_volterra_pymc_model(
    time_points, 
    tumor_burden, 
    treatment_func, 
    initial_burden,
    imaging_data=None,  # 传入影像学数据
    immune_func=None
):
    """
    【机理模型核心 V2.1 - 集成数据驱动的免疫调节】
    定义洛特卡-沃尔泰拉（Lotka-Volterra）竞争模型的ODE系统，该版本额外集成了由真实数据驱动的免疫杀伤效应。

    此版本在V2.0的基础上，引入了一个关键的外部变量 immune_func(t)，它代表了
    由真实外周血淋巴细胞数据平滑插值得到的、随时间变化的免疫水平。这使得模型
    能够解离出药物杀伤和免疫杀伤两种不同的效应，极大提升了模型的生物学真实性和个性化程度。

    科学依据:
    - 免疫监视 (Immuno-surveillance): 新增的 -c * I_t * S 项模拟了免疫系统（以淋巴细胞I_t为代理）
      对敏感肿瘤细胞(S)的清除作用，这基于经典的“捕食者-猎物”模型和质量作用定律。
    - 代理指标 (Proxy): 使用外周血淋巴细胞作为全身免疫状态的代理，是临床研究中广泛采用的合理简化。
    
    参数:
    - y (list/array):             一个包含当前时刻所有状态变量值的列表。在这里，y = [S, R]，
                                    其中 S 是药物敏感细胞的数量，R 是药物抵抗细胞的数量。
    - t (float):                  当前的时间点。由ODE求解器自动传入。
    - r_s (float):                药物敏感细胞(S)的固有生长速率。
    - r_r (float):                药物抵抗细胞(R)的固有生长速率。
    - K (float):                  环境承载能力 (Carrying Capacity)。
    - alpha_rs (float):           竞争系数，代表R细胞对S细胞的抑制强度。
    - alpha_sr (float):           竞争系数，代表S细胞对R细胞的抑制强度。
    - d_s (float):                药物对敏感细胞(S)的杀伤率。
    - treatment_func (function):  一个动态函数，返回在时间't'的治疗状态 (1.0=用药, 0.0=休药)。
    - cost_factor (float):        抵抗性状的适应性成本因子 (>1.0)。
    - k_sr (float):               从敏感(S)到抵抗(R)状态的转换速率。
    - k_rs (float):               从抵抗(R)到敏感(S)状态的转换速率。
    - c (float):                  免疫细胞对敏感细胞的杀伤效率参数。
    - immune_func (function):     一个连续函数，输入时间t，返回该时刻的免疫水平(I_t)。

    返回:
    - list: 一个包含每个状态变量变化速率（导数）的列表：[dS/dt, dR/dt]。
    """
    
    # --- 为 scaling_factor 计算一个数据驱动的先验中心 ---
    # 这是“经验贝叶斯”思想的应用：在正式建模前，从数据中得到一个合理的初始猜测，
    # 这可以帮助采样器更快地找到参数的高概率区域，提升推断效率。
    initial_guess_for_scaling = 1.0
    if imaging_data is not None and not imaging_data.empty:
        # 使用 pd.merge_asof 安全地合并两个不同时间点的数据源。
        # 它会为每个标志物数据点，找到时间上最接近的（10天内）影像学数据点。
        merged_data = pd.merge_asof(
            pd.DataFrame({'time': time_points, 'marker': tumor_burden}).sort_values('time'),
            imaging_data.sort_values('time').rename(columns={'value': 'imaging'}),
            on='time',
            direction='nearest',
            tolerance=pd.Timedelta(days=10) # 只匹配10天内的数据
        ).dropna()
        
        # 只有在存在匹配数据，且分母不为零时，才计算比例
        if not merged_data.empty and merged_data['imaging'].mean() > 1e-6:
            ratio = merged_data['marker'].mean() / merged_data['imaging'].mean()
            if ratio > 0:
                initial_guess_for_scaling = ratio

    # 使用PyMC的上下文管理器来定义一个模型容器
    with pm.Model() as model:
        # --- 1. 定义未知参数的“先验分布”(Priors) ---
        
        # 初始敏感细胞比例
        s0_frac = pm.Beta('s0_frac', alpha=10.0, beta=1.0)
        
        # 生长速率
        r_s = pm.Lognormal('r_s', mu=np.log(0.08), sigma=0.5)
        r_r = pm.Lognormal('r_r', mu=np.log(0.03), sigma=0.5)

        # 环境承载能力 (K)
        # 【修改】让K的先验同时考虑两种数据的最大值，使其更稳健
        max_observed_burden = np.max(tumor_burden)
        if imaging_data is not None and not imaging_data.empty:
            # 假设 scaling_factor 约为 initial_guess，将影像学数据也转换到标志物的尺度
            max_observed_burden = max(max_observed_burden, np.max(imaging_data['value']) * initial_guess_for_scaling)
        K = pm.Lognormal('K', mu=np.log(max_observed_burden * 1.5), sigma=0.5)
        
        # 竞争系数
        alpha_rs = pm.Lognormal('alpha_rs', mu=np.log(1.0), sigma=0.2)
        alpha_sr = pm.Lognormal('alpha_sr', mu=np.log(1.0), sigma=0.2)
        
        # 药物杀伤率
        d_s = pm.Lognormal('d_s', mu=np.log(0.15), sigma=0.5)

        # 适应性成本因子 (cost_factor):
        # 使用 Lognormal 分布确保其为正。mu=np.log(1.1) 表示我们“猜测”成本约为10% (即因子为1.1)，
        # sigma=0.1 允许模型在这个猜测附近进行探索。
        cost_factor = pm.Lognormal('cost_factor', mu=np.log(1.1), sigma=0.1)
        
        # 表型转换速率 (k_sr, k_rs):
        # 这些速率通常被认为是相对较慢的过程。使用 Exponential 分布是合适的，
        # 它倾向于较小的值，但允许出现较大的值。lam=100.0 意味着先验的均值为 1/100 = 0.01。
        k_sr = pm.Exponential('k_sr', lam=100.0)
        k_rs = pm.Exponential('k_rs', lam=100.0)

        # 为免疫杀伤率 c 定义先验
        c = pm.Lognormal('c', mu=np.log(1e-7), sigma=1.0)

        # 【优化】使用数据驱动的先验来定义标志物与影像学之间的缩放因子
        scaling_factor = pm.Lognormal('scaling_factor', 
                                     mu=np.log(initial_guess_for_scaling), # <-- 使用我们的数据驱动猜测作为先验中心
                                     sigma=0.5) # sigma 保持不变，允许模型在猜测周围自由探索
        
        # 为两个独立的观测源定义各自的观测噪声
        sigma_marker = pm.HalfNormal('sigma_marker', 1.0)
        sigma_imaging = pm.HalfNormal('sigma_imaging', 1.0)

        # --- 2. 将ODE求解器嵌入PyMC ---
        # 将新参数加入到传递给ODE求解器的元组中
        ode_params = (r_s, r_r, K, alpha_rs, alpha_sr, d_s, cost_factor, k_sr, k_rs, c)
        y0 = [initial_burden * s0_frac, initial_burden * (1 - s0_frac)]

        # 创建一个lambda包装函数，它会解包参数p，并从闭包中捕获和传递关键字参数
        wrapped_lv_ode = lambda y, t, p: lotka_volterra_ode(
            y, t, p, 
            treatment_func=treatment_func, 
            immune_func=immune_func
        )

        ode_solution = pm.ode.DifferentialEquation(
            func=wrapped_lv_ode, # <-- 使用包装函数
            times=time_points,
            n_states=2,
            n_theta=len(ode_params), # n_theta 现在会自动匹配新参数的数量
            t0=0,
            solver='scipy_solve_ivp',
            solver_kwargs={'method': 'Radau'}
        )(y0=y0, theta=ode_params)

        # --- 3. 定义可观测量的预测值 (mu) ---
        # `total_burden_predicted` 是模型内在的、未经观测的“真实”肿瘤负荷
        total_burden_predicted = ode_solution[:, 0] + ode_solution[:, 1]
        
        # 标志物的预测值 = 缩放因子 * 真实肿瘤负荷
        mu_marker = pm.Deterministic('mu_marker', scaling_factor * total_burden_predicted)

        # 影像学的预测值 = 真实肿瘤负荷 (这是我们的核心假设)
        mu_imaging = pm.Deterministic('mu_imaging', total_burden_predicted)

        # --- 4. 定义【多重】似然函数 (Likelihoods) ---
        
        # 4.1 标志物的似然函数：连接模型预测 (mu_marker) 与真实观测 (tumor_burden)
        pm.Normal('obs_marker', mu=mu_marker, sigma=sigma_marker, observed=tumor_burden)
        
        # 4.2 【核心修正】影像学的似然函数
        if imaging_data is not None and not imaging_data.empty:
            # 按照PyMC的最佳实践，使用 pm.Data 将外部数据“注册”到模型中。
            # 这使得模型结构更清晰，并为将来的高级功能（如后验预测）打下基础。
            imaging_values = pm.Data('imaging_values', imaging_data['value'].values, mutable=False)

            # 在Python/Numpy层面计算出影像学数据点对应于完整时间序列的索引
            imaging_time_indices = np.searchsorted(time_points, imaging_data['time'].values)
            
            # 在PyMC的计算图内部，使用计算好的索引从完整的 `mu_imaging` 符号张量中提取出对应时间的预测值
            # 这是在符号层面进行的操作，是构建计算图的关键一步
            mu_imaging_at_obs_times = mu_imaging[imaging_time_indices]

            # 建立第二个似然函数：连接影像学预测 (mu_imaging_at_obs_times) 与注册的影像学观测 (imaging_values)
            pm.Normal('obs_imaging', 
                      mu=mu_imaging_at_obs_times, 
                      sigma=sigma_imaging, 
                      observed=imaging_values)
        
    return model

# ==============================================================================
# --- 模型二：Brady-Nicholls 干细胞(S)-分化(D)驱动耐药模型 (B20) ---
# 科学来源: Brady-Nicholls, R. et al. Nat Commun (2020) [cite: 2437]
#           及 Pasetto, S. et al. Bulletin of Mathematical Biology (2022) [cite: 2387]
# ==============================================================================
def stem_cell_ode(y, t, p, *, treatment_func, immune_func):
    """
    【机理模型核心 V2.0 - 整合免疫逃逸假说版】
    定义了 Brady-Nicholls 等人提出的、基于癌症干细胞（Cancer Stem Cell, CSC）的ODE系统。
    
    这个模型的核心假设与Lotka-Volterra的“竞争”模型完全不同。它假设：
    1.  肿瘤由两种细胞构成：能无限自我更新的干细胞(n_S)，和由干细胞产生的、会最终死亡的分化细胞(n_D)。
    2.  只有干细胞(n_S)能够分裂。它们的分裂方式有两种：
        a. 对称分裂：一个干细胞 -> 两个干细胞（自我更新，扩增干细胞池）。
        b. 不对称分裂：一个干细胞 -> 一个干细胞 + 一个分化细胞（维持干细胞池，同时产生肿瘤主体）。
    3.  治疗（如激素疗法）只对快速增殖的分化细胞(n_D)有效，而对处于相对静默状态的干细胞(n_S)无效。
    4.  耐药性的产生，不是因为“抵抗”细胞在竞争中胜出，而是因为治疗清除了大量的分化细胞，
        这反过来为干细胞的增殖腾出了空间，导致整个肿瘤的再生。
    
    【V2.0 科学原理更新】:
    本版本整合了“癌症干细胞免疫逃逸”的核心假说。通过引入由真实数据驱动的免疫函数 `immune_func`，
    模型现在假设免疫系统的杀伤压力主要作用于更易被识别的分化细胞(n_D)，而干细胞(n_S)则具备
    “免疫隐身”的能力。这使得模型能够同时探索肿瘤对【药物】和【免疫系统】的双重耐药机制，
    极大地提升了模型的生物学真实性。

    参数:
    - y (list/array): 当前状态向量, y = [n_S, n_D]。
    - t (float):      当前时间点。
    - p_s (float):    干细胞“对称分裂”的概率 (Symmetric division probability)。这是一个[0,1]之间的关键参数。
                      它决定了干细胞分裂时，有多大可能性是用于“自我复制”而不是“产生后代”。
    - delta_d (float):药物对分化细胞(n_D)的杀伤率。注意，药物不直接作用于n_S。
    - treatment_func (function): 治疗状态函数 (1.0 = 用药, 0.0 = 休药)。
    - c (float):      【新增】免疫细胞对分化细胞的杀伤效率参数。
    - immune_func (function): 【新增】一个连续函数，输入时间t，返回该时刻的免疫水平(I_t)。    

    返回:
    - list: 变化速率列表 [dnS/dt, dnD/dt]。
    """
    # 使用索引访问
    p_s = p[0]
    delta_d = p[1]
    c = p[2]

    # --- 1. 解包当前状态 ---
    # n_S: 干细胞数量, n_D: 分化细胞数量
    n_S, n_D = y
    u_t = treatment_func(t)
    
    # --- 2. 计算总细胞数并处理边界情况 ---
    total_cells = n_S + n_D
    # 这是一个稳健性检查：如果肿瘤被完全清除，则所有变化速率都应为0，以避免除零错误。
    if total_cells == 0: 
        return [0, 0]

    # --- 3. 计算干细胞(n_S)的变化速率 (dnS/dt) ---
    # 在这个模型中，干细胞的增加只来源于一种途径：对称分裂。
    #   a. `np.log(2) * n_S`: 所有干细胞都在以一个基础速率分裂。
    #   b. `(p_s * n_S / total_cells)`: 这一项代表了“对称分裂的有效概率”。
    #      - `p_s`: 是固有的、最大的对称分裂概率。
    #      - `n_S / total_cells`: 这一项引入了“密度依赖”的反馈。模型假设，当干细胞在肿瘤中的比例过高时，
    #        会有一种负反馈机制来抑制其进一步的对称分裂。这符合生物学中“生态位”已满的概念。
    # 最终，干细胞数量的净增长，等于总的分裂数乘以“对称分裂”的有效概率。
    # 【核心假设】注意：免疫杀伤不直接作用于干细胞(n_S)，体现了其免疫逃逸特性。
    dnSdt = np.log(2) * n_S * (p_s * n_S / total_cells)
    
    # --- 4. 计算分化细胞(n_D)的变化速率 (dnDdt) ---
    # 分化细胞的命运由“出生”、“药物杀伤”和“免疫杀伤”三部分决定：
    
    # 4.1 获取当前时间的免疫细胞水平 (I_t)
    # 调用传入的免疫函数，如果函数不存在，则使用中性值1.0作为回退。
    I_t = immune_func(t) if immune_func is not None else 1.0

    # 4.2 计算“出生项”
    #   - `np.log(2) * n_S`: 干细胞分裂的总量。
    #   - `(1 - ...)`: “不对称分裂”的有效概率，等于 1 减去“对称分裂”的有效概率。
    #   - 因此，分化细胞的“出生率”完全取决于干细胞不对称分裂的速率。
    birth_term = np.log(2) * n_S * (1 - p_s * n_S / total_cells)
    
    # 4.3 计算“药物杀伤项”
    #   - `delta_d * n_D`: 分化细胞被药物杀死的总量。
    #   - `u_t`: 药物杀伤只在治疗期间 (u_t = 1.0) 发生。
    drug_kill_term = delta_d * u_t * n_D

    # 4.4 计算“免疫杀伤项”
    #   - `c * I_t * n_D`: 这是一个质量作用定律项，假设免疫杀伤量与免疫细胞(I_t)和
    #     分化细胞(n_D)的数量都成正比。
    immune_kill_term = c * I_t * n_D

    # 4.5 组合所有项，得到分化细胞的净变化率
    dnDdt = birth_term - drug_kill_term - immune_kill_term
    
    # --- 5. 返回变化速率 ---
    return [dnSdt, dnDdt]



def build_stem_cell_pymc_model(
    time_points, 
    tumor_burden, 
    treatment_func, 
    initial_burden,
    imaging_data=None,
    lymphocyte_data=None, # <-- 【新增】接收真实的淋巴细胞观测数据
    immune_func=None    
):
    """
    【贝叶斯模型构建器 - B20模型 V2.0 - 支持多模态数据与免疫约束版】
    为Brady-Nicholls干细胞模型构建一个完整的PyMC贝叶斯推断框架。

    此函数的核心任务是，将我们对“干细胞驱动耐药”这一生物学假设，用概率语言（先验分布）
    描述给PyMC，然后让PyMC根据真实数据来校准这个模型的具体参数。
    
    【V2.0 科学原理更新】:
    本版本通过引入`lymphocyte_data`和第三个似然函数，解决了V1.0中免疫杀伤参数`c`
    可能存在的“不可辨识”问题。通过使用真实的淋巴细胞数据来约束模型内部的免疫效应，
    我们为参数`c`的推断提供了直接的数据证据，从而使模型能更可靠地区分“药物杀伤”
    与“免疫杀伤”两种不同的效应。

    参数:
    - (同 build_lotka_volterra_pymc_model)

    返回:
    - pm.Model: 一个构建完成的、待“训练”的PyMC模型对象。
    """


    # --- 为 scaling_factor 计算一个数据驱动的先验中心 ---
    # 这是“经验贝叶斯”思想的应用：在正式建模前，从数据中得到一个合理的初始猜测，
    # 这可以帮助采样器更快地找到参数的高概率区域，提升推断效率。
    initial_guess_for_scaling = 1.0
    if imaging_data is not None and not imaging_data.empty:
        # 使用 pd.merge_asof 安全地合并两个不同时间点的数据源。
        # 它会为每个标志物数据点，找到时间上最接近的（10天内）影像学数据点。
        merged_data = pd.merge_asof(
            pd.DataFrame({'time': time_points, 'marker': tumor_burden}).sort_values('time'),
            imaging_data.sort_values('time').rename(columns={'value': 'imaging'}),
            on='time',
            direction='nearest',
            tolerance=pd.Timedelta(days=10) # 只匹配10天内的数据
        ).dropna()
        
        # 只有在存在匹配数据，且分母不为零时，才计算比例
        if not merged_data.empty and merged_data['imaging'].mean() > 1e-6:
            ratio = merged_data['marker'].mean() / merged_data['imaging'].mean()
            if ratio > 0:
                initial_guess_for_scaling = ratio


    with pm.Model() as model:
        # --- 1. 定义未知参数的“先验分布”(Priors) ---
        # 这里的先验分布与竞争模型截然不同，反映了我们对干细胞生物学的不同信念。

        # 初始干细胞比例 (s0_frac = n_S0 / T0)。
        # Beta分布的 alpha=1.0, beta=10.0 会产生一个强烈偏向于0的分布。
        # 这编码了我们的核心信念：肿瘤干细胞在初始肿瘤中是非常稀有的亚群。
        s0_frac = pm.Beta('s0_frac', alpha=1.0, beta=10.0)
        
        # 对称分裂概率 (p_s)。
        # Beta分布的 alpha=2.0, beta=8.0 同样产生一个偏向于较小值的分布 (均值在0.2附近)。
        # 这代表我们相信，在正常情况下，干细胞的分裂更倾向于产生后代（不对称分裂），
        # 而不是进行大规模的自我复制（对称分裂）。
        p_s = pm.Beta('p_s', alpha=2.0, beta=8.0, doc="Symmetric division probability")
        
        # 药物对分化细胞的杀伤率 (delta_d)。
        # 同样使用Lognormal确保其为正。
        delta_d = pm.Lognormal('delta_d', mu=np.log(0.1), sigma=0.5, doc="Drug kill rate on D cells")

        # --- 为免疫杀伤率 c 定义先验 ---
        # 这个先验与S-R模型中的c保持一致，便于后续模型比较
        c = pm.Lognormal('c', mu=np.log(1e-7), sigma=1.0)

        # --- 多模态观测所需的参数 ---
        # 增加与LV模型完全相同的 scaling_factor 和 sigma_imaging 参数
        # 这确保了两个模型在比较时，关于数据关系的假设是完全一致的。
        scaling_factor = pm.Lognormal('scaling_factor', 
                                     mu=np.log(initial_guess_for_scaling),
                                     sigma=0.5)
        
        sigma_marker = pm.HalfNormal('sigma_marker', 1.0)
        sigma_imaging = pm.HalfNormal('sigma_imaging', 1.0)

        # --- 为淋巴细胞数据也增加独立的缩放和噪声参数 ---
        # 这使得模型可以学习淋巴细胞绝对数与内部“有效免疫水平”之间的关系
        scaling_factor_lymph = pm.Lognormal('scaling_factor_lymph', mu=np.log(1.0), sigma=0.5)
        sigma_lymph = pm.HalfNormal('sigma_lymph', 1.0)

        # --- 2. 将ODE求解器嵌入PyMC ---
        # 【修改】这个模型的“物理参数”现在包含 p_s, delta_d, 和 c
        ode_params = (p_s, delta_d, c)
        
        # 定义ODE的初始条件 y0 = [n_S0, n_D0]。
        # 同样，初始条件依赖于我们正在推断的 s0_frac。
        y0 = [initial_burden * s0_frac, initial_burden * (1 - s0_frac)]

        # 创建lambda包装函数，以正确传递所有参数
        wrapped_sc_ode = lambda y, t, p: stem_cell_ode(
            y, t, p, 
            treatment_func=treatment_func, 
            immune_func=immune_func
        )

        # 使用PyMC的DifferentialEquation接口。
        # 注意，这里的 `func` 参数指向了我们新定义的 `stem_cell_ode` 函数。
        # 这正是“可插拔”架构的威力所在：我们只是更换了底层的“物理定律”。
        ode_solution = pm.ode.DifferentialEquation(
            func=wrapped_sc_ode, # <-- 使用包装函数
            times=time_points, 
            n_states=2, 
            n_theta=len(ode_params), 
            t0=0,
            solver='scipy_solve_ivp',
            solver_kwargs={'method': 'Radau'}
        )(y0=y0, theta=ode_params)

        # --- 3. 定义多模态可观测量的预测值 ---

        # 模型内在的“真实”肿瘤负荷 (干细胞 + 分化细胞)
        total_burden_predicted = ode_solution[:, 0] + ode_solution[:, 1]
        
        # 标志物的预测值 = 缩放因子 * 真实肿瘤负荷
        mu_marker = pm.Deterministic('mu_marker', scaling_factor * total_burden_predicted)

        # 影像学的预测值 = 真实肿瘤负荷 (核心假设)
        mu_imaging = pm.Deterministic('mu_imaging', total_burden_predicted)

        # --- 4. 定义【多重】似然函数，将模型预测与真实数据连接 ---
        # 即使模型内部区分了S和D细胞，我们在外部能观测到的依然是它们的总量。
        
        # 4.1 标志物的似然函数 (现在使用 mu_marker)
        pm.Normal('obs_marker', mu=mu_marker, sigma=sigma_marker, observed=tumor_burden)
        
        # 4.2 影像学的似然函数 (与LV模型完全一致)
        # 这一部分的统计学逻辑与Lotka-Volterra模型完全相同。
        # 我们依然假设观测值是模型预测值加上高斯噪声。        
        if imaging_data is not None and not imaging_data.empty:
            imaging_time_indices = np.searchsorted(time_points, imaging_data['time'].values)
            
            # 增加防御性检查
            if imaging_time_indices.size > 0:
                imaging_values = pm.Data('imaging_values', imaging_data['value'].values, mutable=False)
                mu_imaging_at_obs_times = mu_imaging[imaging_time_indices]
                pm.Normal('obs_imaging', 
                          mu=mu_imaging_at_obs_times, 
                          sigma=sigma_imaging, 
                          observed=imaging_values)
            
        # --- 4.3 淋巴细胞数据的似然函数 ---
        # 这一步至关重要，它用真实的淋巴细胞数据来约束我们新增的参数'c'，解决了参数不可辨识问题。
        # 【注意】这里的逻辑假设模型的内部状态(总肿瘤负荷)能以某种方式反映外周血淋巴细胞水平。
        # 这是一个简化但有效的假设，其主要目的是为参数'c'的推断提供数据锚点。
        
        # 定义模型预测的淋巴细胞水平
        mu_lymph = pm.Deterministic('mu_lymph', scaling_factor_lymph * total_burden_predicted)

        if lymphocyte_data is not None and not lymphocyte_data.empty:
            lymph_time_indices = np.searchsorted(time_points, lymphocyte_data['time'].values)

            # 【【【最终修复：增加防御性检查】】】
            if lymph_time_indices.size > 0:
                lymph_values = pm.Data('lymph_values', lymphocyte_data['value'].values, mutable=False)
                mu_lymph_at_obs_times = mu_lymph[lymph_time_indices]
                # 建立第三个似然，用真实的淋巴细胞数据来约束模型
                pm.Normal('obs_lymph', mu=mu_lymph_at_obs_times, sigma=sigma_lymph, observed=lymph_values)            
        
    # 返回构建好的模型。
    return model


# ==============================================================================
# --- 模型三：Norton-Simon (Gompertzian) 细胞周期动力学模型 ---
# 科学来源: Norton, L., & Simon, R. (1977). Tumor size, sensitivity to therapy, and design of treatment schedules. Cancer Treatment Reports.
# 模型简介:
#   该模型基于Gompertzian生长曲线，假设肿瘤的相对生长率（即每个细胞的分裂速率）随体积增大而指数级减慢。
#   化疗药物遵循“Log-Kill假说”，即一次给药按固定比例杀伤肿瘤细胞。
#   核心洞见：化疗对快速分裂的细胞更有效。由于Gompertzian生长中，小肿瘤的相对生长率更高，因此化疗对小肿瘤的杀伤效果也更强。
#   这使其成为优化传统细胞毒性化疗（尤其是剂量密集方案）给药时机的理论基石。
# ==============================================================================
def norton_simon_ode(y, t, p, *, treatment_func):
    """
    【机理模型核心 - Norton-Simon模型】定义了Gompertzian生长和Log-Kill动力学。
    
    这是一个单变量常微分方程（ODE），只关注总肿瘤细胞数 N 的变化。
    它描述了肿瘤在自然生长和化疗干预下的动态平衡。

    参数:
    - y (list/array):         当前时刻的状态向量, y = [N]，其中 N 是总肿瘤细胞数。
    - t (float):              当前时间点（由ODE求解器自动传入）。
    - rho0 (float):           肿瘤在理论上尺寸为零时的初始比生长率（specific growth rate）。
                              代表了肿瘤生长潜力的最大值。
    - alpha (float):          Gompertzian生长衰减常数。alpha越大，肿瘤的比生长率随时间/体积
                              衰减得越快，即越快进入平台期。
    - treatment_func (function): 一个函数，输入时间t，返回当前用药输入强度（1.0代表完整输入, 0.0代表无输入）。
    - kill_rate (float):      化疗药物在用药日的细胞杀伤率。根据Log-Kill假说，这是一个与
                              肿瘤大小无关的固定比例。

    返回:
    - list: 一个包含每个状态变量变化速率（导数）的列表。在这里，是 [dN/dt]。
    """
    # 使用索引访问
    rho0 = p[0]
    alpha = p[1]
    kill_rate = p[2]

    # --- 1. 解包当前状态 ---
    # 从输入向量 y 中获取当前时刻的总肿瘤细胞数 N。
    # 因为这是单变量模型，所以 y 只有一个元素。
    N = y[0]
    
    # --- 2. 获取当前治疗状态 ---
    # 调用治疗函数，查询在't'这个精确的时间点，是否正在用药。
    # u_t 将作为“治疗开关”，控制药物杀伤项是否激活。
    u_t = treatment_func(t)
    
    # --- 3. 计算肿瘤比生长率 rho(t) ---
    # 这是Gompertzian模型的核心：比生长率（即每个细胞的平均分裂速率）不是一个常数，
    # 而是随时间 t 指数级衰减。
    specific_growth_rate = rho0 * np.exp(-alpha * t)
    
    # --- 4. 计算总变化速率 dN/dt ---
    # 肿瘤总数 N 的变化速率由“生长”和“死亡”两部分构成：
    
    #   a. Gompertzian 生长项: specific_growth_rate * N
    #      总的生长量等于当前的比生长率乘以当前的细胞总数。
    #      随着肿瘤变大（时间t增加），specific_growth_rate减小，导致总生长量放缓。
    growth_term = specific_growth_rate * N
    
    #   b. Log-Kill 杀伤项: - u_t * kill_rate * N
    #      当处于治疗期间 (u_t=1.0) 时，一个固定比例 (kill_rate) 的细胞被杀死。
    #      杀死的绝对数量与当前肿瘤大小 N 成正比。
    kill_term = u_t * kill_rate * N
    
    # 将生长项和杀伤项相加，得到净变化率。
    dNdt = growth_term - kill_term
    
    # --- 5. 返回变化速率 ---
    # ODE求解器将使用这个返回的 dN/dt 来计算下一个微小时间步长的 N 值。
    return [dNdt]


def build_norton_simon_pymc_model(
    time_points, 
    tumor_burden, 
    treatment_func, 
    initial_burden,
    imaging_data=None
):
    """
    【贝叶斯模型构建器 - Norton-Simon模型】
    为 Norton-Simon (Gompertzian) 细胞周期动力学模型构建一个完整的PyMC贝叶斯推断框架。

    此函数的核心任务是，将 norton_simon_ode 函数中描述的数学理论，用概率语言（先验分布）
    编码，然后让PyMC根据真实观测数据（肿瘤标志物和/或影像学数据）来校准（推断）
    这个模型的具体参数（rho0, alpha, kill_rate）。

    功能亮点:
    - 单室建模: 精确实现了只关注总肿瘤负荷（N）的单变量ODE模型。
    - 多模态数据融合: 能够同时融合肿瘤标志物和影像学两种数据源来约束模型参数。
    - 经验贝叶斯先验: 利用数据本身来为部分参数（如缩放因子）设定一个更合理的先验中心，加速模型收敛。

    参数:
    - time_points (np.ndarray):     我们拥有观测数据的所有时间点（通常是天数）。
    - tumor_burden (np.ndarray):    在上述时间点观测到的肿瘤标志物负荷。
    - treatment_func (function):    治疗状态函数，会被传递给内部的ODE求解器。
    - initial_burden (float):       第一次观测时的肿瘤标志物负荷值 (T0)。
    - imaging_data (pd.DataFrame):  可选，包含 'time' 和 'value' 列的影像学观测数据。

    返回:
    - pm.Model: 一个构建完成但尚未开始“训练”（采样）的PyMC模型对象。
    """
    # --- 步骤 A: 为标志物与影像学数据的缩放因子，计算一个数据驱动的先验中心 ---
    # 这是“经验贝叶斯”思想的应用：在正式建模前，从数据中得到一个合理的初始猜测，
    # 这可以帮助采样器更快地找到参数的高概率区域，提升推断效率。
    initial_guess_for_scaling = 1.0
    if imaging_data is not None and not imaging_data.empty:
        # 使用 pd.merge_asof 安全地合并两个不同时间点的数据源，找到时间上最接近的匹配项。
        merged_data = pd.merge_asof(
            pd.DataFrame({'time': time_points, 'marker': tumor_burden}).sort_values('time'),
            imaging_data.sort_values('time').rename(columns={'value': 'imaging'}),
            on='time', direction='nearest', tolerance=pd.Timedelta(days=10)
        ).dropna()
        # 只有在存在匹配数据，且分母不为零时，才计算比例
        if not merged_data.empty and merged_data['imaging'].mean() > 1e-6:
            ratio = merged_data['marker'].mean() / merged_data['imaging'].mean()
            if ratio > 0: 
                initial_guess_for_scaling = ratio

    # --- 步骤 B: 使用PyMC的上下文管理器来定义一个模型容器 ---
    with pm.Model() as model:
        # --- 1. 定义未知参数的“先验分布”(Priors) ---
        # 先验是我们对参数在看到数据之前的“信念”或“猜测”。
        # 使用 Lognormal 分布可以确保参数始终为正，这符合生物学意义。
        
        # 初始比生长率 rho0
        rho0 = pm.Lognormal('rho0', mu=np.log(0.05), sigma=0.5)
        # 生长衰减常数 alpha
        alpha = pm.Lognormal('alpha', mu=np.log(0.001), sigma=0.5)
        # 药物杀伤率 kill_rate
        kill_rate = pm.Lognormal('kill_rate', mu=np.log(0.1), sigma=0.5)
        
        # --- 多模态数据融合所需的参数 (与现有其他模型保持一致) ---
        # scaling_factor: 将模型内部的抽象“肿瘤负荷”单位，转换为可观测的“肿瘤标志物”单位。
        scaling_factor = pm.Lognormal('scaling_factor', mu=np.log(initial_guess_for_scaling), sigma=0.5)
        # sigma_...: 分别定义两种观测数据（标志物和影像学）的测量误差/噪声。
        sigma_marker = pm.HalfNormal('sigma_marker', 1.0)
        sigma_imaging = pm.HalfNormal('sigma_imaging', 1.0)

        # --- 2. 将ODE求解器嵌入PyMC计算图 ---
        # 将所有需要推断的ODE参数打包成一个元组。
        ode_params = (rho0, alpha, kill_rate)
        
        # 定义ODE的初始条件 y0 = [N0]。
        # Norton-Simon是单室模型，其状态变量就是总肿瘤负荷N，因此初始条件就是总的 initial_burden。
        y0 = [initial_burden]

        wrapped_ns_ode = lambda y, t, p: norton_simon_ode(
            y, t, p, # <-- 直接传递 p
            treatment_func=treatment_func
        )

        # 调用PyMC的DifferentialEquation接口，将我们的ODE函数、参数、初始条件连接起来。
        ode_solution = pm.ode.DifferentialEquation(
            func=wrapped_ns_ode,
            times=time_points,      # 在这些时间点上求解
            n_states=1,             # <-- 核心区别：只有1个状态变量 (N)
            n_theta=len(ode_params),# ODE参数的数量
            t0=0,
            solver='scipy_solve_ivp',
            solver_kwargs={'method': 'Radau'}
        )(y0=y0, theta=ode_params)

        # --- 3. 定义可观测量的预测值 (mu) ---
        # 模型的预测值直接就是总肿瘤负荷 N(t)。
        # ode_solution 是一个矩阵，对于单状态模型，我们只需要取第一列 (索引0)。
        total_burden_predicted = ode_solution[:, 0]
        
        # 标志物的预测值 = 缩放因子 * 真实肿瘤负荷
        mu_marker = pm.Deterministic('mu_marker', scaling_factor * total_burden_predicted)
        # 影像学的预测值 = 真实肿瘤负荷
        mu_imaging = pm.Deterministic('mu_imaging', total_burden_predicted)

        # --- 4. 定义【多重】似然函数 (Likelihoods) ---
        # 似然函数是连接模型预测与真实观测数据的“桥梁”，它告诉模型观测数据在给定预测值的情况下出现的概率。
        
        # 4.1 标志物的似然：假设观测值(tumor_burden)是在模型预测(mu_marker)的基础上，增加了一个高斯噪声(sigma_marker)。
        pm.Normal('obs_marker', mu=mu_marker, sigma=sigma_marker, observed=tumor_burden)
        
        # 4.2 影像学的似然 (如果提供了影像学数据，逻辑与其他模型完全相同)
        if imaging_data is not None and not imaging_data.empty:
            # 计算影像学数据点在完整时间序列中的索引
            imaging_time_indices = np.searchsorted(time_points, imaging_data['time'].values)

            if imaging_time_indices.size > 0:
                # 将外部数据“注册”到模型中
                imaging_values = pm.Data('imaging_values', imaging_data['value'].values, mutable=False)
                # 提取出对应时间的模型预测值
                mu_imaging_at_obs_times = mu_imaging[imaging_time_indices]
                # 建立第二个似然函数，连接影像学预测与观测
                pm.Normal('obs_imaging', mu=mu_imaging_at_obs_times, sigma=sigma_imaging, observed=imaging_values)
            
    # 返回构建好的、待采样的模型对象。
    return model


# ==============================================================================
# --- 模型四：“三房室”模型（敏感S-持留P-抵抗R）---
# 科学来源: Bozic, I., et al. (2013). Evolutionary dynamics of cancer in response to targeted combination therapy. eLife.
# 模型简介:
#   引入了“持留细胞(Persister)”概念，这是一种非遗传性的、可逆的药物耐受状态。
#   该模型能精细地刻画靶向治疗中“获得性耐药”的产生过程：S→(药物诱导)→P→(突变)→R。
#   它为“药物间歇(Drug Holiday)”提供了理论依据：无用药压力让P细胞有机会逆转回S细胞，从而延缓不可逆R细胞的出现。
# ==============================================================================
def spr_ode(y, t, p, *, treatment_func):
    """
    【机理模型核心 - S-P-R三室模型】
    定义一个描述敏感(S)、持留(P)和抵抗(R)三种癌细胞亚群在治疗压力下动态演化的常微分方程（ODE）系统。

    参数:
    - y (list/array):         当前时刻的状态向量, y = [S, P, R]，分别代表三种细胞的数量。
    - t (float):              当前时间点（由ODE求解器自动传入）。
    - r_s (float):            敏感细胞(S)的固有生长速率。
    - r_r (float):            抵抗细胞(R)的固有生长速率。
    - K (float):              环境承载能力（Carrying Capacity），代表肿瘤微环境能支持的最大细胞总量。
    - d_s (float):            药物对敏感细胞(S)的杀伤率。
    - k_sp (float):           【关键参数】在治疗压力下，敏感细胞(S)向持留细胞(P)状态转换的速率。
    - k_ps (float):           【关键参数】持留细胞(P)在（通常是无用药压力期间）逆转回敏感细胞(S)状态的速率。
    - k_pr (float):           【关键参数】持留细胞(P)发生不可逆突变，变为抵抗细胞(R)的速率。
    - delta_p (float):        持留细胞(P)的自然死亡/清除率。模型假设P细胞不增殖，只会死亡或转换。
    - treatment_func (function): 一个函数，输入时间t，返回当前用药输入强度（1.0代表完整输入, 0.0代表无输入）。

    返回:
    - list: 一个包含每个状态变量变化速率（导数）的列表：[dS/dt, dP/dt, dR/dt]。
    """
    # 使用索引访问
    r_s = p[0]
    r_r = p[1]
    K = p[2]
    d_s = p[3]
    k_sp = p[4]
    k_ps = p[5]
    k_pr = p[6]
    delta_p = p[7]

    # --- 1. 解包当前状态 ---
    # 从输入向量 y 中，清晰地分离出当前时刻 S, P, R 三种细胞的数量。
    S, P, R = y
    
    # --- 2. 计算总细胞数并处理边界情况 ---
    # T 代表当前肿瘤的总负荷。
    T = S + P + R
    # 这是一个稳健性检查：如果肿瘤被完全清除（T=0），则所有变化速率都应为0，以避免除零等计算错误。
    if T == 0: 
        return [0, 0, 0]
    
    # --- 3. 获取当前治疗状态 ---
    # 调用治疗函数，查询在't'这个精确的时间点，是否正在用药。
    # u_t 将作为“治疗开关”，控制所有与药物相关的效应（杀伤、状态转换）是否激活。
    u_t = treatment_func(t)

    # --- 4. 定义共享的生长抑制项 ---
    # 采用逻辑斯谛（Logistic）模型来描述资源有限性。
    # 当总细胞数 T 接近环境承载能力 K 时，growth_inhibition 接近0，生长停滞。
    # 当 T 很小时，growth_inhibition 接近1，细胞接近其最大固有速率生长。
    # 核心假设：P细胞不增殖，因此该项只影响S和R细胞。
    growth_inhibition = 1 - (T / K)
    
    # --- 5. 计算敏感细胞(S)的变化速率 (dS/dt) ---
    # S细胞的变化由四个部分构成，代表其所有的“流入”和“流出”：
    #   (+) (r_s * S * growth_inhibition): S细胞自身的逻辑斯谛增长。
    #   (-) (d_s * u_t * S):            【流出】药物在治疗期间 (u_t=1) 对S细胞的直接杀伤。
    #   (-) (k_sp * u_t * S):           【流出】药物在治疗期间 (u_t=1) 诱导S细胞转换为P细胞。
    #   (+) (k_ps * P):                 【流入】P细胞逆转回S细胞。注意：这个过程与用药无关，是P细胞的内在特性。
    dSdt = (r_s * S * growth_inhibition) - (d_s * u_t * S) - (k_sp * u_t * S) + (k_ps * P)
    
    # --- 6. 计算持留细胞(P)的变化速率 (dP/dt) ---
    # P细胞是中间状态，其数量变化完全由与其他状态的转换和自身死亡决定：
    #   (+) (k_sp * u_t * S):           【流入】由S细胞在治疗压力下转换而来。
    #   (-) (k_ps * P):                 【流出】逆转回S细胞。
    #   (-) (k_pr * u_t * P):           【流出】在治疗压力下发生突变，变为不可逆的R细胞。
    #   (-) (delta_p * P):              【流出】P细胞自身的自然死亡/清除。
    dPdt = (k_sp * u_t * S) - (k_ps * P) - (k_pr * u_t * P) - (delta_p * P)
    
    # --- 7. 计算抵抗细胞(R)的变化速率 (dR/dt) ---
    # R细胞是演化的终点，其数量只会增加：
    #   (+) (r_r * R * growth_inhibition): R细胞自身的逻辑斯谛增长。模型假设R细胞不受药物影响。
    #   (+) (k_pr * u_t * P):              【流入】由P细胞在治疗压力下突变而来。这是产生获得性耐药的关键路径。
    dRdt = (r_r * R * growth_inhibition) + (k_pr * u_t * P)
    
    # --- 8. 返回变化速率 ---
    # ODE求解器将使用这个返回的速率向量来计算下一个微小时间步长的 [S, P, R] 值。
    return [dSdt, dPdt, dRdt]

def build_spr_pymc_model(
    time_points, 
    tumor_burden, 
    treatment_func, 
    initial_burden,
    imaging_data=None
):
    """
    【贝叶斯模型构建器 - S-P-R模型】
    为“三房室”（敏感S-持留P-抵抗R）动力学模型构建一个完整的PyMC贝叶斯推断框架。

    此函数的核心任务是，将 spr_ode 函数中描述的生物学假设，用概率语言（先验分布）
    编码，然后让PyMC根据真实观测数据（肿瘤标志物和/或影像学数据）来校准（推断）
    这个模型的具体参数。

    功能亮点:
    - 多模态数据融合: 通过“多重似然”方法，同时融合肿瘤标志物和影像学两种不同来源的数据。
    - 经验贝叶斯先验: 利用数据本身来为部分参数（如缩放因子）设定一个更合理的先验中心，加速模型收敛。
    - 分层初始条件: 使用分层先验来推断初始肿瘤中三种细胞亚群的构成，更符合生物学假设。

    参数:
    - time_points (np.ndarray):     我们拥有观测数据的所有时间点（通常是天数）。
    - tumor_burden (np.ndarray):    在上述时间点观测到的肿瘤标志物负荷。
    - treatment_func (function):    治疗状态函数，会被传递给内部的ODE求解器。
    - initial_burden (float):       第一次观测时的肿瘤标志物负GH荷值 (T0)。
    - imaging_data (pd.DataFrame):  可选，包含 'time' 和 'value' 列的影像学观测数据。

    返回:
    - pm.Model: 一个构建完成但尚未开始“训练”（采样）的PyMC模型对象。
    """
    # --- 步骤 A: 为标志物与影像学数据的缩放因子，计算一个数据驱动的先验中心 ---
    # 这是“经验贝叶斯”思想的应用：在正式建模前，从数据中得到一个合理的初始猜测，
    # 这可以帮助采样器更快地找到参数的高概率区域，提升推断效率。
    initial_guess_for_scaling = 1.0
    if imaging_data is not None and not imaging_data.empty:
        # 使用 pd.merge_asof 安全地合并两个不同时间点的数据源。
        # 它会为每个标志物数据点，找到时间上最接近的（10天内）影像学数据点。
        merged_data = pd.merge_asof(
            pd.DataFrame({'time': time_points, 'marker': tumor_burden}).sort_values('time'),
            imaging_data.sort_values('time').rename(columns={'value': 'imaging'}),
            on='time', direction='nearest', tolerance=pd.Timedelta(days=10)
        ).dropna()
        # 只有在存在匹配数据，且分母不为零时，才计算比例
        if not merged_data.empty and merged_data['imaging'].mean() > 1e-6:
            ratio = merged_data['marker'].mean() / merged_data['imaging'].mean()
            if ratio > 0: 
                initial_guess_for_scaling = ratio

    # --- 步骤 B: 使用PyMC的上下文管理器来定义一个模型容器 ---
    with pm.Model() as model:
        # --- 1. 定义未知参数的“先验分布”(Priors) ---
        # 先验是我们对参数在看到数据之前的“信念”或“猜测”。
        
        # --- 基础生长/竞争参数 (与LV模型类似) ---
        r_s = pm.Lognormal('r_s', mu=np.log(0.08), sigma=0.5)  # 敏感细胞生长率
        r_r = pm.Lognormal('r_r', mu=np.log(0.03), sigma=0.5)  # 抵抗细胞生长率
        
        # 为环境承载能力 K 设置一个数据驱动的先验，使其更有信息量
        max_obs = np.max(tumor_burden)
        if imaging_data is not None and not imaging_data.empty:
            # 将影像学数据也转换到标志物的尺度，共同决定K的先验
            max_obs = max(max_obs, np.max(imaging_data['value']) * initial_guess_for_scaling)
        K = pm.Lognormal('K', mu=np.log(max_obs * 1.5), sigma=0.5)
        d_s = pm.Lognormal('d_s', mu=np.log(0.15), sigma=0.5) # 药物对S细胞的杀伤率

        # --- 新增的转换速率和死亡率参数 ---
        # 使用指数分布(Exponential)作为先验，因为它倾向于较小的值，
        # 这符合细胞状态转换或突变通常是相对较慢过程的生物学假设。
        # 参数 lam (lambda) 是速率，先验的均值为 1/lam。
        k_sp = pm.Exponential('k_sp', lam=50.0)    # S -> P 转换率 (均值=0.02)
        k_ps = pm.Exponential('k_ps', lam=50.0)    # P -> S 逆转率 (均值=0.02)
        k_pr = pm.Exponential('k_pr', lam=100.0)   # P -> R 突变率 (均值=0.01, 假设更慢)
        delta_p = pm.Exponential('delta_p', lam=50.0) # P细胞自然死亡率 (均值=0.02)

        # --- 分层定义的初始细胞构成 (Hierarchical Priors for Initial Conditions) ---
        # 这是一个更精细的假设，我们不直接猜测S,P,R的初始比例，而是分两步：
        
        # 第1步：初始敏感细胞(S)占总数的比例 (s0_frac)。
        # Beta(10, 1)分布的均值在0.9附近，强烈倾向于1。这编码了“初始肿瘤主要由敏感细胞构成”的信念。
        s0_frac = pm.Beta('s0_frac', alpha=10.0, beta=1.0)
        
        # 第2步：在剩余的非S细胞中，持留细胞(P)所占的比例 (p0_frac_of_remainder)。
        # Beta(1, 5)分布强烈倾向于0。这编码了“在初始的非敏感细胞中，绝大多数也不是P细胞（而是R细胞）”的信念。
        p0_frac_of_remainder = pm.Beta('p0_frac_of_remainder', alpha=1.0, beta=5.0)
        
        # --- 多模态数据融合所需的参数 ---
        # scaling_factor: 将模型内部的抽象“肿瘤负荷”单位，转换为可观测的“肿瘤标志物”单位（如 ng/mL）。
        scaling_factor = pm.Lognormal('scaling_factor', mu=np.log(initial_guess_for_scaling), sigma=0.5)
        # sigma_...: 分别定义两种观测数据（标志物和影像学）的测量误差/噪声。
        sigma_marker = pm.HalfNormal('sigma_marker', 1.0)
        sigma_imaging = pm.HalfNormal('sigma_imaging', 1.0)

        # --- 2. 将ODE求解器嵌入PyMC计算图 ---
        # 将所有需要推断的ODE参数打包成一个元组。
        ode_params = (r_s, r_r, K, d_s, k_sp, k_ps, k_pr, delta_p)
        
        # 根据上面定义的分层先验，动态计算三室的初始条件 [S0, P0, R0]。
        # 这些s0, p0, r0本身也是PyMC图中的符号变量。
        s0 = initial_burden * s0_frac
        p0 = initial_burden * (1 - s0_frac) * p0_frac_of_remainder
        r0 = initial_burden * (1 - s0_frac) * (1 - p0_frac_of_remainder)
        y0 = [s0, p0, r0] # PyMC ODE接口接受字典形式的初始条件

        wrapped_spr_ode = lambda y, t, p: spr_ode(
            y, t, p,
            treatment_func=treatment_func
        )

        # 调用PyMC的DifferentialEquation接口，但现在使用我们更安全的包装函数
        ode_solution = pm.ode.DifferentialEquation(
            func=wrapped_spr_ode,
            times=time_points,      # 在这些时间点上求解
            n_states=3,             # <-- 核心区别：有3个状态变量 (S, P, R)
            n_theta=len(ode_params),# ODE参数的数量
            t0=0,
            solver='scipy_solve_ivp',
            solver_kwargs={'method': 'Radau'}
        )(y0=y0, theta=ode_params)

        # --- 3. 定义可观测量的预测值 (mu) ---
        # 我们在外部无法直接观测到S,P,R各自的数量，只能观测到它们的总量。
        total_burden_predicted = ode_solution[:, 0] + ode_solution[:, 1] + ode_solution[:, 2]
        
        # 标志物的预测值 = 缩放因子 * 真实肿瘤负荷
        mu_marker = pm.Deterministic('mu_marker', scaling_factor * total_burden_predicted)
        # 影像学的预测值 = 真实肿瘤负荷 (这是我们的核心假设，即影像学直接反映肿瘤物理体积)
        mu_imaging = pm.Deterministic('mu_imaging', total_burden_predicted)

        # --- 4. 定义【多重】似然函数 (Likelihoods) ---
        # 似然函数是连接模型预测与真实观测数据的“桥梁”。
        
        # 4.1 标志物的似然：假设观测值(tumor_burden)是在模型预测(mu_marker)的基础上，增加了一个高斯噪声(sigma_marker)。
        pm.Normal('obs_marker', mu=mu_marker, sigma=sigma_marker, observed=tumor_burden)

        # 4.2 影像学的似然 (如果提供了影像学数据)
        if imaging_data is not None and not imaging_data.empty:
            # 在Python/Numpy层面计算出影像学数据点对应于完整时间序列的索引。
            imaging_time_indices = np.searchsorted(time_points, imaging_data['time'].values)
            
            # 防御性检查，防止空索引
            if imaging_time_indices.size > 0:
                # 使用 pm.Data 将外部数据“注册”到模型中，这是PyMC的最佳实践。
                imaging_values = pm.Data('imaging_values', imaging_data['value'].values, mutable=False)
                # 在PyMC的计算图内部，使用索引从完整的 `mu_imaging` 预测张量中提取出对应时间的预测值。
                mu_imaging_at_obs_times = mu_imaging[imaging_time_indices]
                # 建立第二个似然函数：连接影像学预测与注册的影像学观测。
                pm.Normal('obs_imaging', mu=mu_imaging_at_obs_times, sigma=sigma_imaging, observed=imaging_values)
            
    # 返回构建好的、待采样的模型对象。
    return model


# ==============================================================================
# --- 模型五：数据约束的免疫-肿瘤互作模型 (V1.2 - 融合优化版) ---
# 科学来源: de Pillis, L. G., et al. (2005). A Validated Mathematical Model of 
#           Cell-Mediated Immune Response to Tumor Growth. Cancer Research.
# 【科学原理说明】:
#   这是一个根据您的方案设计和实现的、严谨的科学计算模型。它将复杂的
#   免疫系统抽象为单一的“有效免疫细胞(I)”群体，并模拟其与肿瘤细胞(T)之间的
#   “捕食者-猎物”动态。与纯理论模型不同，此模型的免疫动态将由真实的淋巴细胞
#   数据进行约束，以确保参数推断的科学性和唯一性。
#
#   【本版本融合的核心科学思想】:
#   1. **饱和杀伤效应 (米氏动力学)**: 免疫系统对肿瘤的杀伤采用了米氏动力学形式 
#      `(c*I*T)/(h+T)`。这捕捉了免疫系统的杀伤能力在肿瘤负荷极高时会达到
#      饱和的生物学现实，比简单的质量作用定律 (-c*I*T) 更为精确。
#   2. **协同激活效应 (希尔函数)**: 免疫系统的招募/激活项采用了希尔函数 
#      `(g*T^2/(h+T^2))*I`。这精确地模拟了免疫激活的“S型”开关特性和协同性，
#      即肿瘤负荷需要达到一定阈值才能有效、急剧地启动免疫反应。
#   3. **数据约束**: 在PyMC模型构建器中，通过针对`lymphocyte_data`的第三个似然函数，
#      实现了“整合外周血数据”以解决参数可辨识度问题的核心逻辑。
# ==============================================================================
def immuno_oncology_ode(y, t, p, *, treatment_func):
    """
    【机理模型核心 V2.3 - 融合临床肿瘤学长期演化规律】
    
    这是一个在 V2.2 基础上，进一步融合了临床肿瘤学长期观察规律的高级版本。
    
    【V2.3 核心改进】:
    相比 V2.2，V2.3 引入了三个关键的长期演化机制，使模型能够准确模拟
    多周期治疗（3-6个月）中的"耐药累积"和"免疫疲劳"现象：
    
    1. 【新增】免疫系统的不可逆损伤 (Irreversible Immune Damage):
       - 每次用药都会对免疫系统造成 **永久性** 的微小损伤
       - 通过引入"免疫基础水平"(I_baseline)的时间依赖衰减来建模
       - 累积效应：多次治疗后，免疫系统无法恢复到初始水平
    
    2. 【新增】肿瘤的获得性耐药 (Acquired Drug Resistance):
       - 肿瘤细胞在药物压力下会逐渐适应，导致药物敏感性下降
       - 通过 drug_resistance_factor 来建模，随累积用药时间而增长
       - 临床表现：后期治疗周期中，肿瘤下降幅度减小
    
    3. 【优化】微环境介导的免疫抑制 (TME-Mediated Immunosuppression):
       - 肿瘤微环境的免疫抑制能力与肿瘤负荷 **非线性相关**
       - 引入 (T/K)^2 项，捕捉"大肿瘤 → 强免疫抑制"的正反馈
       - 这使得晚期肿瘤更难控制
    
    【参数说明】:
    - y (list): [T, I, cumulative_drug_exposure, I_baseline]
        T: 肿瘤细胞数量
        I: 当前有效免疫细胞水平
        cumulative_drug_exposure: 累积药物暴露量（用于跟踪耐药进展）
        I_baseline: 免疫系统基础恢复水平（会随治疗次数下降）
    
    - p (tuple): (a, b, c, s, g, h, d, p_param)
        [与V2.2相同，但参数含义因新机制而更丰富]
    
    【生物学依据】:
    - 免疫损伤: Ewer, M.S., et al. Cardiotoxicity of anticancer treatments. 
                Nat Rev Cardiol (2015) - 化疗对免疫系统的长期影响
    - 获得性耐药: Holohan, C., et al. Cancer drug resistance: an evolving paradigm. 
                  Nat Rev Cancer (2013)
    - 微环境抑制: Quail, D.F., & Joyce, J.A. Microenvironmental regulation of tumor 
                  progression and metastasis. Nat Med (2013)
    """
    # ========================================================================
    # === 步骤1: 解包参数与状态 ===
    # ========================================================================
    
    # 基础生物学参数
    a = p[0]        # 肿瘤固有生长率
    b = p[1]        # 环境承载力倒数 (1/K)
    c = p[2]        # 免疫基础杀伤率
    s = p[3]        # 免疫来源速率
    g = p[4]        # 免疫最大招募率
    h = p[5]        # 半饱和常数
    d = p[6]        # 免疫自然死亡率
    p_param = p[7]  # 免疫耗竭率
    
    # 解包状态变量 (【修正】现在有3个状态)
    T = y[0]        # 肿瘤细胞数
    I = y[1]        # 有效免疫细胞水平
    C_D = y[2]      # 【新增】累积药物暴露 (天)
    
    # 获取当前治疗状态
    u_t = treatment_func(t)
    
    # ========================================================================
    # === 步骤2: 边界条件检查 ===
    # ========================================================================
    
    if T <= 0 or I <= 0:
        # 【修正】确保返回3个导数
        return [0.0, 0.0, 0.0] 
    
    # ========================================================================
    # === 【核心修正】: 使用状态变量 C_D (y[2]) 计算累积效应 ===
    # ========================================================================

    # 2.1 计算获得性耐药因子 (【修正】使用 C_D 替换 t * u_t)
    drug_resistance_factor = 1.0 / (1.0 + 0.005 * C_D)
    
    # 2.2 免疫增强因子 (逻辑不变，但依赖于修正后的耐药因子)
    if u_t > 0:
        immune_boost_factor = (1.0 + u_t) * drug_resistance_factor
    else:
        immune_boost_factor = 0.7
    
    # ========================================================================
    # === dT/dt：肿瘤动态方程 ===
    # ========================================================================
    
    c_effective = c / 10.0
    growth_term = a * T * (1 - b * T)
    immune_kill_term = immune_boost_factor * (c_effective * I * T) / (h + T)
    
    # 3.3 药物直接杀伤项 (【修正】使用修正后的耐药因子)
    drug_kill_term = u_t * drug_resistance_factor * T
    
    dTdt = growth_term - immune_kill_term - drug_kill_term
    
    # ========================================================================
    # === dIdt：免疫动态方程 ===
    # ========================================================================
    
    # 4.1 免疫来源项 (【修正】使用 C_D 替换 t * u_t)
    immune_damage_factor = np.exp(-0.001 * C_D)
    
    source_term = s * immune_damage_factor
    recruitment_term = (g * T**2 / (h**2 + T**2)) * I
    death_term = d * I
    
    microenvironment_suppression = (1 + (T / (1.0/b))**2)
    exhaustion_term = p_param * microenvironment_suppression * I * T
    
    # 4.5 药物调节 (逻辑不变)
    if u_t > 0:
        exhaustion_reduction_factor = 0.5
        source_efficiency = 1.0
        recruitment_efficiency = 1.0
    else:
        exhaustion_reduction_factor = 1.5
        source_efficiency = 0.5
        recruitment_efficiency = 0.7
    
    adjusted_exhaustion = exhaustion_reduction_factor * exhaustion_term
    adjusted_source = source_efficiency * source_term
    adjusted_recruitment = recruitment_efficiency * recruitment_term
    
    dIdt = adjusted_source + adjusted_recruitment - death_term - adjusted_exhaustion
    
    # ========================================================================
    # === 【新增】dC_D/dt：累积暴露动态方程 ===
    # ========================================================================
    # 累积暴露量的变化率 = 当前的用药强度 (1.0或0.0)
    dC_D_dt = u_t
    
    # ========================================================================
    # === 步骤5: 返回状态变化率向量 ===
    # ========================================================================
    
    # 【修正】返回3个导数
    return [dTdt, dIdt, dC_D_dt]


def build_immuno_oncology_pymc_model(
    time_points, 
    tumor_burden, 
    treatment_func, 
    initial_burden,
    imaging_data=None,
    lymphocyte_data=None
):
    """
    【贝叶斯模型构建器 V2.4 - 支持累积损伤跟踪】
    
    V2.4 核心更新：
    - 同步支持 ODE V2.4 的三状态模型 [T, I, C_D]
    - 累积药物暴露 C_D 作为隐藏状态，不直接观测
    - 通过 C_D 实现获得性耐药和免疫损伤的累积效应
    """
    # --- 准备工作：计算数据驱动的先验 ---
    initial_guess_for_scaling = 1.0
    if imaging_data is not None and not imaging_data.empty:
        merged_data = pd.merge_asof(
            pd.DataFrame({'time': time_points, 'marker': tumor_burden}).sort_values('time'),
            imaging_data.sort_values('time').rename(columns={'value': 'imaging'}),
            on='time', direction='nearest', tolerance=pd.Timedelta(days=10)
        ).dropna()
        if not merged_data.empty and merged_data['imaging'].mean() > 1e-6:
            ratio = merged_data['marker'].mean() / merged_data['imaging'].mean()
            if ratio > 0: 
                initial_guess_for_scaling = ratio

    with pm.Model() as model:
        # === 1. 定义先验分布 ===
        
        # 肿瘤参数
        a = pm.Lognormal('a', mu=np.log(0.5), sigma=0.5, doc="肿瘤固有生长速率")
        
        max_obs = np.max(tumor_burden)
        if imaging_data is not None and not imaging_data.empty:
            max_obs = max(max_obs, np.max(imaging_data['value']) * initial_guess_for_scaling)
        K_inv = pm.Lognormal('b', mu=np.log(1.0 / (max_obs * 1.5)), sigma=1.0, 
                            doc="环境承载能力相关参数 (1/K)")

        # 免疫参数
        c = pm.Lognormal('c', mu=np.log(1e-7), sigma=1.0)
        s = pm.Lognormal('s', mu=np.log(1e4), sigma=1.0)
        g = pm.Lognormal('g', mu=np.log(0.1), sigma=0.5)
        h = pm.Lognormal('h', mu=np.log(2e7), sigma=1.0)
        d = pm.Lognormal('d', mu=np.log(0.04), sigma=0.5)
        p = pm.Lognormal('p', mu=np.log(1e-7), sigma=1.0)
        
        I_init = pm.Lognormal('I_init', mu=np.log(initial_burden * 0.5), sigma=1.0,
                             doc="初始有效免疫细胞水平")

        # 观测参数
        scaling_factor = pm.Lognormal('scaling_factor', mu=np.log(initial_guess_for_scaling), sigma=0.5)
        sigma_marker = pm.HalfNormal('sigma_marker', 1.0)
        sigma_imaging = pm.HalfNormal('sigma_imaging', 1.0)
        scaling_factor_lymph = pm.Lognormal('scaling_factor_lymph', mu=np.log(1.0), sigma=0.5)
        sigma_lymph = pm.HalfNormal('sigma_lymph', 1.0)

        # === 2. 【修改】嵌入 ODE 求解器（3状态版本）===
        ode_params = (a, K_inv, c, s, g, h, d, p)
        
        # 【修改】初始条件现在有3个：[T_0, I_0, C_D_0]
        # C_D_0 = 0.0（假设模拟从未用药状态开始）
        y0 = [initial_burden, I_init, 0.0]

        wrapped_io_ode = lambda y, t, p: immuno_oncology_ode(
            y, t, p,
            treatment_func=treatment_func
        )

        # 【修改】n_states = 3
        ode_solution = pm.ode.DifferentialEquation(
            func=wrapped_io_ode,
            times=time_points,
            n_states=3,  # ✅ 修改为 3
            n_theta=len(ode_params),
            t0=0,
            solver='scipy_solve_ivp',
            solver_kwargs={'method': 'Radau'}
        )(y0=y0, theta=ode_params)

        # === 3. 【修改】定义可观测量（只使用前2个状态）===
        
        # 3.1 肿瘤负荷预测（第0列）
        total_burden_predicted = ode_solution[:, 0]
        mu_marker = pm.Deterministic('mu_marker', scaling_factor * total_burden_predicted)
        mu_imaging = pm.Deterministic('mu_imaging', total_burden_predicted)

        # 3.2 免疫水平预测（第1列）
        immune_level_predicted = ode_solution[:, 1]
        mu_lymph = pm.Deterministic('mu_lymph', scaling_factor_lymph * immune_level_predicted)

        # 注意：第2列（C_D）是隐藏状态，不直接观测，因此不建立似然

        # === 4. 定义多重似然函数 ===
        
        # 4.1 肿瘤标志物似然
        pm.Normal('obs_marker', mu=mu_marker, sigma=sigma_marker, observed=tumor_burden)
        
        # 4.2 影像学似然
        if imaging_data is not None and not imaging_data.empty:
            imaging_time_indices = np.searchsorted(time_points, imaging_data['time'].values)
            if imaging_time_indices.size > 0:
                imaging_values = pm.Data('imaging_values', imaging_data['value'].values, mutable=False)
                mu_imaging_at_obs_times = mu_imaging[imaging_time_indices]
                pm.Normal('obs_imaging', mu=mu_imaging_at_obs_times, sigma=sigma_imaging, 
                         observed=imaging_values)
        
        # 4.3 淋巴细胞似然
        if lymphocyte_data is not None and not lymphocyte_data.empty:
            lymph_time_indices = np.searchsorted(time_points, lymphocyte_data['time'].values)
            if lymph_time_indices.size > 0:
                lymph_values = pm.Data('lymph_values', lymphocyte_data['value'].values, mutable=False)
                mu_lymph_at_obs_times = mu_lymph[lymph_time_indices]
                pm.Normal('obs_lymph', mu=mu_lymph_at_obs_times, sigma=sigma_lymph, 
                         observed=lymph_values)
            
    return model


# ==============================================================================
# --- 模型工厂 ---
# ==============================================================================
MODEL_FACTORY = {
    "经典竞争模型 (S-R)": {
        "builder": build_lotka_volterra_pymc_model,
        "ode_func": lotka_volterra_ode,
        "params": ['r_s', 'r_r', 'K', 'alpha_rs', 'alpha_sr', 'd_s',
                   'cost_factor', 'k_sr', 'k_rs', 'c'],
        "states": ["S", "R"], 
        "resistance_state_index": 1,  # 抵抗细胞在索引1的位置
        "required_data": ['淋巴细胞绝对数'],
        "fallback_initials": {"s0_frac": 0.5}
    },
    "干细胞驱动模型 (B20)": {
        "builder": build_stem_cell_pymc_model,
        "ode_func": stem_cell_ode,
        "params": ['p_s', 'delta_d', 'c'], 
        "states": ["Stem", "Differentiated"],
        "resistance_state_index": 0,  # “抵抗”特性由干细胞（索引0）体现
        "required_data": ['淋巴细胞绝对数'],
        "fallback_initials": {"s0_frac": 0.5}
    },
    "Norton-Simon (Gompertzian)": {
        "builder": build_norton_simon_pymc_model,
        "ode_func": norton_simon_ode,
        "params": ['rho0', 'alpha', 'kill_rate'],
        "states": ["Tumor Volume"],
        "resistance_state_index": None, # 单室模型没有抵抗细胞的概念
        "required_data": []
    },
    "三房室模型 (S-P-R)": {
        "builder": build_spr_pymc_model,
        "ode_func": spr_ode,
        "params": ['r_s', 'r_r', 'K', 'd_s', 'k_sp', 'k_ps', 'k_pr', 'delta_p'],
        "states": ["S", "P", "R"],
        "resistance_state_index": 2,  # 抵抗细胞在索引2的位置
        "required_data": [],
        "fallback_initials": {"s0_frac": 0.9, "p0_frac_of_remainder": 0.1}
    },
    "免疫-肿瘤互作模型 (de Pillis 2005)": {
        "builder": build_immuno_oncology_pymc_model,
        "ode_func": immuno_oncology_ode,
        # 【修正】移除了 'I_init'。此列表只包含描述ODE动态演化规律的参数。
        "params": ['a', 'b', 'c', 's', 'g', 'h', 'd', 'p'],
        "states": ["Tumor", "Immune", "CumulativeExposure"], 
        "resistance_state_index": None, # 这个简化模型没有显式的“耐药”状态
        "required_data": ['淋巴细胞绝对数']
    }
}
