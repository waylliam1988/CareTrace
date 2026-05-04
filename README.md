# CareTrace 关照轨迹

> A local-first, bilingual lab-trend companion for patients, families, clinicians, and researchers.  
> 一个本地优先、中英双语的化验数据趋势观察工具，写给患者、家属、医生和研究者。

<p align="center">
  <a href="#zh"><strong>中文</strong></a>
  &nbsp;|&nbsp;
  <a href="#en"><strong>English</strong></a>
</p>

**Author / 作者**: Liu Yanwei  
**GitHub**: [waylliam1988/CareTrace](https://github.com/waylliam1988/CareTrace)  
**License / 开源许可证**: Apache License 2.0, with attribution preserved through `NOTICE`.

<a id="zh"></a>

## 中文

> [Switch to English](#en)

### 这个系统为什么存在

CareTrace 最初是为了我的妈妈写的。她患有乳腺癌，曾扩散到腋窝淋巴结，已经做过手术。治疗期间，我们尝试理解适应性治疗、间歇性低剂量治疗、PD-1、抗血管生成药物、肿瘤标志物、血常规、炎症和营养状态之间的关系。

那段时间很黑暗。医院通常必须按照指南和标准路径治疗，这是医学安全的基础；但在医院之外，患者和家属仍然需要做很多事：整理报告、看懂趋势、准备复诊问题、记录身体反应、避免被单次异常吓倒，也避免忽略真正值得复核的变化。CareTrace 就是从这个需求里生长出来的。

它不是为了替代医生，也不是为了鼓励患者自己改药。它想做的是：把零散、宝贵、很少的化验数据整理成更清楚的“观察材料”，让患者在治疗中多一点掌控感，多一点理解，多一点可以带去和医生讨论的问题。

### CareTrace 能帮患者和家属做什么

- **把化验单整理到本地电脑里**：血常规、生化、CRP、肿瘤标志物、甲功等可以按日期保存。
- **减少手动输入**：支持上传报告单图片，用本地 RapidOCR/OpenCV 识别后再人工核对保存。
- **看个人趋势，而不只看一次结果**：系统会比较当前值、医院参考范围和个人历史基线。
- **珍惜稀少数据**：很多肿瘤患者一年只有 3-4 次血常规和肿瘤标志物检测。CareTrace 的小样本模型尽量不丢弃历史点，不用移动平均线把前几个点“吃掉”。
- **给出温和的观察提示**：输出尽量使用“观察、提示、请咨询医生”，避免把统计变化说成病情结论。
- **辅助准备复诊沟通**：例如哪些指标最近偏离个人基线、哪些变化可能只是波动、哪些值得复查。
- **适应性治疗数学模拟**：用机制模型演示不同治疗输入假设下的理论轨迹，帮助提出问题，而不是给出用药方案。
- **保护隐私**：所有病人数据默认保存在你自己的电脑上，不上传到任何服务器。

### 系统边界

CareTrace 不能、也不应该做这些事：

- 不诊断癌症复发、进展、转移或缓解。
- 不判断某个治疗是否有效。
- 不决定 PD-1、抗血管生成药、化疗、内分泌治疗或靶向药剂量。
- 不替代影像、病理、医生查体和正规随访。
- 不用肿瘤标志物单独判断病情。
- 不保证预测准确。

它做的是统计观察。任何治疗、停药、加药、延迟用药、复查间隔、影像检查，都应咨询有资质的肿瘤科医生。

### 适应性治疗是什么，为什么可能延缓耐药

传统治疗常常追求“最大耐受剂量”和尽可能强的杀伤压力。适应性治疗的思想不同：肿瘤内部可能同时存在对药物敏感和相对耐药的细胞。持续高压治疗可能快速清除敏感细胞，让耐药细胞失去竞争对手，反而更快占据优势。

适应性治疗希望在某些场景下保留一部分药物敏感细胞，让它们继续和耐药细胞竞争资源，从而延缓耐药群体扩张。这个思想来自肿瘤演化、生物生态学和数学肿瘤学。

但必须强调：适应性治疗不是“随意减药”，也不是通用方案。不同癌种、药物、分期、肿瘤负荷、免疫状态和影像表现都可能完全不同。CareTrace 的模拟页只展示理论轨迹，帮助患者把问题整理出来，例如：

- 这个指标变化是否足以支持继续观察？
- 是否需要更近一次复查？
- 是否需要把肿瘤标志物和影像、症状一起看？
- 是否存在治疗后短期标志物“假升高”的可能？
- 我的血常规、白蛋白、CRP 是否提示身体状态需要复核？

### 肿瘤标志物、血常规和自我记录能提供什么帮助

肿瘤标志物如 CEA、CA15-3、CA125、CA19-9、NSE、SCC 等，有时能反映肿瘤负荷或治疗反应，但它们也会被炎症、肝胆问题、检测平台、良性疾病和治疗早期波动影响。单次升高不能直接等于进展。

血常规和基础生化不是“癌症雷达”，但它们能提供身体状态的侧面信息：

- 中性粒细胞、淋巴细胞、血小板可组合成 NLR、PLR、SII、SIRI 等炎症/免疫代理指标。
- 白蛋白、淋巴细胞可组成 PNI，作为营养和免疫储备的代理。
- CRP、白蛋白可帮助观察炎症负荷。
- 血红蛋白、白细胞、血小板可提示治疗耐受性、感染风险或骨髓抑制风险。

患者自诉也很重要：疼痛、乏力、食欲、体重、发热、咳嗽、腹泻、皮疹、出血倾向等，都可能帮助医生解释化验波动。CareTrace 当前重点是化验数据，未来适合加入更温和的症状日记，而不是要求患者自己填写“进展/疑似进展”这类沉重标签。

### 如何使用

1. 安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

2. 启动：

```powershell
streamlit run main.py
```

3. 在左侧创建病人档案。
4. 在“录入数据”中选择化验单类型，手动输入或上传报告图片识别。
5. 保存前核对日期、指标、数值和单位。
6. 回到“健康报告”查看趋势、个人基线偏离和观察提示。
7. 数据变化后点击“重新训练模型”。
8. 在“适应性模拟”中只把模拟结果作为和医生讨论的问题清单。

### 作者、仓库和开源许可证

- 作者：Liu Yanwei / 刘彦巍。
- GitHub：<https://github.com/waylliam1988/CareTrace>。
- 许可证：Apache License 2.0。
- 署名：本项目包含 `NOTICE` 文件。根据 Apache-2.0 的 NOTICE 机制，使用、复制、修改、分发或基于 CareTrace 构建衍生项目时，应保留 `LICENSE`、`NOTICE` 和原作者署名。
- 医学边界：CareTrace 不是医疗诊断或治疗系统，不提供用药、剂量、停药、急救或临床决策建议。

### 给医生、医学研究者和深度学习研究者

CareTrace 是一个本地运行的 Streamlit 原型系统，面向的是真实世界中非常常见、但在建模上很棘手的场景：单个肿瘤患者每年只有少量复查点，数据类型主要是血常规、生化、炎症指标和肿瘤标志物，缺乏高频生命体征、连续影像标注和标准化结局标签。系统目标不是端到端诊断，而是建立一个可解释、可审计、可回测的 longitudinal laboratory observation framework。

#### 研究问题

CareTrace 试图回答的不是“病人是否进展”，而是更低一层、更可验证的问题：

- 在个人历史基线之上，当前化验值是否出现了统计上值得复核的偏离？
- 对于一年 3-4 次检测的稀疏纵向数据，如何尽量不丢弃任何历史点？
- 当检测间隔不规则、实验室平台可能变化、肿瘤标志物存在生物学和方法学变异时，如何表达不确定性？
- 多个弱信号指标是否能共同形成一个“数据活动度代理因子”，用于辅助观察，而不直接映射为病情结论？
- 预测区间是否能通过回顾性误差进行校准，而不是只依赖模型内部方差？

#### 数据模型和系统假设

输入数据被视为不规则采样的 patient-level longitudinal table。每一行对应一次化验日期，每一列对应一个化验指标或元数据。系统明确区分三类变量：

- **原始化验指标**：CEA、CA15-3、CA125、CA19-9、NSE、SCC、白细胞、血红蛋白、血小板、白蛋白、CRP 等。
- **衍生宿主状态指标**：NLR、PLR、LMR、SII、SIRI、PNI、mGPS 等，用作炎症、免疫储备、营养储备的 proxy。
- **上下文变量**：检测日期、治疗阶段、参考范围、同平台连续性、用户反馈标签等。

核心假设是保守的：化验数据是 noisy observation，不是疾病状态本身。CareTrace 因此把输出对象限定为“数据观察”“可靠变化概率”“未来化验数据事件概率”和“复核优先级”，避免将模型输出直接解释为 progression、response 或 recurrence。

#### 架构设计

系统采用页面层、控制层、数据层、模型层分离：

- `main.py` 是 Streamlit presentation layer，负责用户交互、图表和双语 UI。
- `language_support.py` 是 bilingual text layer，用轻量 key-value translation table 支持中文/English 切换。
- `controller.py`、`app_core.py`、`app_state.py` 将 Streamlit session state 与核心训练、预测、风险观察逻辑解耦，便于单元测试和未来替换 UI。
- `data_manager.py` 管理 SQLite、参考范围、校准参数、模型文件和本地持久化。
- `feature_engineering.py` 生成时序特征、阶段内趋势、阶段转换、参考范围特征和血常规/炎症/营养衍生指标。
- `baseline_monitor.py` 和 `risk_engine.py` 负责个人基线偏离、启发式规则、DTW 相似性、观察提示聚合和归因展示。
- `analysis_engine.py` 承担 IsolationForest、MOGP、特征权重、SHAP/代理归因和相似性分析。
- `sparse_longitudinal_engine.py` 处理极小样本纵向预测，强调不使用移动平均、不丢弃历史点。
- `predictive_ensemble_engine.py` 是 laboratory-only ensemble layer，整合稀疏贝叶斯趋势、状态空间代理因子、保形区间校准和任务概率。
- `simulation_engine.py` 与 `tumor_models.py` 提供适应性治疗相关的机制模型模拟，同时用宿主状态 proxy 调整不确定性表达。
- `ocr_importer.py` 使用本地 OCR 解析报告单截图，识别结果进入人工核对流程后才写入数据库。

#### 特征工程

CareTrace 的特征工程尽量避免把数据稀释成不可解释的高维向量。主要特征族包括：

- **个人基线特征**：历史中位数、MAD、modified Z-score、相对个人基线变化幅度。
- **参考范围特征**：低于下限、超过上限、接近边界、越界持续时间。
- **时间特征**：距离首次检测天数、距离上次检测天数、阶段内时间、检测间隔。
- **变化特征**：绝对变化、相对变化、斜率、阶段内趋势、阶段转换冲击。
- **衍生免疫炎症特征**：NLR、PLR、LMR、SII、SIRI。
- **营养和炎症复合特征**：PNI、mGPS、白蛋白-CRP 相关代理。

对于样本量不足的阶段，高阶趋势特征会被屏蔽或降权，避免用 1-2 个点制造伪精度。

#### 模型层

**1. 鲁棒个人基线**

系统使用 median/MAD 建立个人内基线，优先回答“和这个人过去相比是否不同”，而不是只回答“是否超出人群参考范围”。modified Z-score 用于识别相对个人基线的显著偏离。该层适合解释单个指标的突然变化，但不直接进行疾病推断。

**2. 规则观察和 IsolationForest**

规则系统处理医学上常见的可解释边界，例如参考范围越界、短期大幅变化、同向多指标变化。IsolationForest 用于发现多指标联合异常模式。两者输出都被降级为 observation，而非 alert/diagnosis。

**3. MOGP 短期趋势**

当某些指标拥有足够历史点时，系统使用 multi-output Gaussian process 思路生成短期趋势和不确定性区间。MOGP 更适合相对密集的指标序列；对于季度级极小样本，系统不会强行让 MOGP 承担全部预测任务。

**4. 稀疏纵向贝叶斯趋势**

`sparse_longitudinal_engine.py` 面向一年约 3-4 次检测的场景。它保留所有非缺失历史点，使用鲁棒个人基线、RCV、平台连续性和贝叶斯收缩。3-4 个点的个人趋势会向弱先验和临床合理范围收缩，以降低过拟合风险。对于低于方法学/生物学变异阈值的变化，系统倾向输出“未见可靠变化”。

**5. 状态空间和动态因子代理**

`predictive_ensemble_engine.py` 把不可直接观察的“数据活动度代理因子”视为 latent state，多个化验指标是带噪声观测。实现上采用轻量 Kalman-style filtering approximation，以适应小样本和本地运行限制。肿瘤标志物、CRP、白蛋白、NLR、PLR、血红蛋白等通过标准化方向共同贡献一个多指标代理因子。该因子只表示 laboratory data activity，不表示肿瘤真实负荷。

**6. 保形预测校准**

模型内部区间常常在小样本下过窄。CareTrace 使用 walk-forward residuals 做 conformal-style interval widening：如果历史一步预测误差大于模型区间暗示的不确定性，未来区间会被经验残差扩张。由于单人数据极少，这不是严格分布无关保证，而是透明的校准护栏。

**7. 集成概率**

集成层合并个人基线、稀疏贝叶斯趋势、动态因子和可选 MOGP 信号。输出任务是未来化验数据事件，例如超过参考范围、达到 RCV 可靠变化阈值或出现可复核的代理因子升高。集成概率用于排序和回测，不直接作为治疗建议。

#### 适应性治疗模拟层

模拟模块使用机制模型探索不同治疗输入假设下的理论轨迹，核心目的是 educational simulation 和 discussion support。模型可结合最新肿瘤标志物值、历史轨迹和宿主状态 proxy，展示理论轨迹和不确定性。系统不会从模拟结果反推出个人剂量，也不会建议停药、加药或延迟治疗。

#### 可解释性

CareTrace 的可解释性来自三个层面：

- 单指标层：当前值、个人基线、参考范围、modified Z-score。
- 多指标层：规则触发项、IsolationForest 贡献、SHAP 或代理归因。
- 模型层：每个预测区间、可靠变化概率、活动代理因子贡献、校准来源和样本量状态。

设计原则是让患者能看懂“为什么系统提示我复核”，同时让专业人员能追溯“这个提示来自哪类模型和哪些输入”。

#### 回顾性验证

`测试脚本/retrospective_validation.py` 提供合成数据上的 retrospective validation。当前指标包括：

- false positive rate；
- false negative rate；
- prediction interval coverage；
- mean interval width；
- Brier score；
- calibration curve；
- false alarms per year；
- mean lead time days；
- sensitivity at fixed false-positive rate；
- sample-size stability。

这些指标的目标不是证明临床有效，而是防止模型在工程迭代中失控：例如区间越来越窄、误报次数增加、概率校准变差、极小样本下过拟合。

#### 局限性

- 当前系统没有真实多中心临床队列验证。
- 合成数据回测不能替代前瞻性临床研究。
- 肿瘤标志物在乳腺癌中的敏感性、特异性和治疗早期 spike 现象均限制其解释。
- CBC 衍生指标是宿主状态 proxy，受感染、药物、营养、采样时点和合并症影响。
- 影像、病理、体格检查和患者症状仍是临床判断不可替代的组成部分。
- 当前深度学习成分是辅助性的；系统更强调小样本统计、可解释模型和校准，而不是大模型端到端诊断。

#### 未来研究方向

- 加入结构化症状日记，而不是要求患者填写“疑似进展”这类心理负担较重的标签。
- 引入匿名真实世界回测数据，分癌种、分药物、分检测平台评估校准。
- 建立医生确认的重要变化标签后，再评估 important-change capture rate。
- 对 OCR 输出加入更强的单位规范化和跨医院项目名映射。
- 将 conformal calibration 从单患者残差扩展为可选择的本地群体校准。
- 探索 federated 或 privacy-preserving evaluation，让患者数据不离开本地也能改进模型评估。

运行测试：

```powershell
python -m unittest discover -s 测试脚本 -p "test_*.py"
python .\测试脚本\retrospective_validation.py
```

### 论文和依据

适应性治疗与肿瘤演化：

- Gatenby RA, Silva AS, Gillies RJ, Frieden BR. Adaptive Therapy. *Cancer Research*, 2009. DOI: [10.1158/0008-5472.CAN-08-3658](https://doi.org/10.1158/0008-5472.CAN-08-3658).
- Enriquez-Navas PM et al. Exploiting evolutionary principles to prolong tumor control in preclinical models of breast cancer. *Science Translational Medicine*, 2016. DOI: [10.1126/scitranslmed.aad7842](https://doi.org/10.1126/scitranslmed.aad7842).
- Zhang J et al. Integrating evolutionary dynamics into treatment of metastatic castrate-resistant prostate cancer. *Nature Communications*, 2017. DOI: [10.1038/s41467-017-01968-5](https://doi.org/10.1038/s41467-017-01968-5).
- Mistry HB. On the reporting and analysis of a cancer evolutionary adaptive dosing trial. *Nature Communications*, 2021. DOI: [10.1038/s41467-020-20174-4](https://doi.org/10.1038/s41467-020-20174-4).

肿瘤标志物和乳腺癌监测边界：

- Harris L et al. American Society of Clinical Oncology 2007 update of recommendations for the use of tumor markers in breast cancer. *Journal of Clinical Oncology*, 2007. DOI: [10.1200/JCO.2007.14.2364](https://doi.org/10.1200/JCO.2007.14.2364).
- Current Approaches and Challenges in Monitoring Treatment Responses in Breast Cancer. *Journal of Cancer*, 2014. [PMC3881221](https://pmc.ncbi.nlm.nih.gov/articles/PMC3881221/).
- Yasasever V et al. Utility of CA 15-3 and CEA in monitoring breast cancer patients with bone metastases: special emphasis on spiking phenomena. *Clinical Biochemistry*, 1997. DOI: [10.1016/S0009-9120(96)00133-6](https://doi.org/10.1016/S0009-9120(96)00133-6).

血常规、炎症和营养代理指标：

- Cupp MA et al. Neutrophil to lymphocyte ratio and cancer prognosis: an umbrella review. *BMC Medicine*, 2020. DOI: [10.1186/s12916-020-01817-1](https://doi.org/10.1186/s12916-020-01817-1).
- Sun K et al. The prognostic significance of the prognostic nutritional index in cancer. *Journal of Cancer Research and Clinical Oncology*, 2014. DOI: [10.1007/s00432-014-1714-3](https://doi.org/10.1007/s00432-014-1714-3).
- Zhou Q et al. Systemic Inflammation Response Index as a prognostic marker in cancer patients. *Dose-Response*, 2021. DOI: [10.1177/15593258211064744](https://doi.org/10.1177/15593258211064744).

状态空间、动态预测和不确定性：

- Kalman RE. A New Approach to Linear Filtering and Prediction Problems. *Journal of Basic Engineering*, 1960. DOI: [10.1115/1.3662552](https://doi.org/10.1115/1.3662552).
- West M, Harrison J. *Bayesian Forecasting and Dynamic Models*. Springer, 1997. DOI: [10.1007/b98971](https://doi.org/10.1007/b98971).
- Angelopoulos AN, Bates S. Conformal Prediction: A Gentle Introduction. *Foundations and Trends in Machine Learning*, 2023. DOI: [10.1561/2200000101](https://doi.org/10.1561/2200000101).
- Zhou Q, Chen ZH, Peng S. Evaluating the application of dynamic prediction models in oncological prognostic studies with repeated measurement predictors. *npj Precision Oncology*, 2025. DOI: [10.1038/s41698-025-01162-7](https://doi.org/10.1038/s41698-025-01162-7).
- Steyerberg EW et al. Assessing the performance of prediction models: a framework for traditional and novel measures. *Epidemiology*, 2010. DOI: [10.1097/EDE.0b013e3181c30fb2](https://doi.org/10.1097/EDE.0b013e3181c30fb2).

PD-1 与抗血管生成联合治疗背景：

- Lee WS et al. Combination of anti-angiogenic therapy and immune checkpoint blockade normalizes vascular-immune crosstalk to potentiate cancer immunity. *Experimental & Molecular Medicine*, 2020. DOI: [10.1038/s12276-020-00500-y](https://doi.org/10.1038/s12276-020-00500-y).
- Kuo HY, Khan KA, Kerbel RS. Antiangiogenic-immune-checkpoint inhibitor combinations: lessons from phase III clinical trials. *Nature Reviews Clinical Oncology*, 2024. DOI: [10.1038/s41571-024-00886-y](https://doi.org/10.1038/s41571-024-00886-y).

<a id="en"></a>

## English

> [切换到中文](#zh)

### Why CareTrace exists

CareTrace began as software written for my mother. She has breast cancer with axillary lymph-node involvement and has undergone surgery. During treatment, I read papers about adaptive therapy, intermittent low-dose approaches, PD-1 blockade, anti-angiogenic treatment, tumor markers, CBC-derived inflammation indices, and nutritional status.

Cancer treatment can be emotionally dark and cognitively exhausting. Hospitals must follow guidelines and standard pathways; that is the foundation of safe medicine. But outside the hospital, patients and families still need to organize reports, understand trends, prepare better questions, record symptoms, avoid panic from single noisy values, and avoid missing changes that deserve review. CareTrace grew from that need.

CareTrace is not a doctor and does not replace a doctor. It is a local tool that turns sparse, precious lab data into clearer observation material for clinician discussions.

### What CareTrace can do for patients and families

- Store lab reports locally: CBC, biochemistry, CRP, tumor markers, thyroid tests, and more.
- Reduce manual typing with local OCR from report screenshots or photos.
- Compare current values with reference ranges and personal historical baselines.
- Preserve sparse data: for patients tested only a few times per year, the sparse longitudinal engine avoids moving-average windows that discard early points.
- Produce gentle observation notes instead of diagnostic claims.
- Help prepare clinic visits by highlighting changes worth discussing.
- Provide adaptive-therapy mathematical simulations for discussion, not prescription.
- Keep data private: patient data stay on the local computer by default.

### Boundaries

CareTrace does not:

- diagnose recurrence, progression, metastasis, or remission;
- determine treatment response;
- recommend dosing, drug holds, or drug changes;
- replace imaging, pathology, physical examination, or oncology follow-up;
- use tumor markers alone to judge disease status;
- guarantee prediction accuracy.

All treatment decisions should be made with qualified oncology professionals.

### Adaptive therapy in plain language

The basic idea of adaptive therapy is evolutionary. Tumors may contain both drug-sensitive and relatively resistant cells. Continuous maximum pressure can remove sensitive cells and leave resistant cells with less competition. In some settings, controlled treatment modulation aims to preserve some sensitive cells so they continue competing with resistant cells, potentially delaying resistance.

This is not a universal rule and not a reason to change treatment on one’s own. It depends on cancer type, tumor burden, drug class, imaging, symptoms, immune state, toxicity, and clinician judgment. CareTrace only simulates theoretical trajectories to help patients ask better questions.

### For clinicians and researchers

CareTrace is a local Streamlit prototype for a common but under-modeled real-world setting: an individual oncology patient may have only a few follow-up laboratory panels per year, mostly CBC, biochemistry, inflammatory markers, and tumor markers, without high-frequency vitals, standardized imaging annotations, or clinician-confirmed outcome labels. The goal is not end-to-end diagnosis. The goal is an interpretable, auditable, retrospectively testable longitudinal laboratory observation framework.

#### Research questions

CareTrace is designed to answer lower-level, more verifiable questions rather than “has the cancer progressed?”:

- Is the current value meaningfully different from the patient’s own historical baseline?
- In a sparse series with only 3-4 annual observations, how can all historical points be preserved rather than discarded by rolling-window smoothing?
- How should uncertainty be expressed when testing intervals are irregular, laboratory platforms may change, and tumor markers have biological and analytical variation?
- Can multiple weak laboratory signals form a “data activity proxy factor” without being interpreted as disease burden?
- Can prediction intervals be calibrated against retrospective errors rather than relying only on model-internal variance estimates?

#### Data model and assumptions

The input is treated as an irregularly sampled patient-level longitudinal table. Each row corresponds to a report date, and each column corresponds to a laboratory variable or metadata field. CareTrace separates three groups of variables:

- **Raw laboratory variables**: CEA, CA15-3, CA125, CA19-9, NSE, SCC, WBC, hemoglobin, platelets, albumin, CRP, and related routine markers.
- **Derived host-state proxies**: NLR, PLR, LMR, SII, SIRI, PNI, and mGPS, used as proxies for inflammation, immune reserve, and nutritional reserve.
- **Context variables**: report date, treatment phase, reference range, same-platform continuity, and optional user feedback labels.

The central assumption is deliberately conservative: laboratory values are noisy observations, not the disease state itself. Therefore, CareTrace limits its outputs to data observations, reliable-change probabilities, future laboratory-event probabilities, and review priority. It avoids translating model outputs directly into progression, response, or recurrence.

#### Architecture

CareTrace separates presentation, control, storage, and modeling concerns:

- `main.py` is the Streamlit presentation layer, including charts, user interactions, and bilingual UI.
- `language_support.py` is a lightweight bilingual text layer for Chinese/English switching.
- `controller.py`, `app_core.py`, and `app_state.py` decouple Streamlit session state from training, prediction, risk observation, and simulation workflows.
- `data_manager.py` manages SQLite storage, reference ranges, calibrated parameters, model artifacts, and local persistence.
- `feature_engineering.py` builds temporal features, phase-aware features, reference-range features, and CBC/inflammation/nutrition-derived indices.
- `baseline_monitor.py` and `risk_engine.py` aggregate personal-baseline deviations, rule-based observations, DTW similarity, and observation notes.
- `analysis_engine.py` provides IsolationForest, MOGP, feature weighting, SHAP/proxy attribution, and similarity analysis.
- `sparse_longitudinal_engine.py` handles extremely small longitudinal samples without moving-average point loss.
- `predictive_ensemble_engine.py` implements the laboratory-only ensemble layer with sparse Bayesian trends, state-space proxy factor, conformal interval calibration, and task probabilities.
- `simulation_engine.py` and `tumor_models.py` implement adaptive-therapy-related mechanistic simulations and host-state proxy uncertainty.
- `ocr_importer.py` parses report screenshots locally and requires human review before writing recognized values to the database.

#### Feature engineering

CareTrace avoids turning sparse laboratory data into an opaque high-dimensional representation. The main feature families are:

- **Personal baseline features**: historical median, MAD, modified Z-score, and relative deviation from individual baseline.
- **Reference-range features**: below lower limit, above upper limit, near-boundary status, and duration since crossing.
- **Temporal features**: days since first report, days since previous report, within-phase time, and irregular interval length.
- **Change features**: absolute change, relative change, slope, within-phase trend, and phase-transition shock.
- **Immune-inflammatory indices**: NLR, PLR, LMR, SII, and SIRI.
- **Nutrition/inflammation composites**: PNI, mGPS, and albumin-CRP proxy signals.

When a phase has too few samples, higher-order trend features are masked or down-weighted to avoid manufacturing precision from one or two observations.

#### Model layer

**1. Robust personal baseline**

The baseline layer uses median/MAD and modified Z-scores to prioritize the question: “Is this value unusual for this person?” rather than only “Is this value outside the population reference range?” This layer is interpretable and useful for single-indicator review, but it does not infer disease state.

**2. Rule-based observations and IsolationForest**

Rules cover clinically interpretable boundaries such as reference-range crossing, abrupt short-term changes, and concordant multi-indicator movement. IsolationForest is used for multi-variable anomaly detection. Both outputs are downgraded to observations rather than alerts or diagnoses.

**3. MOGP short-term trend modeling**

For indicators with sufficient history, CareTrace uses a multi-output Gaussian-process-style forecast to represent short-term trends and uncertainty. MOGP is reserved for denser indicator histories; it is not forced to explain quarterly extremely small samples alone.

**4. Sparse longitudinal Bayesian trend model**

`sparse_longitudinal_engine.py` targets the setting of roughly quarterly testing. It preserves all non-missing historical points and combines robust personal baseline, reference change value (RCV), same-platform continuity, and Bayesian shrinkage. Personal slopes estimated from 3-4 points are shrunk toward weak priors and clinically plausible ranges to reduce overfitting. Changes below biological or analytical variation thresholds are described as “no reliable change observed.”

**5. State-space and dynamic-factor proxy**

`predictive_ensemble_engine.py` treats an unobserved “laboratory data activity proxy” as a latent state and laboratory values as noisy observations. The current implementation uses a lightweight Kalman-style filtering approximation suitable for local execution and small sample sizes. Tumor markers, CRP, albumin, NLR, PLR, hemoglobin, and related features contribute through standardized directions to a multi-indicator proxy factor. This factor represents laboratory activity only; it is not a tumor-burden estimate.

**6. Conformal-style interval calibration**

Model-generated intervals are often overconfident in small samples. CareTrace uses walk-forward residuals for conformal-style interval widening: if previous one-step predictions missed by more than the model interval implied, future intervals are expanded using empirical residuals. Because single-patient histories are tiny, this is not claimed as a strict distribution-free guarantee; it is a transparent calibration guardrail.

**7. Ensemble probability**

The ensemble layer combines personal baseline, sparse Bayesian trend, dynamic-factor proxy, and optional MOGP signals. The predicted task is a future laboratory data event, such as crossing a reference range, reaching an RCV reliable-change threshold, or showing a review-worthy proxy-factor increase. The ensemble probability is used for ranking and retrospective evaluation, not for treatment recommendations.

#### Adaptive-therapy simulation layer

The simulation module explores theoretical trajectories under user-defined treatment-input schedules. Its purpose is educational simulation and discussion support. It can incorporate the latest tumor-marker value, historical trajectory, and host-state proxies to display theoretical curves and uncertainty. It never reverse-engineers personal dosing and never recommends drug holds, escalation, de-escalation, or schedule changes.

#### Interpretability

Interpretability is provided at three levels:

- Single-indicator level: current value, personal baseline, reference range, and modified Z-score.
- Multi-indicator level: rule triggers, IsolationForest contribution, SHAP or proxy attribution.
- Model level: prediction interval, reliable-change probability, activity-proxy contribution, calibration source, and sample-size status.

The design goal is that patients can understand why a review prompt appeared, while clinicians and researchers can trace which model family and which inputs produced it.

#### Retrospective validation

`测试脚本/retrospective_validation.py` provides synthetic-data retrospective validation. Current metrics include:

- false positive rate;
- false negative rate;
- prediction interval coverage;
- mean interval width;
- Brier score;
- calibration curve;
- false alarms per year;
- mean lead time in days;
- sensitivity at fixed false-positive rate;
- sample-size stability.

These metrics do not prove clinical utility. They are guardrails against engineering drift: intervals becoming too narrow, false alarms increasing, probability calibration degrading, or tiny-sample overfitting worsening.

#### Limitations

- CareTrace has not been validated on a real multicenter clinical cohort.
- Synthetic retrospective tests cannot replace prospective clinical evaluation.
- Tumor markers in breast cancer have limited sensitivity and specificity and can show early-treatment spikes.
- CBC-derived indices are host-state proxies affected by infection, drugs, nutrition, sampling time, and comorbidities.
- Imaging, pathology, physical examination, and patient-reported symptoms remain indispensable for clinical judgment.
- Deep learning is currently auxiliary; the system emphasizes small-sample statistics, interpretability, and calibration rather than black-box end-to-end diagnosis.

#### Future directions

- Add structured symptom journaling without asking patients to label themselves as “suspected progression.”
- Evaluate calibration using anonymized real-world retrospective data stratified by cancer type, drug class, and laboratory platform.
- Add clinician-confirmed important-change labels before evaluating important-change capture rate.
- Strengthen OCR unit normalization and cross-hospital item-name mapping.
- Extend conformal calibration from single-patient residuals to optional local cohort calibration.
- Explore federated or privacy-preserving evaluation so patient data can stay local while model evaluation improves.

Install and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run main.py
```

Run tests:

```powershell
python -m unittest discover -s 测试脚本 -p "test_*.py"
python .\测试脚本\retrospective_validation.py
```

### Author, Repository, and License

- Author: Liu Yanwei.
- GitHub: <https://github.com/waylliam1988/CareTrace>.
- License: Apache License 2.0.
- Attribution: CareTrace includes a `NOTICE` file. Under the Apache-2.0 NOTICE mechanism, users who use, copy, modify, distribute, or build upon CareTrace should retain the `LICENSE`, `NOTICE`, and original author attribution.
- Medical boundary: CareTrace is not a medical diagnosis or treatment system. It does not provide medication, dosing, drug-hold, emergency, or clinical decision-making instructions.

### Privacy

CareTrace is local-first. The default database, OCR workflow, model files, and logs stay on the user’s computer. Do not upload real patient databases, report images, model artifacts, or logs to public repositories.
