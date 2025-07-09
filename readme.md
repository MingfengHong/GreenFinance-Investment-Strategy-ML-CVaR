# 基于跳跃扩散、机器学习与CVaR优化的多资产绿色金融投资策略研究 Multi-Asset Green Finance Investment Strategy based on Jump-Diffusion, Machine Learning, and CVaR Optimization

## 1. 项目概述

本项目是《计算金融与仿真》课程论文《基于跳跃扩散、机器学习与CVaR优化的多资产绿色金融投资策略研究》的代码实现。项目聚焦于中国市场的三种代表性绿色金融资产：

* 南方中证新能源ETF (NETF)
* 中债-中国绿色债券指数 (GBI)
* 中国碳排放配额现货 (CEA)

我们构建并回测了一个动态投资组合优化策略。该策略旨在有效捕捉绿色金融资产复杂的动态特性，并进行先进的风险管理。

## 2. 核心方法与特点

本研究框架整合了多个金融工程与机器学习技术：

* **随机过程建模**:
    * 为收益平稳的GBI资产构建**几何布朗运动 (GBM)** 模型。
    * 为具有价格跳跃特性的NETF和CEA资产构建**Merton跳跃扩散模型**。
* **机器学习动态调整**:
    * 创新性地采用**随机森林 (Random Forest)** 算法，基于能源期货市场的特征数据，对Merton模型中的漂移项 (µ) 进行动态预测和调整，以增强模型对市场变化的适应性。
* **相关性建模**:
    * 通过**滚动窗口**和**Cholesky分解**捕捉资产间动态变化的相关性，以生成更贴近现实的多资产联合价格路径。
* **投资组合优化**:
    * 采用**条件风险价值 (CVaR)** 作为风险度量，以最小化投资组合的尾部风险为目标，在满足预期收益约束的前提下进行资产权重配置。

## 3. 代码工作流

本项目代码被划分为6个核心步骤，按顺序执行即可复现整个研究过程。

#### **步骤 0: 数据预处理**
* `convert_json.py`: 将原始JSON格式的数据转换为CSV表格。
* `process_data.py`: 合并多个资产的原始价格数据（`etf_price.xlsx`, `gbi_price.xlsx`, `cea_price.xlsx`），处理列名和日期格式，执行内连接提取共同交易日，生成统一的 `price_data.xlsx` 文件。

#### **步骤 1: 探索性数据分析 (EDA)**
* **`1_exploratory_analysis.py`**:
    * 加载合并后的价格数据。
    * 计算对数收益率。
    * 进行描述性统计、正态性检验（如Jarque-Bera检验）和可视化分析（如时序图、Q-Q图、相关性矩阵热力图）。
    * 根据研究设计，将数据集划分为阶段A（静态参数估计）、B（ML训练）、C（动态策略回测）和D（静态策略延伸回测）。

#### **步骤 2: 静态参数估计**
* **`2_static_parameter_estimation.py`**:
    * 使用阶段A的数据。
    * 为GBI资产估计GBM模型的静态参数 ($\mu_{base}$, $\sigma_{base}$)
    * 为NETF和CEA资产估计Merton跳跃扩散模型的静态参数 ($\mu_{base}$, $\sigma_{base}$, $\lambda_{base}$, $mJ_{base}$, $\sigma J_{base}$)。
    * 将所有基础参数保存到 `all_assets_base_params.json`。

#### **步骤 3: 参数动态化 (机器学习)**
* **`3_parameter_adjusting.py`**:
    * 使用阶段B的数据进行模型训练。
    * 基于能源期货的滞后收益率和滚动平均构建特征。
    * 为NETF和CEA分别训练随机森林回归模型，以预测未来20日的平均年化收益率（即动态漂移项 $\mu_{dyn}$）。
    * 通过`GridSearchCV`和`TimeSeriesSplit`进行超参数调优和时序交叉验证。
    * 保存训练好的模型 (`.joblib`文件) 和特征重要性等可视化结果。

#### **步骤 4: 多资产价格路径模拟**
* **`4_multi_asset_simulation.py`**:
    * 在回测期（阶段C和D）的每个交易日，执行滚动模拟。
    * 使用60日滚动窗口计算资产间的相关性矩阵，并进行Cholesky分解。
    * 结合静态参数和ML模型预测的动态漂移项，使用蒙特卡洛方法模拟生成10,000条未来20天的多资产联合价格路径。
    * 将所有日期的模拟路径结果保存到 `simulation_results_multi_asset.joblib`。

#### **步骤 5: 投资组合优化**
* **`5_portfolio_optimization_cvar.py`**:
    * 在回测期的每个交易日，加载当日生成的模拟价格路径。
    * 构建一个以最小化95% CVaR为目标、年化8%收益为约束的线性规划问题。
    * 求解优化问题，得到当日最优的资产配置权重。
    * 如果优化失败，则退回至等权重策略。
    * 保存每日的优化权重和预期的风险度量（VaR/CVaR）。

#### **步骤 6: 策略回测与绩效评估**
* **`6_portfolio_backtesting.py`**:
    * 加载每日的优化权重和资产的实际日收益率。
    * 计算CVaR优化策略的每日收益，并与基准策略（等权重、单一资产买入持有）进行对比。
    * 计算并展示各策略的关键绩效指标，如年化回报率、年化波动率、夏普比率和最大回撤。
    * 绘制累计净值曲线、回撤曲线等可视化图表，对策略表现进行全面评估。

## 4. 如何运行

1.  **环境配置**
    * 安装 Python 3.x。
    * 安装所有必需的依赖库：
        ```bash
        pip install pandas numpy scikit-learn joblib statsmodels scipy matplotlib seaborn openpyxl
        ```

2.  **数据准备**
    * 确保 `etf_price.xlsx`, `gbi_price.xlsx`, `cea_price.xlsx` 等原始数据文件位于根目录。
    * 运行 `process_data.py` 和 `convert_json.py` (如果需要) 来生成初始的合并价格文件。

3.  **执行工作流**
    * 严格按照脚本文件名前的数字顺序执行 `.py` 文件：
        ```bash
        python 1_exploratory_analysis.py
        python 2_static_parameter_estimation.py
        python 3_parameter_adjusting.py
        python 4_multi_asset_simulation.py
        python 5_portfolio_optimization_cvar.py
        python 6_portfolio_backtesting.py
        ```
    * 每个脚本都会生成其所需的数据文件和可视化图表，供下一步使用。

## 5. 主要结论

* **策略有效性**: 在回测期内，本研究构建的动态CVaR优化策略在综合表现上优于等权重及单一资产买入持有策略。
* **风险控制**: 与高回报的单一资产（如NETF）相比，本策略在波动率和最大回撤方面展现出显著的风险控制优势。
* **风险调整后收益**: 策略的夏普比率高于等权重和高风险的单一资产策略，表明其在每单位风险所获取的超额回报方面具有相对优势。

## 6. 未来展望

本研究框架仍有提升空间，未来可以从以下方面进行深化：
* **模型增强**: 尝试更先进的机器学习模型（如LSTM、GRU）和更复杂的跳跃过程。
* **优化框架**: 考虑多阶段随机规划，并融入交易成本等现实约束。
* **风险建模**: 引入Copula函数等工具刻画非线性依赖结构，或使用更精细的风险因子模型。
* **压力测试**: 模拟不同的极端市场情景，评估策略的稳健性。
* **重要！！！没有考虑交易摩擦！！！**