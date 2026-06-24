# 信用风险评分

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/XGBoost-2.0-green?logo=xgboost&logoColor=white" alt="XGBoost">
  <img src="https://img.shields.io/badge/LightGBM-4.0-blue?logo=lightgbm&logoColor=white" alt="LightGBM">
  <img src="https://img.shields.io/badge/SHAP-0.42-orange?logo=shap&logoColor=white" alt="SHAP">
  <a href="https://github.com/MeaFew/riskscore/actions"><img src="https://github.com/MeaFew/riskscore/workflows/CI/badge.svg" alt="CI"></a>
</p>

<p align="center">
  🏠 <b>主仓：<a href="https://gitee.com/zeroonei1/riskscore">Gitee</a></b> &nbsp;|&nbsp;
  🔗 <a href="https://github.com/MeaFew/riskscore">GitHub（自动同步）</a>
</p>

<p align="center">
  <b>中文</b> | <a href="./README.en.md">English</a>
</p>

## 项目简介

基于 Kaggle Home Credit Default Risk 数据集的端到端信用风险评分管线。实现专业级特征工程（WOE/IV）、多模型对比（LR → RF → XGBoost → LightGBM）与SHAP 可解释性。

## 核心亮点

- **特征工程**：WOE 分箱（分析参考）、目标编码（per-fold，无泄漏）、交叉特征
- **模型栈**：逻辑回归（基线）→ 随机森林 → XGBoost → LightGBM
- **评估**：AUC、KS、Gini、校准曲线、最优阈值下的混淆矩阵
- **可解释性**：SHAP summary、dependence plot、个体 force plot
- **交付**：Streamlit 风险计算器看板

## 技术栈

| 层级 | 工具 | 说明 |
|------|------|------|
| ETL | pandas, scikit-learn | 缺失值填补、异常值截断 |
| 特征工程 | 自研 WOE/IV | 分位数分箱 + 平滑处理 |
| 建模 | XGBoost, LightGBM, sklearn | 5 折分层交叉验证 |
| 可解释性 | SHAP | 梯度提升模型使用 TreeExplainer |
| 评估 | scipy, sklearn | AUC、KS、Gini、PR 曲线、校准 |
| 交付 | Streamlit | 交互式风险计算器 + 模型对比 |
| 质量保障 | pytest, ruff, GitHub Actions | CI 每次 push 跑 lint + 测试 |

## 快速开始

```bash
# 从 Gitee 克隆（国内推荐，速度更快）
git clone https://gitee.com/zeroonei1/riskscore.git

# 或从 GitHub
git clone https://github.com/MeaFew/riskscore.git
cd riskscore

# 下载真实数据集（GitHub Releases，约 40MB）
bash download_data.sh

# 运行完整管线
make all
# Windows (无 GNU Make): python run_all.py

# 或分步执行
make preprocess
make features
make train
make evaluate
make shap

# 启动看板
make dashboard

# 质量门
make verify
```

## 项目结构

```
.
├── scripts/
│   ├── generate_mock_data.py     # 合成数据生成（CI 用）
│   ├── preprocess.py              # 数据清洗与缺失值处理
│   ├── feature_engineering.py     # 交叉特征 + WOE/IV 分析报告（目标编码移入 CV Pipeline）
│   ├── train_models.py            # LR / RF / XGB / LGBM + 交叉验证
│   ├── evaluate.py                # ROC、PR、校准、混淆矩阵
│   └── shap_analysis.py           # SHAP summary、dependence、force plot
├── dashboard/
│   └── app.py                     # Streamlit 交互看板
├── tests/
│   └── test_pipeline.py           # 单元 + 集成测试
├── config.py                      # 集中式路径与超参数配置
├── Makefile                       # 工作流编排
└── requirements.txt
```

## 模型表现

### 基准参照

基于 [Kaggle Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)（7,190+ 队伍，评估指标：AUC-ROC）。

| 参照 | AUC | 说明 |
|------|-----|------|
| Kaggle Starter 基线 | 0.688 | 官方 starter notebook，无特征工程 |
| 单表逻辑回归 | 0.748 | 仅 `application_train` + GridSearchCV |
| 单表 LightGBM | 0.749 | 同上，梯度提升 |
| 竞赛中位数 | ~0.72–0.75 | Leaderboard 中位 |
| 竞赛 Top 10% | ~0.795 | 多表特征 + 集成 |
| **本方案（单表）** | **0.766** | 目标编码（per-fold，无泄漏）+ XGBoost/LightGBM（5 折 CV / OOF） |
| 本方案（多表，规划中） | ~0.78（预估） | 需完整辅助表数据；当前仓库仅实现单表 |

> 注：竞赛 Private Leaderboard 已关闭。上述分数为真实 Kaggle 数据（307,511 训练样本，`application_train`）上的本地 5 折分层交叉验证 + 泄漏无关的 out-of-fold (OOF) 评估。多表结果需在 `make features` 前运行 `scripts/aggregate_auxiliary_features.py` 和 `scripts/merge_auxiliary_features.py`。

### 结果

| 模型 | AUC | KS | Gini |
|------|-----|-----|------|
| 逻辑回归 | 0.626 | 0.192 | 0.251 |
| 随机森林 | 0.746 | 0.365 | 0.491 |
| XGBoost | **0.766** | **0.399** | **0.533** |
| LightGBM | **0.766** | **0.398** | **0.532** |

> 数值来自单表特征（仅 application_train）的真实 Kaggle 数据（307,511 样本）5 折分层交叉验证，目标编码通过 sklearn `Pipeline` 在**每折训练子集**上单独 fit（验证行的 target 不参与自身编码，杜绝泄漏）。XGBoost 的 OOF AUC = **0.766**（每行由未见该行的模型打分，是可靠的泛化估计）。

### 防泄漏说明（重要）

本管线对常见的两类数据泄漏做了修正：

- **目标编码泄漏**：早期版本在全训练集上算完目标编码再喂进 5 折 CV，导致验证行的目标值泄漏进自身编码特征、AUC 偏高。现改为在 sklearn `Pipeline` 内用 `TargetEncoder`，每折仅在训练子集上 fit。IV 特征选择也从训练流程中移除（仅在 `data/processed/iv_report.csv` 作分析参考），避免"用全量 target 选特征再用到 CV 折内"的同类泄漏。
- **伪造 test AUC**：Home Credit 的 `application_test.csv` 无 `TARGET`，没有可评估的标签集。早期版本把训练集 80/20 切分、在全量数据上重训后再评估，得到一个 0.80 的"test AUC"——这是重代入（resubstitution）数字，且高于诚实的 CV AUC，方向错误。现已删除该指标，改为报告 OOF AUC。
- **预处理泄漏**：异常值截断边界与中位数填充现仅在 train 上 fit，再 transform 到 test。

## 相关项目

| 项目 | Gitee（主仓） | GitHub（镜像） |
|------|---------------|-----------------|
| 电商用户行为分析 | [Gitee](https://gitee.com/zeroonei1/shoplytics) | [GitHub](https://github.com/MeaFew/shoplytics) |
| 营销归因与预算优化 | [Gitee](https://gitee.com/zeroonei1/attributor) | [GitHub](https://github.com/MeaFew/attributor) |
| 多元时序预测 | [Gitee](https://gitee.com/zeroonei1/foresight) | [GitHub](https://github.com/MeaFew/foresight) |
| 图神经网络反欺诈 | [Gitee](https://gitee.com/zeroonei1/graphguard) | [GitHub](https://github.com/MeaFew/graphguard) |

## 许可证

MIT
