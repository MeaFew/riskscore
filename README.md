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

基于 Kaggle Home Credit Default Risk 数据集的端到端信用风险评分管线。实现工业级特征工程（WOE/IV）、多模型对比（LR → RF → XGBoost → LightGBM）与生产级可解释性（SHAP）。

## 核心亮点

- **特征工程**：WOE 分箱、IV 筛选、目标编码、交叉特征
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
│   ├── feature_engineering.py     # WOE/IV、目标编码、交叉特征
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
| **本方案（单表）** | **0.763** | WOE/IV + 目标编码 + XGBoost/LightGBM（5 折 CV） |
| **本方案（多表）** | **0.783** | （预期值，需完整 Kaggle 辅助表数据） |

> 注：竞赛 Private Leaderboard 已关闭。上述分数为真实 Kaggle 数据上的本地 5 折分层交叉验证（307,511 训练 / 48,744 测试）。多表结果需在 `make features` 前运行 `scripts/aggregate_auxiliary_features.py` 和 `scripts/merge_auxiliary_features.py`。

### 结果

| 模型 | AUC | KS | Gini |
|------|-----|-----|------|
| 逻辑回归 | 0.654 | 0.233 | 0.308 |
| 随机森林 | 0.745 | 0.366 | 0.490 |
| XGBoost | **0.762** | **0.394** | **0.525** |
| LightGBM | **0.763** | **0.394** | **0.525** |

> 数值来自单表特征（仅 application_train）的真实 Kaggle 数据（307,511 样本）5 折分层交叉验证。留出测试集（80/20 切分）AUC = **0.801**（XGBoost）。引入多表特征（bureau、previous_application 等）可通过运行辅助聚合脚本进一步提升 AUC。

## 相关项目

| 项目 | Gitee（主仓） | GitHub（镜像） |
|------|---------------|-----------------|
| 电商用户行为分析 | [Gitee](https://gitee.com/zeroonei1/shoplytics) | [GitHub](https://github.com/MeaFew/shoplytics) |
| 营销归因与预算优化 | [Gitee](https://gitee.com/zeroonei1/attributor) | [GitHub](https://github.com/MeaFew/attributor) |
| 多元时序预测 | [Gitee](https://gitee.com/zeroonei1/foresight) | [GitHub](https://github.com/MeaFew/foresight) |

## 许可证

MIT
