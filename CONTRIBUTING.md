# Contributing Guide

感谢你对本项目的兴趣. 本指南面向希望本地运行、调试或扩展该信用风险评分项目的开发者.

## 环境准备

```bash
# 1. 克隆仓库
git clone https://github.com/MeaFew/riskscore.git
cd riskscore

# 2. 创建虚拟环境 (推荐 Python 3.12)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt
```

## 数据准备

项目使用 Kaggle Home Credit Default Risk 数据集. 请运行下载脚本获取数据集:

```bash
bash download_data.sh
```

## 本地工作流

```bash
# 1. 数据预处理
make preprocess

# 2. 特征工程 (WOE/IV + target encoding)
make features

# 3. 模型训练 (LR/RF/XGB/LGBM)
make train

# 4. 模型评估 (AUC/KS/Gini)
make evaluate

# 5. SHAP 可解释性分析
make shap

# 6. 启动看板
make dashboard
```

## 代码规范

提交前请确保通过以下检查:

```bash
# Python lint
ruff check scripts/ dashboard/ --ignore E501,F401,E402

# 单元测试
pytest tests/ -v
```

## 提交规范

- `feat:` 新功能
- `fix:` 修复 bug
- `docs:` 文档更新
- `refactor:` 重构
- `ci:` 持续集成相关
- `test:` 测试相关

## 扩展建议

- 新增模型: 在 `scripts/train_models.py` 中添加
- 新增特征: 在 `scripts/feature_engineering.py` 中扩展 WOE/IV 逻辑
- 新增分析: 放在 `scripts/` 并更新 Makefile
