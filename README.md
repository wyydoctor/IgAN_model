# IgAN_model

# 模型训练与评估框架

本仓库提供一个用于对比 `IANet` 和 `ResNet18Baseline` 模型在图像分类任务上性能的训练框架，支持自动评估、早停机制和结果可视化。

## 功能说明
- 同时训练两个模型并实时对比性能
- 计算准确率、AUC、TPR、FPR 等评估指标
- 实现早停机制防止过拟合
- 自动保存最佳模型和训练结果

## 环境依赖
python >= 3.8
pytorch >= 1.10.0
scikit-learn >= 1.0
tqdm >= 4.62.0
matplotlib >= 3.5.0
numpy >= 1.21.0

