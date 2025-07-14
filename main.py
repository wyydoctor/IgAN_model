import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import json
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix
import numpy as np
from tqdm import tqdm
import random
import time
import matplotlib.pyplot as plt
from copy import deepcopy
from jpg_data_pre.data_loader import *
from model_arc.model_code import ResNet18Baseline,IANet


# 设置随机种子确保实验可复现
def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




# 加载数据列表
def load_list_from_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


# 计算评估指标
def calculate_metrics(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    labels_np = labels.cpu().numpy()
    predicted_np = predicted.cpu().numpy()

    tn, fp, fn, tp = confusion_matrix(labels_np, predicted_np).ravel()
    tpr = tp / (tp + fn) if (tp + fn) != 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) != 0 else 0

    probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
    auc = roc_auc_score(labels_np, probs)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return accuracy, auc, tpr, fpr


# 模型评估
def evaluate_model(model, loader, device):
    model.eval()
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            a = inputs[0].to(device)
            b = inputs[1].to(device)
            labels = labels.to(device)

            # 根据模型类型准备输入
            if isinstance(model, ResNet18Baseline):
                outputs = model(a, b)
            else:
                outputs = model(a, b)

            all_outputs.append(outputs)
            all_labels.append(labels)

    return torch.cat(all_outputs), torch.cat(all_labels)


# 早停机制
class EarlyStopping:
    def __init__(self, patience=10, delta=0.001, verbose=True):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
        self.best_epoch = 0

    def __call__(self, val_metric, model, epoch):
        score = val_metric

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = deepcopy(model.state_dict())
            self.best_epoch = epoch
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = deepcopy(model.state_dict())
            self.best_epoch = epoch
            self.counter = 0


def main():
    # 设置随机种子
    set_seed(42)

    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载数据
    # 实际实现中请替换为您的数据加载代码
    data = load_list_from_file("../data/image_padding/result_list.json")
    num_classes = 2

    dataset = CustomDataset(data, num_classes)
    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    from torch.utils.data.dataset import random_split
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset,
                              batch_size=8,
                              shuffle=True,
                              num_workers=4,
                              prefetch_factor=2,
                              persistent_workers=True)

    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False)



    print(f"Dataset sizes: Train={len(train_dataset)}, Test={len(test_dataset)}")

    # 创建模型
    ianet = IANet().to(device)
    resnet_baseline = ResNet18Baseline().to(device)

    # 创建结果目录
    os.makedirs("results", exist_ok=True)

    # 训练参数
    num_epochs = 100
    learning_rate = 0.0001

    # 初始化模型、损失函数和优化器
    models = {
        "IANet": ianet,
        "ResNet18Baseline": resnet_baseline
    }

    optimizers = {
        "IANet": optim.Adam(ianet.parameters(), lr=learning_rate),
        "ResNet18Baseline": optim.Adam(resnet_baseline.parameters(), lr=learning_rate)
    }

    criterion = nn.CrossEntropyLoss()

    # 早停机制
    early_stoppings = {
        "IANet": EarlyStopping(patience=15, delta=0.001, verbose=True),
        "ResNet18Baseline": EarlyStopping(patience=15, delta=0.001, verbose=True)
    }

    # 存储结果
    results = []

    # 训练循环
    for epoch in range(num_epochs):
        epoch_results = {"epoch": epoch + 1}
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'=' * 50}")

        # 训练两个模型
        for model_name in models:
            model = models[model_name]
            optimizer = optimizers[model_name]
            model.train()

            running_loss = 0.0
            train_correct = 0
            train_total = 0

            pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                        desc=f"Training {model_name}")

            for i, (inputs, labels) in pbar:
                a = inputs[0].to(device)
                b = inputs[1].to(device)
                labels = labels.to(device)

                # 准备输入
                if model_name == "ResNet18Baseline":
                    outputs = model(a, b)
                else:
                    outputs = model(a, b)

                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

                # 更新进度条
                current_loss = running_loss / (i + 1)
                current_acc = train_correct / train_total
                pbar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.4f}")

            # 计算训练指标
            train_loss = running_loss / len(train_loader)
            train_accuracy = train_correct / train_total

            # 存储训练结果
            epoch_results[f"{model_name}_train_loss"] = train_loss
            epoch_results[f"{model_name}_train_acc"] = train_accuracy

        # 在测试集上评估两个模型
        print("\nEvaluating models on test set...")
        for model_name in models:
            model = models[model_name]

            # 评估模型
            test_outputs, test_labels = evaluate_model(model, test_loader, device)
            test_accuracy, test_auc, test_tpr, test_fpr = calculate_metrics(test_outputs, test_labels)

            # 打印详细结果
            print(f"\n{model_name} Test Results (Epoch {epoch + 1}):")
            print(f"  Accuracy: {test_accuracy:.4f}")
            print(f"  AUC: {test_auc:.4f}")
            print(f"  TPR: {test_tpr:.4f}")
            print(f"  FPR: {test_fpr:.4f}")

            # 存储测试结果
            epoch_results[f"{model_name}_test_acc"] = test_accuracy
            epoch_results[f"{model_name}_test_auc"] = test_auc
            epoch_results[f"{model_name}_test_tpr"] = test_tpr
            epoch_results[f"{model_name}_test_fpr"] = test_fpr

            # 使用测试集作为验证集进行早停（根据您的要求）
            early_stoppings[model_name](test_auc, model, epoch)

            # 保存最佳模型
            if test_auc == early_stoppings[model_name].best_score:
                torch.save(model.state_dict(), f"results/{model_name}_best_model.pth")
                print(f"✅ Saved best {model_name} model (AUC: {test_auc:.4f})")

        # 保存本轮结果
        results.append(epoch_results)

        # 检查早停
        stop_training = True
        for model_name in models:
            if not early_stoppings[model_name].early_stop:
                stop_training = False
                break

        if stop_training:
            print("⚠️ Early stopping triggered for all models. Stopping training.")
            break

    # 保存最终模型和结果
    for model_name in models:
        torch.save(models[model_name].state_dict(), f"results/{model_name}_final_model.pth")

    # # 保存结果到CSV
    # df = pd.DataFrame(results)
    # df.to_csv("results/training_results.csv", index=False)
    #
    # # 绘制结果图表
    # plt.figure(figsize=(15, 10))
    #
    # # 准确率对比
    # plt.subplot(2, 2, 1)
    # plt.plot(df['epoch'], df['IANet_test_acc'], 'b-', label='IANet Accuracy')
    # plt.plot(df['epoch'], df['ResNet18Baseline_test_acc'], 'r-', label='ResNet18 Accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.title('Test Accuracy Comparison')
    # plt.legend()
    # plt.grid(True)
    #
    # # AUC对比
    # plt.subplot(2, 2, 2)
    # plt.plot(df['epoch'], df['IANet_test_auc'], 'b-', label='IANet AUC')
    # plt.plot(df['epoch'], df['ResNet18Baseline_test_auc'], 'r-', label='ResNet18 AUC')
    # plt.xlabel('Epoch')
    # plt.ylabel('AUC')
    # plt.title('Test AUC Comparison')
    # plt.legend()
    # plt.grid(True)
    #
    # # TPR对比
    # plt.subplot(2, 2, 3)
    # plt.plot(df['epoch'], df['IANet_test_tpr'], 'b-', label='IANet TPR')
    # plt.plot(df['epoch'], df['ResNet18Baseline_test_tpr'], 'r-', label='ResNet18 TPR')
    # plt.xlabel('Epoch')
    # plt.ylabel('TPR')
    # plt.title('True Positive Rate Comparison')
    # plt.legend()
    # plt.grid(True)
    #
    # # FPR对比
    # plt.subplot(2, 2, 4)
    # plt.plot(df['epoch'], df['IANet_test_fpr'], 'b-', label='IANet FPR')
    # plt.plot(df['epoch'], df['ResNet18Baseline_test_fpr'], 'r-', label='ResNet18 FPR')
    # plt.xlabel('Epoch')
    # plt.ylabel('FPR')
    # plt.title('False Positive Rate Comparison')
    # plt.legend()
    # plt.grid(True)
    #
    # plt.tight_layout()
    # plt.savefig('results/performance_comparison.png')
    # plt.close()
    #
    # print("Training completed. Results saved to 'results/' directory.")
    # print(f"Final model performance comparison:")
    # print(f"IANet: Accuracy={df['IANet_test_acc'].iloc[-1]:.4f}, AUC={df['IANet_test_auc'].iloc[-1]:.4f}")
    # print(
    #     f"ResNet18: Accuracy={df['ResNet18Baseline_test_acc'].iloc[-1]:.4f}, AUC={df['ResNet18Baseline_test_auc'].iloc[-1]:.4f}")


if __name__ == "__main__":
    main()