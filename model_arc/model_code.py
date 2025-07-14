import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import time
import matplotlib.pyplot as plt


class SpatialChannelAttention(nn.Module):
    """空间和通道注意力模块"""

    def __init__(self, in_channels, reduction_ratio=16):
        super(SpatialChannelAttention, self).__init__()

        # 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )

        # 空间注意力
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        # 通道注意力
        avg_out = self.channel_fc(self.avg_pool(x))
        max_out = self.channel_fc(self.max_pool(x))
        channel_out = torch.sigmoid(avg_out + max_out)

        # 空间注意力
        avg_out_spatial = torch.mean(x, dim=1, keepdim=True)
        max_out_spatial, _ = torch.max(x, dim=1, keepdim=True)
        spatial_out = torch.sigmoid(self.spatial_conv(torch.cat([avg_out_spatial, max_out_spatial], dim=1)))

        # 融合注意力
        return x * channel_out * spatial_out


class IndividualBranch(nn.Module):
    """独立特征分支模块"""

    def __init__(self, in_dim=512):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // 8, 1),
            nn.ReLU(),
            nn.Conv2d(in_dim // 8, in_dim, 1),
            nn.Sigmoid()
        )
        self.projection = nn.Conv2d(in_dim, 256, 1)

    def forward(self, x):
        attn = self.attention(x)  # 通道注意力
        return self.projection(x * attn)  # 降维保留关键特征


class CrossInteraction(nn.Module):
    """跨肾脏交互模块"""

    def __init__(self):
        super().__init__()
        self.cross_att = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)

    def forward(self, left, right):
        b, c, h, w = left.shape

        # 维度转换：(B, C, H, W) → (B, H*W, C)
        left_flat = left.reshape(b, c, -1).permute(0, 2, 1)
        right_flat = right.reshape(b, c, -1).permute(0, 2, 1)

        # 交叉注意力
        cross_left, _ = self.cross_att(left_flat, right_flat, right_flat)
        cross_right, _ = self.cross_att(right_flat, left_flat, left_flat)

        # 恢复维度：(B, H*W, C) → (B, C, H, W)
        left_cross = cross_left.permute(0, 2, 1).reshape(b, c, h, w)
        right_cross = cross_right.permute(0, 2, 1).reshape(b, c, h, w)

        return left_cross, right_cross


class EnhancedDiff(nn.Module):
    """增强差异模块"""

    def __init__(self):
        super().__init__()
        self.diff_conv = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
            SpatialChannelAttention(512)
        )

    def forward(self, left, right):
        diff = torch.abs(left - right)
        return self.diff_conv(diff)


class IANet(nn.Module):
    """IANet主模型"""

    def __init__(self):
        super().__init__()
        # 共享ResNet主干
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )

        # 特征处理分支
        self.left_branch = IndividualBranch()
        self.right_branch = IndividualBranch()
        self.cross_interaction = CrossInteraction()
        self.diff_module = EnhancedDiff()

        # 多维度融合分类
        self.fusion = nn.Sequential(
            nn.Linear(256 * 2 + 512, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(0.3))
        self.classifier = nn.Linear(512, 2)

    def forward(self, left, right):
        # 基础特征提取
        f_left = self.feature_extractor(left)  # [B,512,31,50]
        f_right = self.feature_extractor(right)

        # 独立特征保留
        ind_left = self.left_branch(f_left)  # [B,256,31,50]
        ind_right = self.right_branch(f_right)

        # 交叉特征交互
        cross_left, cross_right = self.cross_interaction(ind_left, ind_right)

        # 差异特征增强
        diff_feat = self.diff_module(f_left, f_right)  # [B,512,31,50]

        # 多特征融合
        ind_feat = torch.cat([
            cross_left.mean(dim=[2, 3]),
            cross_right.mean(dim=[2, 3])
        ], dim=1)  # [B,512]

        diff_feat = diff_feat.mean(dim=[2, 3])  # [B,512]

        fused = self.fusion(torch.cat([ind_feat, diff_feat], dim=1))
        return self.classifier(fused)

class ResNet18Baseline(nn.Module):
    """ResNet18基线模型（修改为接受6通道输入）"""

    def __init__(self):
        super().__init__()
        # 修改第一层卷积接受6通道输入
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        resnet.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.model = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )
        self.fc = nn.Linear(512, 2)

    def forward(self, left, right):
        # 合并左右肾脏图像 (通道维度拼接)
        combined = torch.cat([left, right], dim=1)
        features = self.model(combined)
        features = torch.flatten(features, 1)
        return self.fc(features)


def generate_kidney_image(batch_size=1):
    """
    生成模拟肾脏超声图像
    返回形状: [batch_size, 3, 970, 1590]
    """
    # 基础图像 - 低强度背景
    image = torch.rand(batch_size, 3, 970, 1590) * 0.3

    # 添加肾脏轮廓 - 椭圆形状
    for b in range(batch_size):
        # 随机生成肾脏位置
        center_x = torch.randint(300, 1300, (1,))
        center_y = torch.randint(300, 700, (1,))

        # 生成椭圆蒙版
        y, x = torch.meshgrid(torch.arange(970), torch.arange(1590))
        ellipse = ((x - center_x) / 600) ** 2 + ((y - center_y) / 300) ** 2 < 1

        # 添加肾脏区域（稍高强度）
        image[b, :, ellipse] += 0.4

        # 添加内部结构（肾盂）
        inner_ellipse = ((x - center_x) / 200) ** 2 + ((y - center_y) / 100) ** 2 < 1
        image[b, :, inner_ellipse] -= 0.2

        # 添加噪声（模拟超声纹理）
        image[b] += torch.randn(3, 970, 1590) * 0.05

    # 裁剪到[0,1]范围
    return torch.clamp(image, 0, 1)


def visualize_kidney_image(image, title="Simulated Kidney Ultrasound"):
    """可视化生成的肾脏图像"""
    # 转换为HWC格式并调整通道顺序
    img_np = image[0].permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(12, 6))
    plt.imshow(img_np, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


def test_models():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 生成模拟数据
    print("Generating simulated kidney images...")
    left_kidney = generate_kidney_image().to(device)
    right_kidney = generate_kidney_image().to(device)

    print(f"Left kidney shape: {left_kidney.shape}")
    print(f"Right kidney shape: {right_kidney.shape}")

    # 可视化样本
    visualize_kidney_image(left_kidney.cpu(), "Left Kidney")
    visualize_kidney_image(right_kidney.cpu(), "Right Kidney")

    # 初始化模型
    print("\nInitializing models...")
    ianet = IANet().to(device)
    resnet_baseline = ResNet18Baseline().to(device)

    # 打印模型参数数量
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"IANet parameters: {count_parameters(ianet):,}")
    print(f"ResNet18Baseline parameters: {count_parameters(resnet_baseline):,}")

    # 测试IANet前向传播
    print("\nTesting IANet forward pass...")
    start_time = time.time()
    ianet_output = ianet(left_kidney, right_kidney)
    ianet_time = time.time() - start_time
    print(f"IANet output shape: {ianet_output.shape}")
    print(f"IANet forward pass time: {ianet_time:.4f} seconds")

    # 测试ResNet18Baseline前向传播
    print("\nTesting ResNet18Baseline forward pass...")
    start_time = time.time()
    resnet_output = resnet_baseline(left_kidney, right_kidney)
    resnet_time = time.time() - start_time
    print(f"ResNet18Baseline output shape: {resnet_output.shape}")
    print(f"ResNet18Baseline forward pass time: {resnet_time:.4f} seconds")

    # 检查输出
    print("\nOutput details:")
    print(f"IANet output: {ianet_output.detach().cpu()}")
    print(f"ResNet18Baseline output: {resnet_output.detach().cpu()}")

    # 显存使用情况
    if torch.cuda.is_available():
        print(f"\nGPU memory usage:")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        print(f"Reserved: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")


if __name__ == "__main__":
    test_models()