import os
os.environ["SM_FRAMEWORK"] = "torch"
os.environ["SMP_DISABLE_VERSION_CHECK"] = "1"  # 禁用版本检查
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# ====================== 配置参数（完全和论文一致） ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# 改成你的 LMHLD_dataset_different_patch_sizes 文件夹路径
DATA_ROOT = "F:\LMHLD\LMHLD_dataset_different_patch_sizes"
# 先选这个区域训练（patch size 32，训练最快）
TRAIN_REGION = "Emilia-Romagna_Italy_32"
BATCH_SIZE = 16  # 显存不够改成 8 或 4
NUM_EPOCHS = 100  # 论文用了 300，先跑 100 看效果
LEARNING_RATE = 0.01
IMAGE_SIZE = 32  # 对应所选区域的 patch size
NUM_CHANNELS = 4  # 蓝、绿、红、近红外
NUM_CLASSES = 1  # 二分类

# ====================== 适配 npy 格式的数据集类 ======================
class LMHLDNpyDataset(Dataset):
    def __init__(self, root_dir, region, split="train", transform=None):
        self.transform = transform
        # 加载官方打包好的 npy 文件
        self.images = np.load(os.path.join(root_dir, region, f"{split}_images.npy"))
        self.labels = np.load(os.path.join(root_dir, region, f"{split}_labels.npy"))

        # 确保标签是 0/1 二值
        self.labels[self.labels > 0] = 1
        print(f"加载 {region} {split} 集：{len(self.images)} 个样本")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1. 取出 numpy 数组
        image = self.images[idx]
        mask = self.labels[idx]

        # 【核心修复1】如果原数据是 (4, 32, 32) [CHW]，需要转成 (32, 32, 4) [HWC] 才能做增强
        if image.ndim == 3 and image.shape[0] == 4:
            image = np.transpose(image, (1, 2, 0))  # C,H,W -> H,W,C

        # 如果标签是 (1, 32, 32)，压缩成 (32, 32)
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]

        # 2. 图像增强（Albumentations 需要输入 HWC 格式）
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # 3. 复制以消除增强翻转带来的负步长 (negative strides) 问题
        image = image.copy()
        mask = mask.copy()

        # 【核心修复2】现在 image 是 HWC (32, 32, 4)，我们用 permute 转回 PyTorch 需要的 CHW (4, 32, 32)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()

        # 将 mask 增加通道维度变成 (1, 32, 32)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

        return image_tensor, mask_tensor

# ====================== 数据增强 ======================
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    # 去掉 ToTensorV2()
], is_check_shapes=False)

val_transform = A.Compose([
    # 去掉 ToTensorV2()
], is_check_shapes=False)

# ====================== 加载数据 ======================
train_dataset = LMHLDNpyDataset(DATA_ROOT, TRAIN_REGION, split="train", transform=train_transform)
val_dataset = LMHLDNpyDataset(DATA_ROOT, TRAIN_REGION, split="val", transform=val_transform)
test_dataset = LMHLDNpyDataset(DATA_ROOT, TRAIN_REGION, split="test", transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ====================== 模型、损失函数、优化器 ======================
model = smp.UnetPlusPlus(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=NUM_CHANNELS,
    classes=NUM_CLASSES,
).to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

# ====================== 训练和验证函数 ======================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, masks in tqdm(loader):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_masks = []

    with torch.no_grad():
        for images, masks in tqdm(loader):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item() * images.size(0)

            preds = torch.sigmoid(outputs) > 0.5
            all_preds.append(preds.cpu().numpy())
            all_masks.append(masks.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_masks = np.concatenate(all_masks)

    tp = np.sum((all_preds == 1) & (all_masks == 1))
    fp = np.sum((all_preds == 1) & (all_masks == 0))
    fn = np.sum((all_preds == 0) & (all_masks == 1))

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    blds = (precision * recall * f1) ** (1 / 3)  # 论文的 BLDS 指标

    return total_loss / len(loader.dataset), precision, recall, f1, blds

# ====================== 开始训练 ======================
best_blds = 0.0
train_losses = []
val_losses = []
val_blds = []

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
    val_loss, precision, recall, f1, blds = validate(model, val_loader, criterion, DEVICE)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_blds.append(blds)

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, BLDS: {blds:.4f}")

    # 按论文的 BLDS 指标保存最好的模型
    if blds > best_blds:
        best_blds = blds
        torch.save(model.state_dict(), f"best_model_{TRAIN_REGION}.pth")
        print(f"✅ 保存最好的模型！BLDS: {best_blds:.4f}")

    scheduler.step()

# ====================== 测试集评估 ======================
print("\n=== 最终测试集结果 ===")
# 【修改1】修复 torch.load 安全警告
model.load_state_dict(torch.load(f"best_model_{TRAIN_REGION}.pth", weights_only=True))
test_loss, test_precision, test_recall, test_f1, test_blds = validate(model, test_loader, criterion, DEVICE)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1: {test_f1:.4f}")
print(f"Test BLDS: {test_blds:.4f}")

# ====================== 可视化预测结果（【修改2】彻底修复卫星影像全黑） ======================
# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 取第5个样本（可以改数字看不同结果）
image, mask = test_dataset[5]
with torch.no_grad():
    pred = model(image.unsqueeze(0).to(DEVICE))
    pred = torch.sigmoid(pred) > 0.5
    pred = pred.squeeze().cpu().numpy()

plt.figure(figsize=(15, 5), dpi=300)  # 300dpi高清

# 1. 原始RGB影像（核心修复）
plt.subplot(1, 3, 1)
# LMHLD通道顺序：[B, G, R, NIR]，取[2,1,0]转成RGB
rgb = image[[2, 1, 0]].permute(1, 2, 0).numpy()
# 【关键】每个通道单独归一化到0-1（颜色最自然）
for i in range(3):
    channel = rgb[:, :, i]
    rgb[:, :, i] = (channel - channel.min()) / (channel.max() - channel.min())
plt.imshow(rgb)
plt.title("原始卫星影像 (RGB)", fontsize=14)
plt.axis('off')

# 2. 真实标签
plt.subplot(1, 3, 2)
plt.imshow(mask.squeeze(), cmap='gray')
plt.title("真实滑坡标签", fontsize=14)
plt.axis('off')

# 3. 模型预测
plt.subplot(1, 3, 3)
plt.imshow(pred, cmap='gray')
plt.title("模型预测结果", fontsize=14)
plt.axis('off')

plt.tight_layout()
# 保存高清图
plt.savefig(f"prediction_{TRAIN_REGION}_sample5.png", dpi=300, bbox_inches='tight')
plt.show()