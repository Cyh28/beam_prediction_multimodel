# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import GeoDataset
from model import GeoMLP
from utils import evaluate


# ====== 参数 ======
CSV_PATH = "datasets/scenario32/scenario32_processed.csv"
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
NUM_CLASSES = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ====== 数据 ======
dataset = GeoDataset(CSV_PATH)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ====== 模型 ======
model = GeoMLP(num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# ====== 训练 ======
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for x, y in train_loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    top1, top3, top5 = evaluate(model, val_loader, DEVICE)

    print(f"Epoch [{epoch+1}/{EPOCHS}]")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Val Top1: {top1:.4f} | Top3: {top3:.4f} | Top5: {top5:.4f}")
    print("-" * 40)