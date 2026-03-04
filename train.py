import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import FusionDataset
from model import FusionModel
from utils import evaluate


CSV_PATH = "datasets/scenario33/scenario33_processed.csv"
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4
NUM_CLASSES = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


dataset = FusionDataset(CSV_PATH)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


model = FusionModel(NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for image, geo, label in train_loader:
        image = image.to(DEVICE)
        geo = geo.to(DEVICE)
        label = label.to(DEVICE)

        optimizer.zero_grad()
        output = model(image, geo)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    top1, top3, top5 = evaluate(model, val_loader, DEVICE)

    print(f"Epoch {epoch+1}")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Val Top1: {top1:.4f} | Top3: {top3:.4f} | Top5: {top5:.4f}")
    print("-"*40)