import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import IPRDataset, collate_fn, evaluate_model
from models import DualGraphTransformerIPR

NUM_CLASSES = 6
BATCH_SIZE = 8
LR = 1e-4
EPOCHS = 50
HIDDEN_SIZE = 256
N_STEPS = 3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = IPRDataset(num_samples=50, num_features=128, num_classes=NUM_CLASSES)
test_dataset = IPRDataset(num_samples=10, num_features=128, num_classes=NUM_CLASSES)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

model = DualGraphTransformerIPR(in_feats=128, hidden_size=HIDDEN_SIZE, n_steps=N_STEPS, n_classes=NUM_CLASSES).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for pair_graphs, scene_graphs, labels in train_loader:
        pair_graphs, scene_graphs, labels = pair_graphs.to(DEVICE), scene_graphs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(pair_graphs, scene_graphs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    acc, per_class_acc, mAP = evaluate_model(model, test_loader, DEVICE, NUM_CLASSES)
    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss/len(train_loader):.4f} | Acc: {acc:.4f} | mAP: {mAP:.4f}")
    print(f"Per-class Acc: {per_class_acc}")
