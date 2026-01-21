import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from province_head import ProvinceHead
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pipeline', 'data_streaming'))
from geo_dataset import GeopakDataset

def train():
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'merged_training_data_with_cells.csv')
    csv_path = os.path.abspath(csv_path)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    cache_dir = os.path.join(os.path.dirname(__file__), '..', 'cache', 'images')
    dataset = GeopakDataset(csv_path=csv_path, cache_dir=cache_dir, transform=transform)
    
    df = dataset.df
    province_counts = df['province_id'].value_counts().sort_index()
    total = len(df)
    weights = (total / province_counts).values
    weights = weights / weights.sum() * 7
    weights = torch.FloatTensor(weights).to(device)
    
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=False)
    
    model = ProvinceHead(fusion_dim=512, hidden_dim=256, freeze_clip=True, freeze_scene=True)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    num_epochs = 8
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            province_ids = batch['province_id'].to(device)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, province_ids)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += province_ids.size(0)
            correct += (predicted == province_ids).sum().item()
            
            if batch_idx % 100 == 0:
                acc = 100 * correct / total if total > 0 else 0
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}, Acc: {acc:.2f}%')
        
        epoch_acc = 100 * correct / total
        epoch_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
        
        print(f'Saving model...')
        torch.save(model.state_dict(), f'province_head_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    train()
