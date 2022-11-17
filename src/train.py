import torch
from dataset import QuickDrawDataset, get_loader
from tqdm import tqdm

def train_loop(
    net,
    data_loader,
    device,
    criterion,
    optimizer,
):

    net.train()

    accum_loss = 0
    
    for i, (images, labels) in enumerate(data_loader):
        # print(labels)
        optimizer.zero_grad()
        
        images, labels = images.to(device), labels.to(device)
        outputs = net.forward(images)
        
        loss = criterion(outputs, labels)
        accum_loss += loss.item()
        # print(loss)

        
        loss.backward()
        optimizer.step()

    return accum_loss / len(data_loader)
    