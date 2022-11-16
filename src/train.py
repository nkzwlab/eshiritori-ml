import torch
from dataset import QuickDrawDataset, get_loader

def train_loop(
    net,
    data_loader,
    device,
    criterion,
    optimizer,
):
    print('***************************')
    print('start train')
    print('***************************')
    net.train()

    accum_loss = 0
    
    for i, (images, labels) in enumerate(data_loader):
        # print(labels)
        images, labels = images.to(device), labels.to(device)
        outputs = net.forward(images)
        
        loss = criterion(outputs, labels)
        accum_loss += loss.item()
        # print(loss)

        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return accum_loss / len(data_loader)
    