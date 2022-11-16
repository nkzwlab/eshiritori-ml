import torch

def eval_loop(
    net,
    data_loader,
    device,
    criterion,
):
    print('***************************')
    print('start eval')
    print('***************************')
    net.eval()
    
    for i, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = net.forward(images)
        
        loss = criterion(outputs, labels)
        
        print(loss)