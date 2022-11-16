import torch
from dataset import QuickDrawDataset, get_loader
from train import train_loop
from eval import eval_loop
from model import Net
import torch.nn as nn

if __name__ == '__main__':
    data_dir = './data'
    max_examples_per_class = 100
    train_val_split_pct = .1
    lr = 0.01
    num_epochs = 10
    batch_size = 128
    shuffle = True
    num_workers = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = QuickDrawDataset(
        root=data_dir,
        max_items_per_class=max_examples_per_class,
        class_limit=None,
        is_download=False,
    )
    train_ds, val_ds = ds.split(train_val_split_pct)
    
    train_loader = get_loader(train_ds, batch_size, shuffle, num_workers)
    val_loader = get_loader(val_ds, batch_size, shuffle, num_workers)
    
    net = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    
    for epoch in range (1, num_epochs):
        print('==========================')
        print('epoch: ', epoch)
        print('==========================')
        train_loop(net, train_loader, device, criterion, optimizer)
        eval_loop(net, val_loader, device, criterion)