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

    correct = 0
    total = 0
    for i, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = net.forward(images)
        
        loss = criterion(outputs, labels)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        acc = 100 * correct / total

        print(loss)
        print(f'Accuracy of the network on the test images: {acc:.2f}%')
        return acc 

# def test(network, testloader):
#   """
#   simple test. network weights are stored in `save_path` and the function itself will return the trained network.
#   """
#   print(f"testing...")
#   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#   net = network
#   net.to(device)
#   net.eval()
#   correct = 0
#   total = 0
#   with torch.no_grad():
#       for data in tqdm(testloader):
#           images, labels = data[0].to(device), data[1].to(device)
#           outputs = net(images)
#           _, predicted = torch.max(outputs.data, 1)
#           total += labels.size(0)
#           correct += (predicted == labels).sum().item()

#   print('Accuracy of the network on the test images: %d %%' % (
#       100 * correct / total))