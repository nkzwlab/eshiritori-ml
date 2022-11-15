from torch import nn
import torchvision.models as models

model = nn.Sequential(
    nn.Conv2d(1, 64, 3, padding='same'),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(64, 128, 3, padding='same'),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(128, 256, 3, padding='same'),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(2304, 512),
    nn.ReLU(),
    nn.Linear(512, 345), # output classes = 345
)

class Net(nn.Module):
    def __init__(self, num_classes=345, pretrained=False):
        super(Net, self).__init__()        
        model = models.resnet34(pretrained=pretrained)

        model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=model.conv1.kernel_size,
            stride=model.conv1.stride,
            padding=model.conv1.padding,
            bias=False
        )
        
        model.fc = nn.Linear(
            in_features=model.fc.in_features,
            out_features=num_classes
        )
        
        self.resnet = model
        
    def forward(self, images):
        return self.resnet(images)
    
if __name__ == '__main__':
    net = Net()
    print(net)