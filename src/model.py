from torch import nn
import torchvision.models as models

class Net(nn.Module):
    def __init__(self, num_classes=345, pretrained=False, rn="resnet50"):
        super(Net, self).__init__()   
        if rn == "resnet18":
            model = models.resnet18(pretrained=pretrained)
        elif rn == "resnet34":
            model = models.resnet34(pretrained=pretrained)     
        elif rn == "resnet50":
            model = models.resnet50(pretrained=pretrained)
        elif rn == "resnet101":
            model = models.resnet101(pretrained=pretrained)
        elif rn == "resnet152":
            model = models.resnet152(pretrained=pretrained)        
        else:
            raise ValueError("rn must be one of resnet18, resnet34, resnet50, resnet152, resnet101")

        # model.conv1 = nn.Conv2d(
        #     in_channels=1,
        #     out_channels=64,
        #     kernel_size=model.conv1.kernel_size, #7,7,
        #     stride=model.conv1.stride,
        #     padding=model.conv1.padding, #3,3,
        #     bias=False
        # )
        
        model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=3,#model.conv1.kernel_size 7,7,
            stride=1,#model.conv1.stride,
            padding=2,#model.conv1.padding 3,3,
            bias=False
        )
        
        model.fc = nn.Linear(
            in_features=model.fc.in_features,
            out_features=num_classes
        )
        
        self.resnet = model
        
    def forward(self, images):
        return self.resnet(images)


# keras code
    # model = Sequential()
    # model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(15, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.2))
    # model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dense(50, activation='relu'))
    # model.add(Dense(num_classes, activation='softmax'))

# pytorch code
    # model = nn.Sequential(
    #     nn.Conv2d(1, 30, kernel_size=5, stride=1, padding=2),
    #     nn.ReLU(),
    #     nn.MaxPool2d(kernel_size=2, stride=2),
    #     nn.Conv2d(30, 15, kernel_size=3, stride=1, padding=1),
    #     nn.ReLU(),
    #     nn.MaxPool2d(kernel_size=2, stride=2),
    #     nn.Dropout2d(0.2),
    #     nn.Flatten(),
    #     nn.Linear(15*7*7, 128),
    #     nn.ReLU(),
    #     nn.Linear(128, 50),
    #     nn.ReLU(),
    #     nn.Linear(50, num_classes),
    #     nn.Softmax(dim=1)
    # )

class CNN(nn.Module):
    def __init__(self, num_classes=345):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 30, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(30, 15, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(0.2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(15*56*56, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 50)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(50, num_classes)
        # self.softmax = nn.Softmax(dim=1)
    
    def forward(self, images):
        x = self.conv1(images)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.fc3(x)
        # x = self.softmax(x)
        return x


if __name__ == '__main__':
    net = Net()
    print(net)
    print(net.resnet.conv1.kernel_size)
    print(net.resnet.conv1.stride)
    print(net.resnet.conv1.padding)