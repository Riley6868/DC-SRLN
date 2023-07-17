import uuut as models
import torch
import torch.optim as optim
import uuut as models
# from data_loader import get_test_loader, get_train_loader
from configfa import get_config
from uty import accuracy, AverageMeter, loader_model, load_GPUS
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

class Net(nn.Module):
    def __init__(self, seresnet50, resnet50):
        super(Net, self).__init__()
        self.conv1 = list(seresnet50.pretrained_model.children())[0]
        self.bn1 = list(seresnet50.pretrained_model.children())[1]
        self.relu = list(seresnet50.pretrained_model.children())[2]
        self.maxpool = list(seresnet50.pretrained_model.children())[3]
        self.layer1 = list(seresnet50.pretrained_model.children())[4]
        self.layer2 = list(seresnet50.pretrained_model.children())[5]
        self.layer3 = list(seresnet50.pretrained_model.children())[6]
        self.layer4 = list(seresnet50.pretrained_model.children())[7]
        self.avgpool = list(seresnet50.pretrained_model.children())[8]

        self.conv1a = list(resnet50.pretrained_model.children())[0]
        self.bn1a = list(resnet50.pretrained_model.children())[1]
        self.relua = list(resnet50.pretrained_model.children())[2]
        self.maxpoola = list(resnet50.pretrained_model.children())[3]
        self.layer1a = list(resnet50.pretrained_model.children())[4]
        self.layer2a = list(resnet50.pretrained_model.children())[5]
        self.layer3a = list(resnet50.pretrained_model.children())[6]
        self.layer4a = list(resnet50.pretrained_model.children())[7]
        self.avgpoola = list(resnet50.pretrained_model.children())[8]

        self.dropout4 = nn.Dropout(0.2)
        self.dropout8 = nn.Dropout(0.2)
        self.fc6 = nn.Linear(4096, 2048)
        self.fc10 = nn.Linear(2048, 2)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)


        x1 = self.conv1a(input)
        x1 = self.bn1a(x1)
        x1 = self.relua(x1)
        x1 = self.maxpoola(x1)
        x1 = self.layer1a(x1)
        x1 = self.layer2a(x1)
        x1 = self.layer3a(x1)
        x1 = self.layer4a(x1)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)*0.5295
        x1 = self.avgpoola(x1)
        x1 = x1.reshape(x1.size(0), -1)*0.4705

        output = torch.cat((x, x1), 1)
        output = self.dropout4(output)
        output = self.fc6(output)
        output = self.dropout8(output)
        output = self.fc10(output)
        return output

def getModel():
    path1 = "/DATA/HJY/NTS-Net-master/data_4t/yangz/models/SE50_20221123_201553/seresnet50_082.ckpt"
    path2 = "/DATA/HJY/NTS-Net-master/data_4t/yangz/models/resnet50_20221123_193715/resnet50_071.ckpt"
    seres50 = loader_model(2, "se_resnet50", path1)
    res50 = loader_model(2, "resnet50", path2)

    for index, p in enumerate(seres50.parameters()):
        p.requires_grad = False
    for index, p in enumerate(res50.parameters()):
        p.requires_grad = False
    model = Net(seres50, res50)

    return model
getModel()
