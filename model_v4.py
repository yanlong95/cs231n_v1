import numpy as np
import torch.nn as nn
from torch.utils import model_zoo
import torch.nn.functional as F
from torchvision.models.resnet import model_urls
import sklearn.metrics as skm

"""
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER
"""
def conv3x3(in_planes, out_planes, stride=1, groups=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

"""
    Identity block
"""
class vgg16(nn.Module):
    def __init__(self, params, num_classes=8):
        super(fc5, self).__init__()
        self.fc1 = nn.Linear(256*256*3, 5000)
        self.fc2 = nn.Linear(5000, 5000)
        self.fc3 = nn.Linear(5000, num_classes)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=params.dropout_rate, inplace=True)

    def forward(self, x):
        x_out = x.view(x.size(0), -1)

        x_out = self.fc1(x_out)
        x_out = self.relu(x_out)
        # x_out = self.dropout(x_out)

        x_out = self.fc2(x_out)
        x_out = self.relu(x_out)
        # x_out = self.dropout(x_out)

        x_out = self.fc3(x_out)
        x_out = self.relu(x_out)
        # x_out = self.dropout(x_out)

        x_out = F.log_softmax(x_out, dim=1)

        return x_out


class vgg16(nn.Module):
    def __init__(self, params, num_classes=8):
        super(vgg16, self).__init__()
        self.conv11 = conv3x3(in_planes=3, out_planes=64, stride=1)
        self.conv12 = conv3x3(in_planes=64, out_planes=64, stride=1)

        self.conv21 = conv3x3(in_planes=64, out_planes=128, stride=1)
        self.conv22 = conv3x3(in_planes=128, out_planes=128, stride=1)

        self.conv31 = conv3x3(in_planes=128, out_planes=256, stride=1)
        self.conv32 = conv3x3(in_planes=256, out_planes=256, stride=1)
        self.conv33 = conv3x3(in_planes=256, out_planes=256, stride=1)

        self.conv41 = conv3x3(in_planes=256, out_planes=512, stride=1)
        self.conv42 = conv3x3(in_planes=512, out_planes=512, stride=1)
        self.conv43 = conv3x3(in_planes=512, out_planes=512, stride=1)

        self.conv51 = conv3x3(in_planes=512, out_planes=512, stride=1)
        self.conv52 = conv3x3(in_planes=512, out_planes=512, stride=1)
        self.conv53 = conv3x3(in_planes=512, out_planes=512, stride=1)

        self.fc1 = nn.Linear(7 * 7 * 512, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=params.dropout_rate, inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_out = self.conv11(x)
        x_out = self.relu(x_out)
        x_out = self.conv12(x_out)
        x_out = self.relu(x_out)
        x_out = self.maxpool(x_out)

        x_out = self.conv21(x_out)
        x_out = self.relu(x_out)
        x_out = self.conv22(x_out)
        x_out = self.relu(x_out)
        x_out = self.maxpool(x_out)

        x_out = self.conv31(x_out)
        x_out = self.relu(x_out)
        x_out = self.conv32(x_out)
        x_out = self.relu(x_out)
        x_out = self.conv33(x_out)
        x_out = self.relu(x_out)
        x_out = self.maxpool(x_out)

        x_out = self.conv41(x_out)
        x_out = self.relu(x_out)
        x_out = self.conv42(x_out)
        x_out = self.relu(x_out)
        x_out = self.conv43(x_out)
        x_out = self.relu(x_out)
        x_out = self.maxpool(x_out)

        x_out = self.conv51(x_out)
        x_out = self.relu(x_out)
        x_out = self.conv52(x_out)
        x_out = self.relu(x_out)
        x_out = self.conv53(x_out)
        x_out = self.relu(x_out)
        x_out = self.maxpool(x_out)

        x_out = x_out.view(x_out.size(0), -1)
        x_out = self.fc1(x_out)
        x_out = self.relu(x_out)
        # x_out = self.dropout(x_out)
        x_out = self.fc2(x_out)
        x_out = self.relu(x_out)
        # x_out = self.dropout(x_out)
        x_out = self.fc3(x_out)
        x_out = self.relu(x_out)
        # x_out = self.dropout(x_out)
        x_out = F.log_softmax(x_out, dim=1)

        return x_out

class Identity_block(nn.Module):
    channel_expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, groups=1, base_width=64):
        super(Identity_block, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        x_shortcut = x

        # first layer
        x_out = self.conv1(x)
        x_out = self.bn1(x_out)
        x_out = self.relu(x_out)

        # second layer
        x_out = self.conv2(x_out)
        x_out = self.bn2(x_out)

        if self.downsample is not None:
            x_shortcut = self.downsample(x)

        # add shortcut
        x_out += x_shortcut
        x_out = self.relu(x_out)

        return x_out


"""
    The convolution block
"""
class Conv_block(nn.Module):
    channel_expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, groups=1, base_width=64):
        super(Conv_block, self).__init__()

        width = int(out_channels * (base_width / 64.)) * groups

        self.conv1 = conv1x1(in_channels, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = conv3x3(width, width, stride, groups)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, out_channels * self.channel_expansion)
        self.bn3 = nn.BatchNorm2d(out_channels * self.channel_expansion)
        self.relu =nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        x_shortcut = x

        # first layer
        x_out = self.conv1(x)
        x_out = self.bn1(x_out)
        x_out = self.relu(x_out)

        # second later
        x_out = self.conv2(x_out)
        x_out = self.bn2(x_out)
        x_out = self.relu(x_out)

        # third layer
        x_out = self.conv3(x_out)
        x_out = self.bn3(x_out)

        if self.downsample is not None:
            x_shortcut = self.downsample(x)

        # add shortcut
        x_out += x_shortcut
        x_out = self.relu(x_out)

        return x_out

"""
The ResNet class, which group a batch of blocks together
"""
class ResNet(nn.Module):

    def __init__(self, params, block, stages, num_classes=8, zero_init_residual=False, groups=1, width_per_group=64):
        super(ResNet, self).__init__()

        self.in_channel = 64
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage1 = self._make_layer(block, 64, stages[0])
        self.stage2 = self._make_layer(block, 128, stages[1], stride=2)
        self.stage3 = self._make_layer(block, 256, stages[2], stride=2)
        self.stage4 = self._make_layer(block, 512, stages[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.channel_expansion, num_classes)
        self.dropout = nn.Dropout(p=params.dropout_rate, inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # No idea why, but may be delete later
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Conv_block):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, Identity_block):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, out_channel, blocks, stride=1, norm_layer=None):
        downsample = None
        if stride != 1 or self.in_channel != out_channel * block.channel_expansion:
            downsample = nn.Sequential(conv1x1(self.in_channel, out_channel * block.channel_expansion, stride),
                nn.BatchNorm2d(out_channel * block.channel_expansion), )

        layers = []
        layers.append(block(self.in_channel, out_channel, stride, downsample, self.groups, self.base_width, ))
        self.in_channel = out_channel * block.channel_expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channel, out_channel, groups=self.groups, base_width=self.base_width))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # x = self.dropout(x)
        x = F.log_softmax(x, dim=1)

        return x


def resnet18(params, num_classes, pretrained=False, **kwargs):
    model = ResNet(params, Identity_block, [2, 2, 2, 2], num_classes=num_classes, **kwargs)

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))

    return model


def resnet34(params, num_classes, pretrained=False, **kwargs):
    model = ResNet(params, Identity_block, [3, 4, 6, 3], num_classes=num_classes, **kwargs)

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))

    return model


"""
    create resnet50
"""
def resnet50(params, num_classes, pretrained=False, **kwargs):
    model = ResNet(params, Conv_block, [3, 4, 6, 3], num_classes, **kwargs)

    # used for bughole detection -- need to implement
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))

    return model


def resnet101(params, num_classes, pretrained=False, **kwargs):
    model = ResNet(params, Conv_block, [3, 4, 23, 3], num_classes, **kwargs)

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))

    return model


def resnet152(params, num_classes, pretrained=False, **kwargs):
    model = ResNet(params, Conv_block, [3, 8, 36, 3], num_classes, **kwargs)

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))

    return model

# def vgg16(params, num_classes, pretrained=False, **kwargs):



def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.

    Args:
        outputs: (np.ndarray) dimension batch_size x 6 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns: (float) accuracy in [0,1]
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs==labels) / float(labels.size)

def confusion_matrix(outputs, labels):
    outputs = np.argmax(outputs, axis=1)
    return skm.confusion_matrix(labels, outputs, labels=[0,1,2,3,4,5,6,7], normalize='true')

def classification_report(outputs, labels, output_dict, zero_division=0):
    outputs = np.argmax(outputs, axis=1)
    # f1 score = 2 * (precision * recall) / (precision + recall)
    # precision = tp / (tp + fp)
    # recall = tp / (tp + fn)
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    return skm.classification_report(labels, outputs, labels=[0,1,2,3,4,5,6,7], output_dict=output_dict)
# ['0','1','2','3','4','5','6','7']
    
# metrics for evaluation and result output
metrics = {
    'accuracy': accuracy,
           }









