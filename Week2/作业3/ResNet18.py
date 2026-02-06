import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_c, out_c, stride, DownSample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        
        self.DownSample = DownSample
    
    def forward(self, x):
        res = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.DownSample:
            res = self.DownSample(x)
            
        out = self.relu(out + res)
        
        return out
    

class ResNet18(nn.Module):
    def __init__(self, block=BasicBlock, num_classes=10, channels=3):
        super(ResNet18, self).__init__()
        
        # 输入层，要先过一个7*7的卷积核
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # 池化层
        self.max_pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 4个残差层
        self.layer1 = self._make_layer(block, 64, 64, stride=1, num_blocks=2)
        self.layer2 = self._make_layer(block, 64, 128, stride=2, num_blocks=2)
        self.layer3 = self._make_layer(block, 128, 256, stride=2, num_blocks=2)
        self.layer4 = self._make_layer(block, 256, 512, stride=2, num_blocks=2)
        
        # 1个自适应平均池化,任意width*height尺寸输入，输出1*1
        self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        
        # 1个全连接层，将512通道映射到所需num_classes维
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # 权重初始化：卷积/BN层默认初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    
    def _make_layer(self, block, in_c, out_c, stride, num_blocks):
        downsample = None
        
        if stride != 1 or in_c != out_c * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_c, out_c * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c * block.expansion)
            )
        
        layers = []
        layers.append(block(in_c, out_c, stride=stride, DownSample=downsample))
        for _ in range(1, num_blocks):
            layers.append(block(out_c, out_c, stride=1))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.max_pooling(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avg_pooling(out)
        out = self.fc(torch.flatten(out, 1))
        
        return out
        
        