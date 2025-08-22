import torch.nn as nn

# Define ResNet18 Architecture
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False) # bias를 왜 False로 두는지?
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        '''
        self.shortcut = nn.Sequential()은 ResNet의 Skip Connection을 구현.
        아무런 연산 없이(nn.Sequential()) x를 통과시켜 그대로 x가 출력됨.
        
        if는 out += self.shortcut(x) 연산 때문에 존재
        out과 x의 모양이 다르면 + 연산을 할 수 없기 때문에 모양을 맞춰주기 위한 작업.
        조건 1. stride != 1 이면, feature map의 크기가 줄어듦 -> ex) stride=2, 32x32 -> 16x16
        조건 2. in_channels != self.expansion * out_channels이면, 블록을 거치면서 채널 수가 바뀐것을 감지하는 조건 -> ex) 64채널 -> 128채널

        위 도 조건을 한 번에 해결할 수 있도록 if 안의 연산을 진행.
        조건1 -> stride=stride를 통해 downsampling
        조건2 -> in_channels, self.expansion * out_channels를 통해 Channel Projection
        '''
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                                          nn.BatchNorm2d(self.expansion * out_channels))
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)

        out = self.relu(out)
        return out
    
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[0], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[0], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[0], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        
        '''
        _make_layer가 받은 block은 class BasicBlock
        아래 for loop은 layer당 2개의 block을 생성.

        block은 conv1 - bn1 - relu - conv2 - bn2 - skip connection - relu로 구성

        block(self.in_channels, out_channels, s)
        self.in_channels = out_channels * block.expansion을 통해 이전 레이어의 out_channels를 self.in_channels에 저장
        다음 block을 만들 때 in_channels로 정보 전달

        예시
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        * 첫 번째 블록: BasicBlock(64, 64, 1) 생성 -> self.in_channels는 64로 업데이트.
        * 두 번째 블록: BasicBlock(64, 64, 1) 생성 -> self.in_channels는 64로 다시 업데이트.
        * layer1 생성이 끝나면, self.in_channels는 여전히 64

        self.layer2 = self._make_layer(block, 128, num_block[2], stride=2)
        * layers.append(block(self.in_channels, out_channels, s))에서 남아 있는 self.in_channels가 64로 호출
        * 이제 self.in_channels = out_channels * block.expansion을 통해 128 저장
        '''

        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)

        out = out.view(out.size(0), -1)

        '''
        fully-connected layer에 넣기 위해 텐서 모양을 바꿔주는 코드.
        Flatten 과정이라고 함.

        self.avgpool(out) 이후 out의 모양은 4차원 텐서.
        (batch, 512, 1, 1)의 모양을 가지고 있을 것임.
        out.size(0): batch의 모양은 그대로 유지
        -1: 512 * 1 * 1을 한 줄로 쭉 펼침

        결과 텐서: (batch, 512)

        linear 레이어는 2차원 텐서 (배치 크기, 입력 특성 수)만 입력으로 받을 수 있기 때문에
        이렇게 해서 linear 레이어에 넣어야 함.
        '''

        out = self.linear(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)