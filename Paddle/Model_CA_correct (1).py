import paddle
import paddle.nn as nn
import paddle.vision
from AttentionModule import *
from functools import partial
import paddle.nn.functional as F


class Block(nn.Layer):
    def __init__(self, inplanes, planes, kernel=3, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2D(inplanes, inplanes, kernel_size=3, stride=stride,
                               padding=1, bias_attr=False)
        if (kernel == 1):
            pad = 0
        else:
            pad = 1
        self.conv2 = nn.Conv2D(inplanes, planes, kernel_size=kernel, stride=stride,
                               padding=pad, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(inplanes)
        self.bn2 = nn.BatchNorm2D(planes)
        self.relu = nn.ReLU()

    def forward(self, input):
        conv1 = self.conv1(input)
        bn1 = self.relu(self.bn1(conv1))

        conv2 = self.conv2(bn1)
        out = self.relu(self.bn2(conv2))

        return out


class Resnet(nn.Layer):
    def __init__(self):
        super(Resnet, self).__init__()
        resnet = paddle.vision.resnet34(pretrained=True)
        # resnet.conv1.in_channels =6

        self.firstconv = resnet.conv1
        # self.firstconv = nn.Conv2D(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=3, bias_attr=False)
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        e0 = self.firstmaxpool(x)
        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        return [x, e1, e2, e3, e4]


# 11/1
class Semantic_Block(nn.Layer):
    def __init__(self, inplanes, planes, num_class):
        # backbone encoder
        super(Semantic_Block, self).__init__()
        self.relu = nn.ReLU()
        # self.CBAM_semantic_1 = BasicBlock_orgin(inplanes=inplanes, planes=planes, stride=1)
        # self.CBAM_semantic_2 = BasicBlock_orgin(inplanes=inplanes, planes=planes, stride=1)
        self.finalconv0 = nn.Conv2D(inplanes, planes, kernel_size=3, stride=1,
                                    padding=1, bias_attr=False)
        # self.finalconv1 = nn.Conv2D(planes, planes, kernel_size=3, stride=1,
        #                             padding=1, bias_attr=False)
        # self.finalconv2 = nn.Conv2D(planes, planes, kernel_size=3, stride=1,
        #                             padding=1, bias_attr=False)

        self.conv1d_img = nn.Conv2D(planes, num_class, kernel_size=1, stride=1,
                                    padding=0, bias_attr=False,
                             weight_attr=nn.initializer.KaimingUniform())

    def forward(self, feature, mask):
        output = feature * mask
        # output = self.relu(self.CBAM_semantic_1(output))
        # output = self.relu(self.CBAM_semantic_2(output))
        output = self.relu(self.finalconv0(output))
        # output = self.relu(self.finalconv1(output))
        # output = self.relu(self.finalconv2(output))
        output = self.conv1d_img(output)

        return output


# four layer output decoder
class Semantic_DeBlock(nn.Layer):
    def __init__(self, inplanes, planes, num_class):
        # backbone encoder
        super(Semantic_DeBlock, self).__init__()
        self.relu = nn.ReLU()

        self.finalconv1 = nn.Conv2D(inplanes, planes, kernel_size=3, stride=1,
                                    padding=1, bias_attr=False)
        # self.finalconv2 = nn.Conv2D(planes, planes, kernel_size=3, stride=1,
        #                              padding=1, bias_attr=False)
        # self.finalconv3 = nn.Conv2D(planes, planes, kernel_size=3, stride=1,
        #                             padding=1, bias_attr=False)

        self.conv1d_img = nn.Conv2D(planes, num_class, kernel_size=1, stride=1,
                                    padding=0, bias_attr=False,
                             weight_attr=nn.initializer.KaimingUniform())

    def forward(self, feature):
        output = feature
        output = self.relu(self.finalconv1(output))
        # output = self.relu(self.finalconv2(output))
        # output = self.relu(self.finalconv3(output))
        output = self.conv1d_img(output)

        return output

# The structure of PCFN
# proposed method
# mutli-task change detection
class PCF_Unet_CD(nn.Layer):
    def __init__(self,num_class_LCM = 5, num_class_semantic=14):
        # backbone encoder
        super(PCF_Unet_CD, self).__init__()
        self.resnet = Resnet()

        self.CA = ChangeAttention() #Feature Difference Enhancement (FDE)

        # 9/18 double u-net
        self.CBAM1_1 = BasicBlock(inplanes=2 * 512, planes=512, stride=1)
        self.CBAM1_2 = BasicBlock(inplanes=2 * 256 + 512, planes=512, stride=1)
        self.CBAM1_3 = BasicBlock(inplanes=2 * 128 + 512, planes=256, stride=1)
        self.CBAM1_4 = BasicBlock(inplanes=2 * 64 + 256, planes=128, stride=1)
        self.CBAM1_5 = BasicBlock(inplanes=2 * 64 + 128, planes=64, stride=1)

        self.CBAM2_1 = BasicBlock(inplanes=2 * 512, planes=512, stride=1)
        self.CBAM2_2 = BasicBlock(inplanes=2 * 256 + 512, planes=512, stride=1)
        self.CBAM2_3 = BasicBlock(inplanes=2 * 128 + 512, planes=256, stride=1)
        self.CBAM2_4 = BasicBlock(inplanes=2 * 64 + 256, planes=128, stride=1)
        self.CBAM2_5 = BasicBlock(inplanes=2 * 64 + 128, planes=64, stride=1)

        self.up = nn.Upsample(scale_factor=2)
        self.pool = nn.MaxPool2D(kernel_size=2)
        self.relu = nn.ReLU()

        self.CBAM_semantic_1 = Semantic_DeBlock(inplanes=64, planes=32, num_class=num_class_LCM)
        self.CBAM_semantic_2 = Semantic_DeBlock(inplanes=64, planes=32, num_class=num_class_LCM)

        self.finalconv0 = nn.Conv2D(128 , 32, kernel_size=3, stride=1,
                                    padding=1, bias_attr=False)
        self.bn1= nn.BatchNorm2D(32)
        self.finalconv1 = nn.Conv2D(32, 32, kernel_size=3, stride=1,
                                    padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(32)
        # self.finalconv2 = nn.Conv2D(64, 64, kernel_size=3, stride=1,
        #                             padding=1, bias_attr=False)

        self.semantic_from_to = Semantic_DeBlock(inplanes=32, planes=32, num_class=num_class_semantic)

    def forward(self, input1, input2):
        # 9/13 double u net
        [feat1_1, feat1_2, feat1_3, feat1_4, feat1_5] = self.resnet(input1)
        [feat2_1, feat2_2, feat2_3, feat2_4, feat2_5] = self.resnet(input2)

        cbam1_1 = self.CBAM1_1(paddle.concat([feat1_5, feat2_5], 1))
        cbam1_2 = self.CBAM1_2(paddle.concat([self.up(cbam1_1), feat1_4, feat2_4], 1))
        cbam1_3 = self.CBAM1_3(paddle.concat([self.up(cbam1_2), feat1_3, feat2_3], 1))
        cbam1_4 = self.CBAM1_4(paddle.concat([self.up(cbam1_3), feat1_2, feat2_2], 1))
        cbam1_5 = self.CBAM1_5(paddle.concat([self.up(cbam1_4), feat1_1, feat2_1], 1))

        cbam2_1 = self.CBAM2_1(paddle.concat([feat1_5, feat2_5], 1))
        cbam2_2 = self.CBAM2_2(paddle.concat([self.up(cbam2_1), feat1_4, feat2_4], 1))
        cbam2_3 = self.CBAM2_3(paddle.concat([self.up(cbam2_2), feat1_3, feat2_3], 1))
        cbam2_4 = self.CBAM2_4(paddle.concat([self.up(cbam2_3), feat1_2, feat2_2], 1))
        cbam2_5 = self.CBAM2_5(paddle.concat([self.up(cbam2_4), feat1_1, feat2_1], 1))

        # # semantic
        # cbam1 = self.conv1(cbam1_5)
        # cbam2 = self.conv2(cbam2_5)
        output1 = self.CBAM_semantic_1(self.up(cbam1_5))
        output2 = self.CBAM_semantic_2(self.up(cbam2_5))

        output = self.CA(cbam1_5,cbam2_5) # Feature Difference Enhancement (FDE)

        output = self.relu((self.bn1(self.finalconv0(output))))
        output = self.relu((self.bn2(self.finalconv1(output))))
        # output = self.relu(self.finalconv2(output))
        output = self.semantic_from_to(self.up(output))

        return output1, output2, output

# proposed method_32channel
# mutli-task change detection
class PCF_Unet_CD_32(nn.Layer):
    def __init__(self,num_class_LCM = 5, num_class_semantic=14):
        # backbone encoder
        super(PCF_Unet_CD_32, self).__init__()
        self.resnet = Resnet()
        # self.dblock = Dblock(512)

        self.CA = ChangeAttention()

        # 9/18 double u-net
        self.CBAM1_1 = BasicBlock(inplanes=2 * 512, planes=512, stride=1)
        self.CBAM1_2 = BasicBlock(inplanes=2 * 256 + 512, planes=256, stride=1)
        self.CBAM1_3 = BasicBlock(inplanes=2 * 128 + 256, planes=128, stride=1)
        self.CBAM1_4 = BasicBlock(inplanes=2 * 64 + 128, planes=32, stride=1)
        self.CBAM1_5 = BasicBlock(inplanes=2 * 64 + 32, planes=32, stride=1)

        self.CBAM2_1 = BasicBlock(inplanes=2 * 512, planes=512, stride=1)
        self.CBAM2_2 = BasicBlock(inplanes=2 * 256 + 512, planes=256, stride=1)
        self.CBAM2_3 = BasicBlock(inplanes=2 * 128 + 256, planes=128, stride=1)
        self.CBAM2_4 = BasicBlock(inplanes=2 * 64 + 128, planes=32, stride=1)
        self.CBAM2_5 = BasicBlock(inplanes=2 * 64 + 32, planes=32, stride=1)

        self.up = nn.Upsample(scale_factor=2)
        self.pool = nn.MaxPool2D(kernel_size=2)
        self.relu = nn.ReLU()

        self.CBAM_semantic_1 = Semantic_DeBlock(inplanes=32, planes=32, num_class=num_class_LCM)
        self.CBAM_semantic_2 = Semantic_DeBlock(inplanes=32, planes=32, num_class=num_class_LCM)

        self.finalconv0 = nn.Conv2D(64 , 32, kernel_size=3, stride=1,
                                    padding=1, bias_attr=False)
        self.bn1= nn.BatchNorm2D(32)
        self.finalconv1 = nn.Conv2D(32, 32, kernel_size=3, stride=1,
                                    padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(32)
        # self.finalconv2 = nn.Conv2D(64, 64, kernel_size=3, stride=1,
        #                             padding=1, bias_attr=False)

        self.semantic_from_to = Semantic_DeBlock(inplanes=32, planes=32, num_class=num_class_semantic)

    def forward(self, input1, input2):
        # 9/13 double u net
        [feat1_1, feat1_2, feat1_3, feat1_4, feat1_5] = self.resnet(input1)
        [feat2_1, feat2_2, feat2_3, feat2_4, feat2_5] = self.resnet(input2)

        cbam1_1 = self.CBAM1_1(paddle.concat([feat1_5, feat2_5], 1))
        cbam1_2 = self.CBAM1_2(paddle.concat([self.up(cbam1_1), feat1_4, feat2_4], 1))
        cbam1_3 = self.CBAM1_3(paddle.concat([self.up(cbam1_2), feat1_3, feat2_3], 1))
        cbam1_4 = self.CBAM1_4(paddle.concat([self.up(cbam1_3), feat1_2, feat2_2], 1))
        cbam1_5 = self.CBAM1_5(paddle.concat([self.up(cbam1_4), feat1_1, feat2_1], 1))

        cbam2_1 = self.CBAM2_1(paddle.concat([feat1_5, feat2_5], 1))
        cbam2_2 = self.CBAM2_2(paddle.concat([self.up(cbam2_1), feat1_4, feat2_4], 1))
        cbam2_3 = self.CBAM2_3(paddle.concat([self.up(cbam2_2), feat1_3, feat2_3], 1))
        cbam2_4 = self.CBAM2_4(paddle.concat([self.up(cbam2_3), feat1_2, feat2_2], 1))
        cbam2_5 = self.CBAM2_5(paddle.concat([self.up(cbam2_4), feat1_1, feat2_1], 1))

        # # semantic
        # cbam1 = self.conv1(cbam1_5)
        # cbam2 = self.conv2(cbam2_5)
        output1 = self.CBAM_semantic_1(self.up(cbam1_5))
        output2 = self.CBAM_semantic_2(self.up(cbam2_5))

        output = self.CA(cbam1_5,cbam2_5)

        output = self.relu((self.bn1(self.finalconv0(self.up(output)))))
        output = self.relu((self.bn2(self.finalconv1(output))))
        # output = self.relu(self.finalconv2(output))
        output = self.semantic_from_to(output)

        return output1, output2, output





# mutli-task change detection for visualization of FDE
class PCF_Unet_CD_visualize_FDE(nn.Layer):
    def __init__(self,num_class_LCM = 5, num_class_semantic=14):
        # backbone encoder
        super(PCF_Unet_CD_visualize_FDE, self).__init__()
        self.resnet = Resnet()
        # self.dblock = Dblock(512)

        self.CA = ChangeAttention_visualize_FDE()

        # 9/18 double u-net
        self.CBAM1_1 = BasicBlock(inplanes=2 * 512, planes=512, stride=1)
        self.CBAM1_2 = BasicBlock(inplanes=2 * 256 + 512, planes=512, stride=1)
        self.CBAM1_3 = BasicBlock(inplanes=2 * 128 + 512, planes=256, stride=1)
        self.CBAM1_4 = BasicBlock(inplanes=2 * 64 + 256, planes=128, stride=1)
        self.CBAM1_5 = BasicBlock(inplanes=2 * 64 + 128, planes=64, stride=1)

        self.CBAM2_1 = BasicBlock(inplanes=2 * 512, planes=512, stride=1)
        self.CBAM2_2 = BasicBlock(inplanes=2 * 256 + 512, planes=512, stride=1)
        self.CBAM2_3 = BasicBlock(inplanes=2 * 128 + 512, planes=256, stride=1)
        self.CBAM2_4 = BasicBlock(inplanes=2 * 64 + 256, planes=128, stride=1)
        self.CBAM2_5 = BasicBlock(inplanes=2 * 64 + 128, planes=64, stride=1)

        self.up = nn.Upsample(scale_factor=2)
        self.pool = nn.MaxPool2D(kernel_size=2)
        self.relu = nn.ReLU()

        self.CBAM_semantic_1 = Semantic_DeBlock(inplanes=64, planes=32, num_class=num_class_LCM)
        self.CBAM_semantic_2 = Semantic_DeBlock(inplanes=64, planes=32, num_class=num_class_LCM)

        self.finalconv0 = nn.Conv2D(128 , 32, kernel_size=3, stride=1,
                                    padding=1, bias_attr=False)
        self.bn1= nn.BatchNorm2D(32)
        self.finalconv1 = nn.Conv2D(32, 32, kernel_size=3, stride=1,
                                    padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(32)
        # self.finalconv2 = nn.Conv2D(64, 64, kernel_size=3, stride=1,
        #                             padding=1, bias_attr=False)

        self.semantic_from_to = Semantic_DeBlock(inplanes=32, planes=32, num_class=num_class_semantic)

    def forward(self, input1, input2):
        # 9/13 double u net
        [feat1_1, feat1_2, feat1_3, feat1_4, feat1_5] = self.resnet(input1)
        [feat2_1, feat2_2, feat2_3, feat2_4, feat2_5] = self.resnet(input2)

        cbam1_1 = self.CBAM1_1(paddle.concat([feat1_5, feat2_5], 1))
        cbam1_2 = self.CBAM1_2(paddle.concat([self.up(cbam1_1), feat1_4, feat2_4], 1))
        cbam1_3 = self.CBAM1_3(paddle.concat([self.up(cbam1_2), feat1_3, feat2_3], 1))
        cbam1_4 = self.CBAM1_4(paddle.concat([self.up(cbam1_3), feat1_2, feat2_2], 1))
        cbam1_5 = self.CBAM1_5(paddle.concat([self.up(cbam1_4), feat1_1, feat2_1], 1))

        cbam2_1 = self.CBAM2_1(paddle.concat([feat1_5, feat2_5], 1))
        cbam2_2 = self.CBAM2_2(paddle.concat([self.up(cbam2_1), feat1_4, feat2_4], 1))
        cbam2_3 = self.CBAM2_3(paddle.concat([self.up(cbam2_2), feat1_3, feat2_3], 1))
        cbam2_4 = self.CBAM2_4(paddle.concat([self.up(cbam2_3), feat1_2, feat2_2], 1))
        cbam2_5 = self.CBAM2_5(paddle.concat([self.up(cbam2_4), feat1_1, feat2_1], 1))

        # # semantic
        # cbam1 = self.conv1(cbam1_5)
        # # cbam2 = self.conv2(cbam2_5)
        # output1 = self.CBAM_semantic_1(self.up(cbam1_5))
        # output2 = self.CBAM_semantic_2(self.up(cbam2_5))

        output = self.CA(cbam1_5,cbam2_5)

        return output


# proposed method
# ablation study
class PCF_wo_attention(nn.Layer):
    def __init__(self,num_class_LCM = 5, num_class_semantic=14):
        # backbone encoder
        super(PCF_wo_attention, self).__init__()
        self.resnet = Resnet()
        # self.dblock = Dblock(512)

        # self.CA = ChangeAttention()

        # 9/18 double u-net
        self.CBAM1_1 = Block(inplanes=2 * 512, planes=512, stride=1)
        self.CBAM1_2 = Block(inplanes=2 * 256 + 512, planes=512, stride=1)
        self.CBAM1_3 = Block(inplanes=2 * 128 + 512, planes=256, stride=1)
        self.CBAM1_4 = Block(inplanes=2 * 64 + 256, planes=128, stride=1)
        self.CBAM1_5 = Block(inplanes=2 * 64 + 128, planes=32, stride=1)

        self.CBAM2_1 = Block(inplanes=2 * 512, planes=512, stride=1)
        self.CBAM2_2 = Block(inplanes=2 * 256 + 512, planes=512, stride=1)
        self.CBAM2_3 = Block(inplanes=2 * 128 + 512, planes=256, stride=1)
        self.CBAM2_4 = Block(inplanes=2 * 64 + 256, planes=128, stride=1)
        self.CBAM2_5 = Block(inplanes=2 * 64 + 128, planes=32, stride=1)

        self.up = nn.Upsample(scale_factor=2)
        self.pool = nn.MaxPool2D(kernel_size=2)
        self.relu = nn.ReLU()

        self.CBAM_semantic_1 = Semantic_DeBlock(inplanes=32, planes=32, num_class=num_class_LCM)
        self.CBAM_semantic_2 = Semantic_DeBlock(inplanes=32, planes=32, num_class=num_class_LCM)

        self.finalconv0 = nn.Conv2D(64 , 32, kernel_size=3, stride=1,
                                    padding=1, bias_attr=False)
        self.bn1= nn.BatchNorm2D(32)
        self.finalconv1 = nn.Conv2D(32, 32, kernel_size=3, stride=1,
                                    padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(32)
        # self.finalconv2 = nn.Conv2D(64, 64, kernel_size=3, stride=1,
        #                             padding=1, bias_attr=False)

        self.semantic_from_to = Semantic_DeBlock(inplanes=32, planes=32, num_class=num_class_semantic)

    def forward(self, input1, input2):
        # 9/13 double u net
        [feat1_1, feat1_2, feat1_3, feat1_4, feat1_5] = self.resnet(input1)
        [feat2_1, feat2_2, feat2_3, feat2_4, feat2_5] = self.resnet(input2)

        cbam1_1 = self.CBAM1_1(paddle.concat([feat1_5, feat2_5], 1))
        cbam1_2 = self.CBAM1_2(paddle.concat([self.up(cbam1_1), feat1_4, feat2_4], 1))
        cbam1_3 = self.CBAM1_3(paddle.concat([self.up(cbam1_2), feat1_3, feat2_3], 1))
        cbam1_4 = self.CBAM1_4(paddle.concat([self.up(cbam1_3), feat1_2, feat2_2], 1))
        cbam1_5 = self.CBAM1_5(paddle.concat([self.up(cbam1_4), feat1_1, feat2_1], 1))

        cbam2_1 = self.CBAM2_1(paddle.concat([feat1_5, feat2_5], 1))
        cbam2_2 = self.CBAM2_2(paddle.concat([self.up(cbam2_1), feat1_4, feat2_4], 1))
        cbam2_3 = self.CBAM2_3(paddle.concat([self.up(cbam2_2), feat1_3, feat2_3], 1))
        cbam2_4 = self.CBAM2_4(paddle.concat([self.up(cbam2_3), feat1_2, feat2_2], 1))
        cbam2_5 = self.CBAM2_5(paddle.concat([self.up(cbam2_4), feat1_1, feat2_1], 1))

        # semantic
        output1 = self.CBAM_semantic_1(self.up(cbam1_5))
        output2 = self.CBAM_semantic_2(self.up(cbam2_5))

        # output = self.CA(cbam1_5,cbam2_5)

        output = self.relu(self.bn1(self.finalconv0(paddle.concat([cbam1_5, cbam2_5],axis =1 ))))
        output = self.relu(self.bn2(self.finalconv1(output)))
        # output = self.relu(self.finalconv2(output))
        output = self.semantic_from_to(self.up(output))


        return output1, output2, output

