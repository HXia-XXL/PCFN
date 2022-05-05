import torch
import torch.nn as nn
import torchvision
from AttentionModule import *
from functools import partial
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=3, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes // 2, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        if (kernel == 1):
            pad = 0
        else:
            pad = 1
        self.conv2 = nn.Conv2d(in_planes // 2, out_planes, kernel_size=kernel, stride=stride,
                               padding=pad, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes // 2)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        conv1 = self.conv1(input)
        bn1 = self.relu(self.bn1(conv1))

        conv2 = self.conv2(bn1)
        out = self.relu(self.bn2(conv2))

        return out


class Out_Block(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=3, stride=1):
        super(Out_Block, self).__init__()

        if (kernel == 1):
            pad = 0
        else:
            pad = 1

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride,
                               padding=pad, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        conv1 = self.conv1(input)
        bn1 = self.bn1(conv1)

        return self.relu(bn1)

# Feature Difference Enhancement (FDE)
class FDE(nn.Module):
    def __init__(self, kernel_size=7):
        super(FDE, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x1,x2):
        feat = torch.cat([x1, x2], 1)
        x = torch.abs(x2-x1)
        
        avg_out = torch.mean(x, axis=1, keepdim=1)
        max_out = torch.max(x, axis=1, keepdim=1)
        
        x = torch.concat([avg_out, max_out], axis=1)
        x = self.conv1(x)
        x = self.sigmoid(x)
        out = feat*x + feat
        return out
    

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        model = torchvision.models.vgg16(pretrained=True)

        model.features[0].in_channels = 6

        self.relu = model.features[1]
        self.pool = model.features[4]
        self.conv1 = nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = model.features[2]
        self.conv3 = model.features[5]
        self.conv4 = model.features[7]
        self.conv5 = model.features[10]
        self.conv6 = model.features[12]
        self.conv7 = model.features[14]
        self.conv8 = model.features[17]
        self.conv9 = model.features[19]
        self.conv10 = model.features[21]
        self.conv11 = model.features[24]
        self.conv12 = model.features[26]
        self.conv13 = model.features[28]

    def forward(self, input):
        conv1 = self.relu(self.conv1(input))
        conv2 = self.relu(self.conv2(conv1))
        conv3 = self.relu(self.conv3(self.pool(conv2)))
        conv4 = self.relu(self.conv4(conv3))
        conv5 = self.relu(self.conv5(self.pool(conv4)))
        conv6 = self.relu(self.conv6(conv5))
        conv7 = self.relu(self.conv7(conv6))
        conv8 = self.relu(self.conv8(self.pool(conv7)))
        conv9 = self.relu(self.conv9(conv8))
        conv10 = self.relu(self.conv10(conv9))
        conv11 = self.relu(self.conv11(self.pool(conv10)))
        conv12 = self.relu(self.conv12(conv11))
        conv13 = self.relu(self.conv13(conv12))

        return [conv2, conv4, conv7, conv10, conv13]


nonlinearity = partial(F.relu, inplace=True)


class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
        resnet = torchvision.models.resnet34(pretrained=True)
        # resnet.conv1.in_channels =6

        self.firstconv = resnet.conv1
        # self.firstconv = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=3, bias=False)
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


class Dblock(nn.Module):
    def __init__(self, channel):
        super(Dblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        # self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        # dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out  # + dilate5_out
        return out

    
# PCFN pytorch version
class PCFN(nn.Module):
    def __init__(self):
        # backbone encoder
        super(PCFN, self).__init__(LCM_class_num, SCD_class_num)
        self.resnet = Resnet()

        # Siamese Decoders
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

        self.up = nn.UpsamplingNearest2d(scale_factor=2)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()
        
        # SFN
        self.finalconv1 = nn.Conv2d(128, 64, kernel_size=3, stride=1,
                                    padding=1, bias=False)
        self.finalconv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1,
                                    padding=1, bias=False)
        self.finalconv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1,
                                    padding=1, bias=False)
        
        self.conv1d_SCD = nn.Conv2d(64, SCD_class_num, kernel_size=1, stride=1,
                                     padding=0, bias=False)
        
        # LCM predict
        self.LCM_conv1 = nn.Conv2d(64, 32, kernel_size=3, stride=1,
                                    padding=1, bias=False)
        self.LCM_conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=1,
                                    padding=1, bias=False)
        self.conv1d_LCM1 = nn.Conv2d(32, LCM_class_num, kernel_size=1, stride=1,
                                     padding=0, bias=False)
        self.conv1d_LCM2 = nn.Conv2d(32, LCM_class_num, kernel_size=1, stride=1,
                                     padding=0, bias=False)

    def forward(self, input1, input2):
        # encoding 
        [feat1_1, feat1_2, feat1_3, feat1_4, feat1_5] = self.resnet(input1)
        [feat2_1, feat2_2, feat2_3, feat2_4, feat2_5] = self.resnet(input2)
        
        # decoding
        cbam1_1 = self.CBAM1_1(torch.cat([feat1_5, feat2_5])
        cbam1_2 = self.CBAM1_2(torch.cat([self.up(cbam1_1), feat1_4, feat2_4], 1))
        cbam1_3 = self.CBAM1_3(torch.cat([self.up(cbam1_2), feat1_3, feat2_3], 1))
        cbam1_4 = self.CBAM1_4(torch.cat([self.up(cbam1_3), feat1_2, feat2_2], 1))
        cbam1_5 = self.CBAM1_5(torch.cat([self.up(cbam1_4), feat1_1, feat2_1], 1)

        cbam2_1 = self.CBAM2_1(torch.cat([feat1_5, feat2_5])
        cbam2_2 = self.CBAM2_2(torch.cat([self.up(cbam2_1), feat1_4, feat2_4], 1))
        cbam2_3 = self.CBAM2_3(torch.cat([self.up(cbam2_2), feat1_3, feat2_3], 1))
        cbam2_4 = self.CBAM2_4(torch.cat([self.up(cbam2_3), feat1_2, feat2_2], 1))
        cbam2_5 = self.CBAM2_5(torch.cat([self.up(cbam2_4), feat1_1, feat2_1], 1))
           
        # SCD output
        output = self.relu(self.finalconv1(torch.cat([cbam1_5,cbam2_5]))
        output = self.relu(self.finalconv2(output))
        output = self.relu(self.finalconv3(self.up(output)))
                               
        output = self.conv1d_SCD(output)
                           
        # LCM predictions
        output1 = self.conv1d_LCM1(self.relu(self.LCM_conv1(self.up(cbam1_5))))
        output2 = self.conv1d_LCM2(self.relu(self.LCM_conv2(self.up(cbam2_5))))

        return output1, output2, output


# difference discrimination network
class CDNet(nn.Module):
    def __init__(self):
        # backbone encoder
        super(CDNet, self).__init__()
        self.vgg1 = VGG16()
        # # # decoder
        # # self.CBAM1 = BasicBlock(inplanes=1024, planes=256, stride=1)
        # # self.CBAM2 = BasicBlock(inplanes=2 * 512+256, planes=128, stride=1)
        # # self.CBAM3 = BasicBlock(inplanes=2 * 256+128, planes=64, stride=1)
        # # self.CBAM4 = BasicBlock(inplanes=2 * 128+64, planes=32, stride=1)
        # # self.CBAM5 = BasicBlock(inplanes=2 * 64+32, planes=16, stride=1)

        # self.CBAM1 = Block(in_planes=1024/2, out_planes=256, stride=1)
        # self.CBAM2 = Block(in_planes=512 + 256, out_planes=128, stride=1)
        # self.CBAM3 = Block(in_planes=256 + 128, out_planes=64, stride=1)
        # self.CBAM4 = Block(in_planes=128 + 64, out_planes=32, stride=1)
        # self.CBAM5 = Block(in_planes=64 + 32, out_planes=16, stride=1)

        # self.up = nn.Upsample(scale_factor=2)
        # # self.conv1d1 = nn.Conv2d(16, 1, kernel_size=1, stride=1,
        # # padding=0, bias=False)
        # self.conv1d2 = Block(80, 7, kernel=1)
        # self.conv1d3 = Block(80, 7, kernel=1)

        # self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax2d()



        # 9/13 double u-net
        self.CBAM1_1 = BasicBlock(inplanes=1024, planes=512, stride=1)
        self.CBAM1_2 = BasicBlock(inplanes=512 + 512, planes=256, stride=1)
        self.CBAM1_3 = BasicBlock(inplanes=256 + 256, planes=48, stride=1)
        self.CBAM1_4 = BasicBlock(inplanes=128 + 48, planes=16, stride=1)
        self.CBAM1_5 = BasicBlock(inplanes=64 + 16, planes=16, stride=1)

        self.CBAM2_1 = BasicBlock(inplanes=1024, planes=512, stride=1)
        self.CBAM2_2 = BasicBlock(inplanes=512 + 512, planes=256, stride=1)
        self.CBAM2_3 = BasicBlock(inplanes=256 + 256, planes=48, stride=1)
        self.CBAM2_4 = BasicBlock(inplanes=128 + 48, planes=16, stride=1)
        self.CBAM2_5 = BasicBlock(inplanes=64 + 16, planes=16, stride=1)

        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()

        self.conv1 = Out_Block(in_planes=512, out_planes=1024)
        self.conv1d_img1 = nn.Conv2d(16, 7, kernel_size=3, stride=1,
                                     padding=1, bias=False)
        self.conv1d_img2 = nn.Conv2d(16, 7, kernel_size=3, stride=1,
                                     padding=1, bias=False)

    def forward(self, input1):
        # [feat1_1, feat1_2, feat1_3, feat1_4, feat1_5] = self.vgg1(input1)
        # [feat2_1, feat2_2, feat2_3, feat2_4, feat2_5] = self.vgg1(input2)
        #
        # # [feat1_1, feat1_2, feat1_3, feat1_4, feat1_5] = self.encoding(input1)
        # # [feat2_1, feat2_2, feat2_3, feat2_4, feat2_5] = self.encoding(input2)
        #
        # cbam1 = self.CBAM1(torch.cat([feat1_5, feat2_5], 1))
        # cbam2 = self.CBAM2(torch.cat([self.up(cbam1), feat1_4, feat2_4], 1))
        # cbam3 = self.CBAM3(torch.cat([self.up(cbam2), feat1_3, feat2_3], 1))
        # cbam4 = self.CBAM4(torch.cat([self.up(cbam3), feat1_2, feat2_2], 1))
        # cbam5 = self.CBAM5(torch.cat([self.up(cbam4), feat1_1, feat2_1], 1))
        # # mutli-class CD outputs
        # output_mutli_img1 = self.conv1d2(torch.cat([feat1_1, cbam5],1))
        # output_mutli_img2 = self.conv1d3(torch.cat([feat2_1, cbam5],1))
        # # binary CD output
        # # output_binary = self.conv1d1(cbam5)
        #
        # return output_mutli_img1, output_mutli_img2
        # # , self.sigmoid(output_binary)

        # 9/13 double u net
        [feat1_1, feat1_2, feat1_3, feat1_4, feat1_5] = self.vgg1(input1)
        # [feat2_1, feat2_2, feat2_3, feat2_4, feat2_5] = self.vgg1(input2)

        encode_feat = self.conv1(self.pool(feat1_5))

        cbam1_1 = self.CBAM1_1(self.up(encode_feat))
        cbam1_2 = self.CBAM1_2(torch.cat([self.up(cbam1_1), feat1_4], 1))
        cbam1_3 = self.CBAM1_3(torch.cat([self.up(cbam1_2), feat1_3], 1))
        cbam1_4 = self.CBAM1_4(torch.cat([self.up(cbam1_3), feat1_2], 1))
        cbam1_5 = self.CBAM1_5(torch.cat([self.up(cbam1_4), feat1_1], 1))

        cbam2_1 = self.CBAM2_1(self.up(encode_feat))
        cbam2_2 = self.CBAM2_2(torch.cat([self.up(cbam2_1), feat1_4], 1))
        cbam2_3 = self.CBAM2_3(torch.cat([self.up(cbam2_2), feat1_3], 1))
        cbam2_4 = self.CBAM2_4(torch.cat([self.up(cbam2_3), feat1_2], 1))
        cbam2_5 = self.CBAM2_5(torch.cat([self.up(cbam2_4), feat1_1], 1))

        output1 = self.conv1d_img1(cbam1_5)
        output2 = self.conv1d_img2(cbam2_5)

        return output1, output2


# difference discrimination network
class CDNet_Resnet(nn.Module):
    def __init__(self):
        # backbone encoder
        super(CDNet_Resnet, self).__init__()
        self.resnet = Resnet()

        self.dilate_center = Dblock(1024)

        # # self.dilate_up = Dblock(128)
        # self.conv1d_1 = nn.Conv2d(64, 8, kernel_size=1, stride=1,
        #                              padding=0, bias=False)
        # self.conv1d_2 = nn.Conv2d(64, 8, kernel_size=1, stride=1,
        #                           padding=0, bias=False)

        # 9/18 double u-net
        self.CBAM1_1 = BasicBlock(inplanes=2 * 512, planes=512, stride=1)
        self.CBAM1_2 = BasicBlock(inplanes=2 * 256 + 512, planes=512, stride=1)
        self.CBAM1_3 = BasicBlock(inplanes=2 * 128 + 512, planes=256, stride=1)
        self.CBAM1_4 = BasicBlock(inplanes=2 * 64 + 256, planes=64, stride=1)
        self.CBAM1_5 = BasicBlock(inplanes=64, planes=64, stride=1)

        self.CBAM2_1 = BasicBlock(inplanes=2 * 512, planes=512, stride=1)
        self.CBAM2_2 = BasicBlock(inplanes=2 * 256 + 512, planes=512, stride=1)
        self.CBAM2_3 = BasicBlock(inplanes=2 * 128 + 512, planes=256, stride=1)
        self.CBAM2_4 = BasicBlock(inplanes=2 * 64 + 256, planes=64, stride=1)
        self.CBAM2_5 = BasicBlock(inplanes=64, planes=64, stride=1)

        self.up = nn.UpsamplingNearest2d(scale_factor=2)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()

        self.finalconv1 = nn.Conv2d(64, 32, kernel_size=3, stride=1,
                                    padding=1, bias=False)
        self.finalconv2 = nn.Conv2d(64, 32, kernel_size=3, stride=1,
                                    padding=1, bias=False)

        self.conv1d_img1 = nn.Conv2d(32, 7, kernel_size=1, stride=1,
                                     padding=0, bias=False)
        self.conv1d_img2 = nn.Conv2d(32, 7, kernel_size=1, stride=1,
                                     padding=0, bias=False)

    def forward(self, input1, input2):
        # 9/13 double u net
        [feat1_1, feat1_2, feat1_3, feat1_4, feat1_5] = self.resnet(input1)
        [feat2_1, feat2_2, feat2_3, feat2_4, feat2_5] = self.resnet(input2)

        encode_feat = self.dilate_center(torch.cat([feat1_5, feat2_5], 1))
        # dilate_up = self.dilate_up(torch.cat([feat1_1, feat2_1], 1))

        cbam1_1 = self.CBAM1_1(encode_feat)
        cbam1_2 = self.CBAM1_2(torch.cat([self.up(cbam1_1), feat1_4, feat2_4], 1))
        cbam1_3 = self.CBAM1_3(torch.cat([self.up(cbam1_2), feat1_3, feat2_3], 1))
        cbam1_4 = self.CBAM1_4(torch.cat([self.up(cbam1_3), feat1_2, feat2_2], 1))
        cbam1_5 = self.CBAM1_5(self.up(cbam1_4))

        cbam2_1 = self.CBAM2_1(encode_feat)
        cbam2_2 = self.CBAM2_2(torch.cat([self.up(cbam2_1), feat1_4, feat2_4], 1))
        cbam2_3 = self.CBAM2_3(torch.cat([self.up(cbam2_2), feat1_3, feat2_3], 1))
        cbam2_4 = self.CBAM2_4(torch.cat([self.up(cbam2_3), feat1_2, feat2_2], 1))
        cbam2_5 = self.CBAM2_5(self.up(cbam2_4))

        img1 = self.relu(self.finalconv1(self.up(cbam1_5)))
        img2 = self.relu(self.finalconv2(self.up(cbam2_5)))

        output1 = self.conv1d_img1(img1)
        output2 = self.conv1d_img2(img2)

        return output1, output2
