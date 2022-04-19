import paddle
import paddle.nn as nn

# Feature Difference Enhancement (FDE)
class ChangeAttention(nn.Layer):
    def __init__(self, kernel_size=7):
        super(ChangeAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2D(2, 1, kernel_size, padding=padding, bias_attr=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x1,x2):
        feat = paddle.concat([x1, x2], axis=1)
        x = paddle.abs(x2-x1)
        
        avg_out = paddle.mean(x, axis=1, keepdim=1)
        max_out = paddle.max(x, axis=1, keepdim=1)
        
        x = paddle.concat([avg_out, max_out], axis=1)
        x = self.conv1(x)
        x = self.sigmoid(x)
        out = feat*x + feat
        return out

class ChangeAttention_visualize_FDE(nn.Layer):
    def __init__(self, kernel_size=7):
        super(ChangeAttention_visualize_FDE, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2D(2, 1, kernel_size, padding=padding, bias_attr=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x1,x2):
        feat = paddle.concat([x1, x2], axis=1)
        x = paddle.abs(x2-x1)
        
        avg_out = paddle.mean(x, axis=1, keepdim=1)
        max_out = paddle.max(x, axis=1, keepdim=1)
        
        x = paddle.concat([avg_out, max_out], axis=1)
        x = self.conv1(x)
        x = self.sigmoid(x)
        
        return x

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2D(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias_attr=False)


class CBAM_Module(nn.Layer):
    def __init__(self, channels, reduction):
        super(CBAM_Module, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.max_pool = nn.AdaptiveMaxPool2D(1)
        self.fc1 = nn.Conv2D(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2D(channels // reduction, channels, kernel_size=1, padding=0)

        self.sigmoid_channel = nn.Sigmoid()
        self.conv_after_concat = nn.Conv2D(2, 1, kernel_size=7, stride=1, padding=3)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # Channel Attention Module
        module_input = x
        avg = self.relu(self.fc1(self.avg_pool(x)))
        avg = self.fc2(avg)
        mx = self.relu(self.fc1(self.max_pool(x)))
        mx = self.fc2(mx)
        x = avg + mx
        x = self.sigmoid_channel(x)

        # Spatial Attention Module
        x = module_input * x
        module_input = x
        avg = paddle.mean(x, axis=1, keepdim=1)
        mx = paddle.max(x, axis=1, keepdim=1)
        # print(avg.shape, mx.shape)
        x = paddle.concat([avg, mx], axis=1)
        x = self.conv_after_concat(x)
        x = self.sigmoid_spatial(x)
        x = module_input * x

        return x


class ChannelAttention(nn.Layer):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.max_pool = nn.AdaptiveMaxPool2D(1)

        self.fc1 = nn.Conv2D(in_planes, in_planes // 16, 1, bias_attr=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2D(in_planes // 16, in_planes, 1, bias_attr=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.relu1(self.fc1(self.avg_pool(x)))
        avg_out = self.fc2(avg_out)
        max_out = self.relu1(self.fc1(self.max_pool(x)))
        max_out = self.fc2(max_out)
        out = avg_out + max_out
        out = self.sigmoid(out)
        out = out * x
        return out


class SpatialAttention(nn.Layer):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2D(2, 1, kernel_size, padding=padding, bias_attr=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input_model = x
        avg_out = paddle.mean(x, axis=1, keepdim=1)
        max_out = paddle.max(x, axis=1, keepdim=1)
        # max_out = paddle.cast(max_out,'float32')
        x = paddle.concat([avg_out, max_out], axis=1)
        x = self.conv1(x)
        x = self.sigmoid(x)
        x = input_model * x
        return x


# class BasicBlock(nn.Layer):
#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = nn.BatchNorm2D(planes)
#         self.relu = nn.ReLU()
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2D(planes)

#         self.ca = ChannelAttention(inplanes)
#         self.sa = SpatialAttention()

#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         # residual = x
#         out = self.ca(x)
#         out = self.conv1(out)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         # out = self.relu(out)

        
#         out = self.sa(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         # out += residual
#         out = self.relu(out)

#         return out

class BasicBlock(nn.Layer):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2D(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2D(planes)

        self.ca = ChannelAttention(inplanes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # residual = x
        out = self.ca(x)
        
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.sa(out)
        # out += residual
        out = self.relu(out)

        return out


class BasicBlock_CA(nn.Layer):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_CA, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2D(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2D(planes)

        self.ca = ChannelAttention(inplanes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # residual = x
        out = self.ca(x)
        
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # out += residual
        out = self.relu(out)

        return out



class BasicBlock_orgin(nn.Layer):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_orgin, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2D(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2D(planes)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock_FPN(nn.Layer):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_FPN, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2D(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2D(planes)

        self.ca = ChannelAttention(inplanes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # residual = x
        out = self.ca(x) * x

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        # out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2D(inplanes, planes, kernel_size=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(planes)
        self.conv2 = nn.Conv2D(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(planes)
        self.conv3 = nn.Conv2D(planes, planes * 4, kernel_size=1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(planes * 4)
        self.relu = nn.ReLU()

        self.ca = ChannelAttention(planes * 4)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
