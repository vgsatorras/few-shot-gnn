import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models import gnn_iclr

from typing import Optional, List, Type, Callable, Any, Union
from torch import Tensor


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,

    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class EmbeddingOmniglot(nn.Module):
    ''' In this network the input image is supposed to be 28x28 '''

    def __init__(self, args, emb_size):
        super(EmbeddingOmniglot, self).__init__()
        self.emb_size = emb_size
        self.nef = 64
        self.args = args

        # input is 1 x 28 x 28
        self.conv1 = nn.Conv2d(1, self.nef, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.nef)
        # state size. (nef) x 14 x 14
        self.conv2 = nn.Conv2d(self.nef, self.nef, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.nef)

        # state size. (1.5*ndf) x 7 x 7
        self.conv3 = nn.Conv2d(self.nef, self.nef, 3, bias=False)
        self.bn3 = nn.BatchNorm2d(self.nef)
        # state size. (2*ndf) x 5 x 5
        self.conv4 = nn.Conv2d(self.nef, self.nef, 3, bias=False)
        self.bn4 = nn.BatchNorm2d(self.nef)
        # state size. (2*ndf) x 3 x 3
        # self.fc_last = nn.Linear(3 * 3 * self.nef, self.emb_size, bias=False)
        self.fc_last2 = nn.Linear(2 * 2 * self.nef, self.emb_size, bias=False)
        self.bn_last = nn.BatchNorm1d(self.emb_size)

        # self.block: Type[Union[BasicBlock, Bottleneck]]
        # self.layers: List[int]
        self.block = BasicBlock
        self.layers = [1,1,1,1]
        self.replace_stride_with_dilation: Optional[List[bool]] = None

        if self.replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            self.replace_stride_with_dilation = [False, False, False]
        if len(self.replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {self.replace_stride_with_dilation}"
            )

        self._norm_layer = nn.BatchNorm2d
        self.dilation = 1
        self.inplanes = 64
        self.groups = 1
        self.base_width = 64

        self.layer1 = self._make_layer(self.block, self.nef, self.layers[0])
        self.layer2 = self._make_layer(self.block, self.nef, self.layers[1], stride=1,   # stride=2,
                                       dilate=self.replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(self.block, self.nef, self.layers[2], stride=2,
                                       dilate=self.replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(self.block, self.nef, self.layers[3], stride=2,
                                       dilate=self.replace_stride_with_dilation[2])

    def forward(self, inputs):
        e1 = F.max_pool2d(self.bn1(self.conv1(inputs)), 2)
        x = F.leaky_relu(e1, 0.1, inplace=True)

        e2 = F.max_pool2d(self.bn2(self.layer2(x)), 2)
        x = F.leaky_relu(e2, 0.1, inplace=True)

        e3 = self.bn3(self.layer3(x))
        x = F.leaky_relu(e3, 0.1, inplace=True)
        e4 = self.bn4(self.layer4(x))
        x = F.leaky_relu(e4, 0.1, inplace=True)
        x = x.view(-1, 2 * 2 * self.nef)
        #  3*3  # RuntimeError: shape '[-1, 576]' is invalid for input of size 19200

        output = F.leaky_relu(self.bn_last(self.fc_last2(x)))

        return [e1, e2, e3, output]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)


class EmbeddingImagenet(nn.Module):
    ''' In this network the input image is supposed to be 28x28 '''

    def __init__(self, args, emb_size):
        super(EmbeddingImagenet, self).__init__()
        self.emb_size = emb_size
        self.ndf = 64
        self.args = args

        # Input 84x84x3
        self.conv1 = nn.Conv2d(3, self.ndf, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.ndf)

        # Input 42x42x64
        self.conv2 = nn.Conv2d(self.ndf, int(self.ndf * 1.5), kernel_size=3, bias=False)
        self.bn2 = nn.BatchNorm2d(int(self.ndf * 1.5))

        # Input 20x20x96
        self.conv3 = nn.Conv2d(int(self.ndf * 1.5), self.ndf * 2, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.ndf * 2)
        self.drop_3 = nn.Dropout2d(0.4)

        # Input 10x10x128
        self.conv4 = nn.Conv2d(self.ndf * 2, self.ndf * 4, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(self.ndf * 4)
        self.drop_4 = nn.Dropout2d(0.5)

        # Input 5x5x256
        self.fc1 = nn.Linear(self.ndf * 4 * 5 * 5, self.emb_size, bias=True)
        self.bn_fc = nn.BatchNorm1d(self.emb_size)

    def forward(self, input):
        e1 = F.max_pool2d(self.bn1(self.conv1(input)), 2)
        x = F.leaky_relu(e1, 0.2, inplace=True)
        e2 = F.max_pool2d(self.bn2(self.conv2(x)), 2)
        x = F.leaky_relu(e2, 0.2, inplace=True)
        e3 = F.max_pool2d(self.bn3(self.conv3(x)), 2)
        x = F.leaky_relu(e3, 0.2, inplace=True)
        x = self.drop_3(x)
        e4 = F.max_pool2d(self.bn4(self.conv4(x)), 2)
        x = F.leaky_relu(e4, 0.2, inplace=True)
        x = self.drop_4(x)
        x = x.view(-1, self.ndf * 4 * 5 * 5)
        output = self.bn_fc(self.fc1(x))

        return [e1, e2, e3, e4, None, output]


class MetricNN(nn.Module):
    def __init__(self, args, emb_size):
        super(MetricNN, self).__init__()

        self.metric_network = args.metric_network
        self.emb_size = emb_size
        self.args = args

        if self.metric_network == 'gnn_iclr_nl':
            assert (self.args.train_N_way == self.args.test_N_way)
            num_inputs = self.emb_size + self.args.train_N_way
            if self.args.dataset == 'mini_imagenet':
                self.gnn_obj = gnn_iclr.GNN_nl(args, num_inputs, nf=96, J=1)
            elif 'omniglot' in self.args.dataset:
                self.gnn_obj = gnn_iclr.GNN_nl_omniglot(args, num_inputs, nf=96, J=1)
        elif self.metric_network == 'gnn_iclr_active':
            assert (self.args.train_N_way == self.args.test_N_way)
            num_inputs = self.emb_size + self.args.train_N_way
            self.gnn_obj = gnn_iclr.GNN_active(args, num_inputs, 96, J=1)
        else:
            raise NotImplementedError

    def gnn_iclr_forward(self, z, zi_s, labels_yi):
        # Creating WW matrix
        zero_pad = Variable(torch.zeros(labels_yi[0].size()))
        if self.args.cuda:
            zero_pad = zero_pad.cuda()

        labels_yi = [zero_pad] + labels_yi
        zi_s = [z] + zi_s
        nodes = [torch.cat([zi, label_yi], 1) for zi, label_yi in zip(zi_s, labels_yi)]
        nodes = [node.unsqueeze(1) for node in nodes]
        nodes = torch.cat(nodes, 1)

        logits = self.gnn_obj(nodes).squeeze(-1)
        outputs = F.sigmoid(logits)

        return outputs, logits

    def gnn_iclr_active_forward(self, z, zi_s, labels_yi, oracles_yi, hidden_layers):
        # Creating WW matrix
        zero_pad = Variable(torch.zeros(labels_yi[0].size()))
        if self.args.cuda:
            zero_pad = zero_pad.cuda()

        labels_yi = [zero_pad] + labels_yi
        zi_s = [z] + zi_s

        nodes = [torch.cat([label_yi, zi], 1) for zi, label_yi in zip(zi_s, labels_yi)]
        nodes = [node.unsqueeze(1) for node in nodes]
        nodes = torch.cat(nodes, 1)

        oracles_yi = [zero_pad] + oracles_yi
        oracles_yi = [oracle_yi.unsqueeze(1) for oracle_yi in oracles_yi]
        oracles_yi = torch.cat(oracles_yi, 1)

        logits = self.gnn_obj(nodes, oracles_yi, hidden_layers).squeeze(-1)
        outputs = F.sigmoid(logits)

        return outputs, logits

    def forward(self, inputs):
        '''input: [batch_x, [batches_xi], [labels_yi]]'''
        [z, zi_s, labels_yi, oracles_yi, hidden_labels] = inputs

        if 'gnn_iclr_active' in self.metric_network:
            return self.gnn_iclr_active_forward(z, zi_s, labels_yi, oracles_yi, hidden_labels)
        elif 'gnn_iclr' in self.metric_network:
            return self.gnn_iclr_forward(z, zi_s, labels_yi)
        else:
            raise NotImplementedError


class SoftmaxModule():
    def __init__(self):
        self.softmax_metric = 'log_softmax'

    def forward(self, outputs):
        if self.softmax_metric == 'log_softmax':
            return F.log_softmax(outputs)
        else:
            raise (NotImplementedError)


def load_model(model_name, args, io):
    try:
        model = torch.load('checkpoints/%s/models/%s.t7' % (args.exp_name, model_name))
        io.cprint('Loading Parameters from the last trained %s Model' % model_name)
        return model
    except:
        io.cprint('Initiallize new Network Weights for %s' % model_name)
        pass
    return None


def create_models(args):
    print(args.dataset)

    if 'omniglot' == args.dataset:
        enc_nn = EmbeddingOmniglot(args, 64)
    elif 'mini_imagenet' == args.dataset:
        enc_nn = EmbeddingImagenet(args, 128)
    else:
        raise NameError('Dataset ' + args.dataset + ' not knows')
    return enc_nn, MetricNN(args, emb_size=enc_nn.emb_size)
