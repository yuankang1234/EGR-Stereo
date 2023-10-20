from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F

from models.deformable_refine import DeformableRefineF
from models.effnetv2 import effnetv2_ykdesign_6
from models.relu.submodule_cfnet import *

import math
class SubModule(nn.Module):
    def __init__(self):
        super(SubModule, self).__init__()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
                # if m.bias is not None:
                #     m.bias.data.zero_()


class feature_extraction(nn.Module):
    def __init__(self, concat_feature=False, concat_feature_channel=12):
        super(feature_extraction, self).__init__()
        self.concat_feature = concat_feature
        efficinet_v2 = effnetv2_ykdesign_6()

        self.inplanes = 32
        # self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
        #                                Mish(),
        #                                convbn(32, 32, 3, 1, 1, 1),
        #                                Mish(),
        #                                convbn(32, 32, 3, 1, 1, 1),
        #                                Mish())

        # self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        # self.layer2 = self._make_layer(BasicBlock, 64, 1, 1, 1, 1)
        # self.layer3 = self._make_layer(BasicBlock, 128, 1, 2, 1, 1)
        # self.layer4 = self._make_layer(BasicBlock, 192, 1, 2, 1, 1)
        # self.layer5 = self._make_layer(BasicBlock, 256, 1, 2, 1, 1)
        # self.layer6 = self._make_layer(BasicBlock, 512, 1, 2, 1, 1)
        # efficientv2特征提取
        self.conv_stem = efficinet_v2.features[0]  # conv2d
        self.block1 = torch.nn.Sequential(*efficinet_v2.features[1:4])  # 1~3(3)
        self.block2 = torch.nn.Sequential(*efficinet_v2.features[4:9])  # 4~8(5)
        self.block3 = torch.nn.Sequential(*efficinet_v2.features[9:14])  # 9~13(5)
        self.block4 = torch.nn.Sequential(*efficinet_v2.features[14:26])  # 14~25(12)
        self.block5 = torch.nn.Sequential(*efficinet_v2.features[26:40])  # 26~39(14)
        self.block6 = torch.nn.Sequential(*efficinet_v2.features[40:56])  # 40~55(16)
        self.block7 = torch.nn.Sequential(*efficinet_v2.features[56:74])  # 56~73(18)

        self.pyramid_pooling = pyramidPooling(512, None, fusion_mode='sum', model_name='icnet')
        self.upconv6 = nn.Sequential(nn.Upsample(scale_factor=2),
                                     convbn(176, 176, 3, 1, 1, 1),
                                     Mish())
        self.iconv5 = nn.Sequential(convbn(320, 256, 3, 1, 1, 1),
                                    Mish())
        self.upconv5 = nn.Sequential(nn.Upsample(scale_factor=2),
                                     convbn(256, 192, 3, 1, 1, 1),
                                     Mish())
        self.iconv4 = nn.Sequential(convbn(320, 192, 3, 1, 1, 1),
                                    Mish())
        self.upconv4 = nn.Sequential(nn.Upsample(scale_factor=2),
                                     convbn(192, 128, 3, 1, 1, 1),
                                     Mish())
        self.iconv3 = nn.Sequential(convbn(240, 128, 3, 1, 1, 1),
                                    Mish())
        self.upconv3 = nn.Sequential(nn.Upsample(scale_factor=2),
                                     convbn(128, 64, 3, 1, 1, 1),
                                     Mish())
        self.iconv2 = nn.Sequential(convbn(112, 64, 3, 1, 1, 1),
                                    Mish())
        # self.upconv2 = nn.Sequential(nn.Upsample(scale_factor=2),
        #                              convbn(64, 32, 3, 1, 1, 1),
        #                              nn.ReLU(inplace=True))

        # self.gw1 = nn.Sequential(convbn(32, 40, 3, 1, 1, 1),
        #                          nn.ReLU(inplace=True),
        #                          nn.Conv2d(40, 40, kernel_size=1, padding=0, stride=1,
        #                                    bias=False))

        self.gw2 = nn.Sequential(convbn(64, 80, 3, 1, 1, 1),
                                 Mish(),
                                 nn.Conv2d(80, 80, kernel_size=1, padding=0, stride=1,
                                           bias=False))

        self.gw3 = nn.Sequential(convbn(128, 160, 3, 1, 1, 1),
                                 Mish(),
                                 nn.Conv2d(160, 160, kernel_size=1, padding=0, stride=1,
                                           bias=False))

        self.gw4 = nn.Sequential(convbn(192, 160, 3, 1, 1, 1),
                                 Mish(),
                                 nn.Conv2d(160, 160, kernel_size=1, padding=0, stride=1,
                                           bias=False))

        self.gw5 = nn.Sequential(convbn(256, 320, 3, 1, 1, 1),
                                 Mish(),
                                 nn.Conv2d(320, 320, kernel_size=1, padding=0, stride=1,
                                           bias=False))

        self.gw6 = nn.Sequential(convbn(176, 320, 3, 1, 1, 1),
                                 Mish(),
                                 nn.Conv2d(320, 320, kernel_size=1, padding=0, stride=1,
                                           bias=False))

        if self.concat_feature:
            # self.concat1 = nn.Sequential(convbn(32, 16, 3, 1, 1, 1),
            #                              nn.ReLU(inplace=True),
            #                              nn.Conv2d(16, concat_feature_channel // 4, kernel_size=1, padding=0, stride=1,
            #                                        bias=False))

            self.concat2 = nn.Sequential(convbn(64, 32, 3, 1, 1, 1),
                                         Mish(),
                                         nn.Conv2d(32, concat_feature_channel // 2, kernel_size=1, padding=0, stride=1,
                                                   bias=False))
            self.concat3 = nn.Sequential(convbn(128, 128, 3, 1, 1, 1),
                                         Mish(),
                                         nn.Conv2d(128, concat_feature_channel, kernel_size=1, padding=0, stride=1,
                                                   bias=False))

            self.concat4 = nn.Sequential(convbn(192, 128, 3, 1, 1, 1),
                                         Mish(),
                                         nn.Conv2d(128, concat_feature_channel, kernel_size=1, padding=0, stride=1,
                                                   bias=False))

            self.concat5 = nn.Sequential(convbn(256, 128, 3, 1, 1, 1),
                                         Mish(),
                                         nn.Conv2d(128, concat_feature_channel, kernel_size=1, padding=0, stride=1,
                                                   bias=False))

            self.concat6 = nn.Sequential(convbn(176, 128, 3, 1, 1, 1),
                                         Mish(),
                                         nn.Conv2d(128, concat_feature_channel, kernel_size=1, padding=0, stride=1,
                                                   bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_stem(x)  # (1,32,128,256)
        # x = self.layer1(x)
        l1 = self.block1(x)  # (1,24,128,256)  # 1/2
        l2 = self.block2(l1)  # (1,48,128,256)  # 1/2
        l3 = self.block3(l2)  # (1,64,64,128)  # 1/4
        l4 = self.block4(l3)  # (1,112,64,128)  # 1/4
        l5 = self.block5(l4)  # (1,128,32,64) # 1/8
        l6 = self.block6(l5)  # (1,144,16,32) #1/16
        l7 = self.block7(l6)  # (1,176,8,16)

        concat5 = torch.cat((l6, self.upconv6(l7)), dim=1)  # (1,320,16,32)
        decov_5 = self.iconv5(concat5)  # (1,256,16,32)
        concat4 = torch.cat((l5, self.upconv5(decov_5)), dim=1)  # (1,320,32,64)
        # concat4 = torch.cat((l4, self.upconv5(l5)), dim=1)
        decov_4 = self.iconv4(concat4)  # (1,192,32,64)
        concat3 = torch.cat((l4, self.upconv4(decov_4)), dim=1)  # (1,256,64,128)
        decov_3 = self.iconv3(concat3)  # (1,128,64,128)
        concat2 = torch.cat((l2, self.upconv3(decov_3)), dim=1)  # (1,128,128,256)
        decov_2 = self.iconv2(concat2)  # (1,64,128,256)
        # decov_1 = self.upconv2(decov_2)

        # gw1 = self.gw1(decov_1)
        gw2 = self.gw2(decov_2)  # (1,80,128,256)
        gw3 = self.gw3(decov_3)  # (1,160,64,128)
        gw4 = self.gw4(decov_4)  # (1,160,32,64)
        gw5 = self.gw5(decov_5)  # (1,320,16,32)
        gw6 = self.gw6(l7)  # (1,320,8,16)

        if not self.concat_feature:
            return {"gw2": gw2, "gw3": gw3, "gw4": gw4}
        else:
            # concat_feature1 = self.concat1(decov_1)
            concat_feature2 = self.concat2(decov_2)  # (1,6,128,256)
            concat_feature3 = self.concat3(decov_3)  # (1,12,64,128)
            concat_feature4 = self.concat4(decov_4)  # (1,12,32,64)
            concat_feature5 = self.concat5(decov_5)  # (1,12,16,32)
            concat_feature6 = self.concat6(l7)  # (1,12,8,16)
            return {"gw2": gw2, "gw3": gw3, "gw4": gw4, "gw5": gw5, "gw6": gw6,
                    "concat_feature2": concat_feature2, "concat_feature3": concat_feature3,
                    "concat_feature4": concat_feature4,
                    "concat_feature5": concat_feature5, "concat_feature6": concat_feature6}


class hourglassup(nn.Module):
    def __init__(self, in_channels):
        super(hourglassup, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, in_channels * 2, kernel_size=3, stride=2,
                               padding=1, bias=False)

        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
                                   Mish())

        self.conv3 = nn.Conv3d(in_channels * 2, in_channels * 4, kernel_size=3, stride=2,
                               padding=1, bias=False)

        self.conv4 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
                                   Mish())

        self.conv8 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))

        self.combine1 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 2, 3, 1, 1),
                                      Mish())
        self.combine2 = nn.Sequential(convbn_3d(in_channels * 6, in_channels * 4, 3, 1, 1),
                                      Mish())
        self.combine3 = nn.Sequential(convbn_3d(in_channels * 6, in_channels * 4, 3, 1, 1),
                                      Mish())

        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)
        self.redir3 = convbn_3d(in_channels * 4, in_channels * 4, kernel_size=1, stride=1, pad=0)

    def forward(self, x, feature4, feature5):
        conv1 = self.conv1(x)  # 1/8
        conv1 = torch.cat((conv1, feature4), dim=1)  # 1/8
        conv1 = self.combine1(conv1)  # 1/8
        conv2 = self.conv2(conv1)  # 1/8

        conv3 = self.conv3(conv2)  # 1/16
        conv3 = torch.cat((conv3, feature5), dim=1)  # 1/16
        conv3 = self.combine2(conv3)  # 1/16
        conv4 = self.conv4(conv3)  # 1/16

        conv8 = FMish(self.conv8(conv4) + self.redir2(conv2))  # 1/8--yk
        conv9 = FMish(self.conv9(conv8) + self.redir1(x))  # 1/4--yk

        return conv9


class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(in_channels, in_channels * 2, 3, 2, 1),
                                   Mish())

        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
                                   Mish())

        self.conv3 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 4, 3, 2, 1),
                                   Mish())

        self.conv4 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
                                   Mish())

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))

        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        # conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        # conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)

        conv5 = FMish(self.conv5(conv4) + self.redir2(conv2))
        conv6 = FMish(self.conv6(conv5) + self.redir1(x))

        return conv6

class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, **kwargs):
        super(BasicConv, self).__init__()

        self.relu = relu
        self.use_bn = bn
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm2d(out_channels)
        self.LeakyReLU = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = self.LeakyReLU(x)  # , inplace=True)
        return x

class channelAtt(SubModule):
    # 原本是SubModule,我这边试试nn.Module有没有问题
    def __init__(self, cv_chan, im_chan, D):
        super(channelAtt, self).__init__()

        self.im_att = nn.Sequential(
            BasicConv(im_chan, im_chan // 2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(im_chan // 2, cv_chan, 1))

        self.weight_init()

    def forward(self, cv, im):
        '''
        '''
        channel_att = self.im_att(im).unsqueeze(2)
        cv = torch.sigmoid(channel_att) * cv
        return cv

class cfnet(nn.Module):
    def __init__(self, maxdisp, use_concat_volume=False):
        super(cfnet, self).__init__()
        self.maxdisp = maxdisp
        self.use_concat_volume = use_concat_volume
        self.v_scale_s1 = 1
        self.v_scale_s2 = 2
        self.v_scale_s3 = 3
        self.sample_count_s1 = 6
        self.sample_count_s2 = 10
        self.sample_count_s3 = 14
        self.num_groups = 40
        self.uniform_sampler = UniformSampler()
        self.spatial_transformer = SpatialTransformer()

        if self.use_concat_volume:
            self.concat_channels = 12
            self.feature_extraction = feature_extraction(concat_feature=True,
                                                         concat_feature_channel=self.concat_channels)
        else:  # 走这个
            self.concat_channels = 0
            self.feature_extraction = feature_extraction(concat_feature=False)

        self.dres0 = nn.Sequential(convbn_3d(self.num_groups + self.concat_channels * 2, 32, 3, 1, 1),
                                   Mish(),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   Mish())

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   Mish(),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.dres0_5 = nn.Sequential(convbn_3d(self.num_groups + self.concat_channels * 2, 64, 3, 1, 1),
                                     Mish(),
                                     convbn_3d(64, 64, 3, 1, 1),
                                     Mish())

        self.dres1_5 = nn.Sequential(convbn_3d(64, 64, 3, 1, 1),
                                     Mish(),
                                     convbn_3d(64, 64, 3, 1, 1))

        self.dres0_6 = nn.Sequential(convbn_3d(self.num_groups + self.concat_channels * 2, 64, 3, 1, 1),
                                     Mish(),
                                     convbn_3d(64, 64, 3, 1, 1),
                                     Mish())

        self.dres1_6 = nn.Sequential(convbn_3d(64, 64, 3, 1, 1),
                                     Mish(),
                                     convbn_3d(64, 64, 3, 1, 1))

        self.combine1 = hourglassup(32)

        # self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        # self.dres4 = hourglass(32)

        self.confidence0_s3 = nn.Sequential(convbn_3d(self.num_groups + self.concat_channels * 2 + 1, 32, 3, 1, 1),
                                            Mish(),
                                            convbn_3d(32, 32, 3, 1, 1),
                                            Mish())

        self.confidence1_s3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                            Mish(),
                                            convbn_3d(32, 32, 3, 1, 1))

        self.confidence2_s3 = hourglass(32)

        self.confidence3_s3 = hourglass(32)

        self.confidence0_s2 = nn.Sequential(convbn_3d(self.num_groups // 2 + self.concat_channels + 1, 16, 3, 1, 1),
                                            Mish(),
                                            convbn_3d(16, 16, 3, 1, 1),
                                            Mish())

        self.confidence1_s2 = nn.Sequential(convbn_3d(16, 16, 3, 1, 1),
                                            Mish(),
                                            convbn_3d(16, 16, 3, 1, 1))

        self.confidence2_s2 = hourglass(16)

        self.confidence3_s2 = hourglass(16)

        # self.confidence0_s1 = nn.Sequential(convbn_3d(self.num_groups // 4 + self.concat_channels // 2 + 1, 16, 3, 1, 1),
        #                                     nn.ReLU(inplace=True),
        #                                     convbn_3d(16, 16, 3, 1, 1),
        #                                     nn.ReLU(inplace=True))
        #
        # self.confidence1_s1 = nn.Sequential(convbn_3d(16, 16, 3, 1, 1),
        #                                     nn.ReLU(inplace=True),
        #                                     convbn_3d(16, 16, 3, 1, 1))
        #
        # self.confidence2_s1 = hourglass(16)

        # self.confidence3 = hourglass(32)
        #
        # self.confidence4 = hourglass(32)

        self.confidence_classif0_s3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                                    Mish(),
                                                    nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.confidence_classif1_s3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                                    Mish(),
                                                    nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.confidence_classifmid_s3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                                      Mish(),
                                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.confidence_classif0_s2 = nn.Sequential(convbn_3d(16, 16, 3, 1, 1),
                                                    Mish(),
                                                    nn.Conv3d(16, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.confidence_classif1_s2 = nn.Sequential(convbn_3d(16, 16, 3, 1, 1),
                                                    Mish(),
                                                    nn.Conv3d(16, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.confidence_classifmid_s2 = nn.Sequential(convbn_3d(16, 16, 3, 1, 1),
                                                      Mish(),
                                                      nn.Conv3d(16, 1, kernel_size=3, padding=1, stride=1, bias=False))

        # self.confidence_classif1_s1 = nn.Sequential(convbn_3d(16, 16, 3, 1, 1),
        #                                             nn.ReLU(inplace=True),
        #                                             nn.Conv3d(16, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif0 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      Mish(),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      Mish(),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      Mish(),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))
        # 这个node_n=2对应论文里面N=2
        # N是控制reassembled（重新组装的）邻居的数量的超参数
        # modulation=True表示对不同邻居给予不同的权重
        self.refine_module = DeformableRefineF(feature_c=64, node_n=2, modulation=True, cost=True)
        self.change_160_32 = nn.Sequential(convbn(160, 128, 3, 1, 1, 1),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(128, 32, kernel_size=1, padding=0, stride=1, bias=False))
        self.change_80_16 = nn.Sequential(convbn(80, 128, 3, 1, 1, 1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(128, 16, kernel_size=1, padding=0, stride=1, bias=False))
        self.channelAtt_1_4 = channelAtt(32, 32, int(self.maxdisp // 4))
        self.channelAtt_1_2 = channelAtt(16, 16, int(self.maxdisp // 2))
        self.gamma_s3 = nn.Parameter(torch.zeros(1))
        self.beta_s3 = nn.Parameter(torch.zeros(1))
        self.gamma_s2 = nn.Parameter(torch.zeros(1))
        self.beta_s2 = nn.Parameter(torch.zeros(1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
            # m.bias.data.zero_()

    def generate_search_range(self, sample_count, input_min_disparity, input_max_disparity, scale):
        """
        Description:    Generates the disparity search range.

        Returns:
            :min_disparity: Lower bound of disparity search range
            :max_disparity: Upper bound of disaprity search range.
        """

        min_disparity = torch.clamp(input_min_disparity - torch.clamp((
                sample_count - input_max_disparity + input_min_disparity), min=0) / 2.0, min=0,
                                    max=self.maxdisp // (2 ** scale) - 1)
        max_disparity = torch.clamp(input_max_disparity + torch.clamp(
            sample_count - input_max_disparity + input_min_disparity, min=0) / 2.0, min=0,
                                    max=self.maxdisp // (2 ** scale) - 1)

        return min_disparity, max_disparity

    def generate_disparity_samples(self, min_disparity, max_disparity, sample_count=12):
        """
        Description:    Generates "sample_count" number of disparity samples from the
                                                            search range (min_disparity, max_disparity)
                        Samples are generated by uniform sampler

        Args:
            :min_disparity: LowerBound of the disaprity search range.
            :max_disparity: UpperBound of the disparity search range.
            :sample_count: Number of samples to be generated from the input search range.

        Returns:
            :disparity_samples:
        """
        disparity_samples = self.uniform_sampler(min_disparity, max_disparity, sample_count)

        disparity_samples = torch.cat((torch.floor(min_disparity), disparity_samples, torch.ceil(max_disparity)),
                                      dim=1).long()  # disparity level = sample_count + 2
        return disparity_samples

    def cost_volume_generator(self, left_input, right_input, disparity_samples, model='concat', num_groups=40):
        """
        Description: Generates cost-volume using left image features, disaprity samples
                                                            and warped right image features.
        Args:
            :left_input: Left Image fetaures
            :right_input: Right Image features
            :disparity_samples: Disaprity samples
            :model : concat or group correlation

        Returns:
            :cost_volume:
            :disaprity_samples:
        """

        right_feature_map, left_feature_map = self.spatial_transformer(left_input,
                                                                       right_input, disparity_samples)
        disparity_samples = disparity_samples.unsqueeze(1).float()
        if model == 'concat':
            cost_volume = torch.cat((left_feature_map, right_feature_map), dim=1)
        else:
            cost_volume = groupwise_correlation_4D(left_feature_map, right_feature_map, num_groups)

        return cost_volume, disparity_samples

    def forward(self, left, right):
        features_left = self.feature_extraction(left)
        features_right = self.feature_extraction(right)

        gwc_volume3 = build_gwc_volume(features_left["gw3"], features_right["gw3"], self.maxdisp // 4,
                                       self.num_groups)  # (1,40,48,64,128)#1/4

        gwc_volume4 = build_gwc_volume(features_left["gw4"], features_right["gw4"], self.maxdisp // 8,
                                       self.num_groups)  # (1,40,24,32,64)#1/8

        gwc_volume5 = build_gwc_volume(features_left["gw5"], features_right["gw5"], self.maxdisp // 16,
                                       self.num_groups)  # (1,40,12,16,32)#1/16

        gwc_volume6 = build_gwc_volume(features_left["gw6"], features_right["gw6"], self.maxdisp // 32,
                                       self.num_groups)  # (1,40,6,8,16)#1/32
        if self.use_concat_volume:  # 走这个
            concat_volume3 = build_concat_volume(features_left["concat_feature3"], features_right["concat_feature3"],
                                                 self.maxdisp // 4)  # (1,24,48,64,128)
            concat_volume4 = build_concat_volume(features_left["concat_feature4"], features_right["concat_feature4"],
                                                 self.maxdisp // 8)  # (1,24,24,32,64)
            concat_volume5 = build_concat_volume(features_left["concat_feature5"], features_right["concat_feature5"],
                                                 self.maxdisp // 16)  # (1,24,12,16,32)
            # concat_volume6 = build_concat_volume(features_left["concat_feature6"], features_right["concat_feature6"],
            #                                      self.maxdisp // 32)  # (1,24,6,8,16)
            volume3 = torch.cat((gwc_volume3, concat_volume3), 1)  # (1,64,48,64,128)#1/4
            volume4 = torch.cat((gwc_volume4, concat_volume4), 1)  # (1,64,24,32,64)#1/8
            volume5 = torch.cat((gwc_volume5, concat_volume5), 1)  # (1,64,12,16,32)#1/16
            # volume6 = torch.cat((gwc_volume6, concat_volume6), 1)  # (1,64,6,8,16)#1/32

        else:
            volume3 = gwc_volume3

        cost0_4 = self.dres0(volume3)  # (1,32,48,64,128)
        # 插入GCE操作
        # 此时需要一个(1,32,64,128)的左图特征
        # 230803_10_44,目前只在这个地方加了个GCE，想想别的地方能不能加,volumn5和volumn6还没加呢
        # left_fea_1_4 = self.change_160_32(features_left["gw3"])  # (1,32,64,128)
        # cost0_4 = self.channelAtt_1_4(cost0_4, left_fea_1_4)  # (1,32,48,64,128)  # 这一步就是GCE的关键所在

        cost0_4 = self.dres1(cost0_4) + cost0_4  # (1,32,48,64,128)#1/4

        cost0_5 = self.dres0_5(volume4)  # (1,64,24,32,64)#1/8
        cost0_5 = self.dres1_5(cost0_5) + cost0_5  # (1,64,24,32,64)#1/8
        cost0_6 = self.dres0_6(volume5)  # (1,64,12,16,32)
        cost0_6 = self.dres1_6(cost0_6) + cost0_6  # (1,64,12,16,32)#1/16
        out1_4 = self.combine1(cost0_4, cost0_5, cost0_6)  # (1,32,48,64,128)#1/4
        out2_4 = self.dres3(out1_4)  # (1,32,48,64,128)#1/4
        #GCE
        left_fea_1_4 = self.change_160_32(features_left["gw3"])  # (1,32,64,128)
        out2_4 = self.channelAtt_1_4(out2_4, left_fea_1_4)  # (1,32,48,64,128)  # 这一步就是GCE的关键所在

        cost2_s4 = self.classif2(out2_4)  # (1,1,48,64,128)#1/4
        cost2_s4 = torch.squeeze(cost2_s4, 1)  # (1,48,64,128)#1/4
        #开始CSR
        cost2_s4 = self.refine_module(left, cost2_s4)
        pred2_possibility_s4 = F.softmax(cost2_s4, dim=1)  # (1,48,64,128)#1/4
        pred2_s4 = disparity_regression(pred2_possibility_s4, self.maxdisp // 4).unsqueeze(1)  # (1,1,64,128)
        pred2_s4_cur = pred2_s4.detach()  # (1,1,64,128)
        pred2_v_s4 = disparity_variance(pred2_possibility_s4, self.maxdisp // 4,
                                        pred2_s4_cur)  # (1,1,32,64)  # get the variance
        pred2_v_s4 = pred2_v_s4.sqrt()  # (1,1,64,128)
        mindisparity_s3 = pred2_s4_cur - (self.gamma_s3 + 1) * pred2_v_s4 - self.beta_s3  # (1,1,64,128)#1/4
        maxdisparity_s3 = pred2_s4_cur + (self.gamma_s3 + 1) * pred2_v_s4 + self.beta_s3  # (1,1,64,128)
        maxdisparity_s3 = F.upsample(maxdisparity_s3 * 2, [left.size()[2] // 4, left.size()[3] // 4], mode='bilinear',
                                     align_corners=True)  # (1,1,64,128)#1/4
        mindisparity_s3 = F.upsample(mindisparity_s3 * 2, [left.size()[2] // 4, left.size()[3] // 4], mode='bilinear',
                                     align_corners=True)  # (1,1,64,128)#1/4

        mindisparity_s3_1, maxdisparity_s3_1 = self.generate_search_range(self.sample_count_s3 + 1, mindisparity_s3,
                                                                          maxdisparity_s3,
                                                                          scale=2)  # (1,1,64,128)and#(1,1,64,128)#1/4and1/4
        disparity_samples_s3 = self.generate_disparity_samples(mindisparity_s3_1, maxdisparity_s3_1,
                                                               self.sample_count_s3).float()  # (1,16,64,128)#1/4
        confidence_v_concat_s3, _ = self.cost_volume_generator(features_left["concat_feature3"],
                                                               features_right["concat_feature3"], disparity_samples_s3,
                                                               'concat')  # (1,24,16,64,128)
        confidence_v_gwc_s3, disparity_samples_s3 = self.cost_volume_generator(features_left["gw3"],
                                                                               features_right["gw3"],
                                                                               disparity_samples_s3, 'gwc',
                                                                               self.num_groups)  # (1,40,16,64,128)and(1,1,16,64,128)
        confidence_v_s3 = torch.cat((confidence_v_gwc_s3, confidence_v_concat_s3, disparity_samples_s3), dim=1)
        # confidence_v_s3(1,65,16,64,128)

        disparity_samples_s3 = torch.squeeze(disparity_samples_s3, dim=1)  # (1,16,64,128)

        cost0_s3 = self.confidence0_s3(confidence_v_s3)  # (1,32,16,64,128)
        cost0_s3 = self.confidence1_s3(cost0_s3) + cost0_s3  # (1,32,16,64,128)
        #GCE
        # left_fea_1_4 = self.change_160_32(features_left["gw3"])  # (1,32,64,128)
        cost0_s3 = self.channelAtt_1_4(cost0_s3, left_fea_1_4)  # (1,32,16,64,128)  # 这一步就是GCE的关键所在

        out1_s3 = self.confidence2_s3(cost0_s3)  # (1,32,16,64,128)
        out2_s3 = self.confidence3_s3(out1_s3)  # (1,32,16,64,128)

        cost1_s3 = self.confidence_classif1_s3(out2_s3).squeeze(1)  # (1,16,64,128)
        cost1_s3_possibility = F.softmax(cost1_s3, dim=1)  # (1,16,64,128)
        pred1_s3 = torch.sum(cost1_s3_possibility * disparity_samples_s3, dim=1, keepdim=True)  # (1,1,64,128)
        pred1_s3_cur = pred1_s3.detach()  # (1,1,64,128)
        pred1_v_s3 = disparity_variance_confidence(cost1_s3_possibility, disparity_samples_s3,
                                                   pred1_s3_cur)  # (1,1,64,128)
        pred1_v_s3 = pred1_v_s3.sqrt()  # (1,1,64,128)
        mindisparity_s2 = pred1_s3_cur - (self.gamma_s2 + 1) * pred1_v_s3 - self.beta_s2  # (1,1,64,128)
        maxdisparity_s2 = pred1_s3_cur + (self.gamma_s2 + 1) * pred1_v_s3 + self.beta_s2  # (1,1,64,128)
        maxdisparity_s2 = F.upsample(maxdisparity_s2 * 2, [left.size()[2] // 2, left.size()[3] // 2], mode='bilinear',
                                     align_corners=True)  # (1,1,128,256)
        mindisparity_s2 = F.upsample(mindisparity_s2 * 2, [left.size()[2] // 2, left.size()[3] // 2], mode='bilinear',
                                     align_corners=True)  # (1,1,128,256)

        mindisparity_s2_1, maxdisparity_s2_1 = self.generate_search_range(self.sample_count_s2 + 1, mindisparity_s2,
                                                                          maxdisparity_s2,
                                                                          scale=1)  # (1,1,128,256)and#(1,1,128,256)
        disparity_samples_s2 = self.generate_disparity_samples(mindisparity_s2_1, maxdisparity_s2_1,
                                                               self.sample_count_s2).float()  # (1,12,128,256)
        confidence_v_concat_s2, _ = self.cost_volume_generator(features_left["concat_feature2"],
                                                               features_right["concat_feature2"], disparity_samples_s2,
                                                               'concat')  # (1,12,12,128,256)
        confidence_v_gwc_s2, disparity_samples_s2 = self.cost_volume_generator(features_left["gw2"],
                                                                               features_right["gw2"],
                                                                               disparity_samples_s2, 'gwc',
                                                                               self.num_groups // 2)  # (1,20,12,128,256)and#(1,1,12,128,256)
        confidence_v_s2 = torch.cat((confidence_v_gwc_s2, confidence_v_concat_s2, disparity_samples_s2),
                                    dim=1)  # (1,33,12,128,256)

        disparity_samples_s2 = torch.squeeze(disparity_samples_s2, dim=1)  # (1,12,128,256)

        cost0_s2 = self.confidence0_s2(confidence_v_s2)  # (1,16,12,128,256)
        cost0_s2 = self.confidence1_s2(cost0_s2) + cost0_s2  # (1,16,12,128,256)

        #GCE
        left_fea_1_2 = self.change_80_16(features_left["gw2"])  # (1,32,64,128)
        cost0_s2 = self.channelAtt_1_2(cost0_s2, left_fea_1_2)  # (1,32,16,64,128)  # 这一步就是GCE的关键所在

        out1_s2 = self.confidence2_s2(cost0_s2)  # (1,16,12,128,256)
        out2_s2 = self.confidence3_s2(out1_s2)  # (1,16,12,128,256)

        cost1_s2 = self.confidence_classif1_s2(out2_s2).squeeze(1)  # (1,12,128,256)
        cost1_s2_possibility = F.softmax(cost1_s2, dim=1)  # (1,12,128,256)
        # cost1_s2_possibility = self.refine_module(left, cost1_s2_possibility)
        pred1_s2 = torch.sum(cost1_s2_possibility * disparity_samples_s2, dim=1, keepdim=True)  # (1,1,128,256)

        # pred1_v_s2 = disparity_variance_confidence(cost1_s2_possibility, disparity_samples_s2, pred1_s2)
        # pred1_v_s2 = pred1_v_s2.sqrt()

        if self.training:
            cost0_4 = self.classif0(cost0_4)  # (1,1,24,32,64)#1/8
            cost1_4 = self.classif1(out1_4)  # (1,1,24,32,64)

            cost0_4 = F.upsample(cost0_4, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear',
                                 align_corners=True)  # (1,1,192,256,512)
            cost0_4 = torch.squeeze(cost0_4, 1)  # (1,192,256,512)
            # 开始CSR
            # costr = self.refine_module(left, cost0_4)  # (1,192,256,512)

            pred0_4 = F.softmax(cost0_4, dim=1)  # (1,192,256,512)
            # 输出1，只针对1/8分辨率的一个结果
            pred0_4 = disparity_regression(pred0_4, self.maxdisp)  # (1,256,512)

            cost1_4 = F.upsample(cost1_4, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear',
                                 align_corners=True)  # (1,1,192,256,512)
            cost1_4 = torch.squeeze(cost1_4, 1)  # (1,192,256,512)
            pred1_4 = F.softmax(cost1_4, dim=1)  # (1,192,256,512)
            # 输出2，应该和输出1一样

            pred1_4 = disparity_regression(pred1_4, self.maxdisp)  # (1,256,512)

            pred2_s4 = F.upsample(pred2_s4 * 8, [left.size()[2], left.size()[3]], mode='bilinear',
                                  align_corners=True)  # (1,1,256,512)
            # 输出3，应该是3个cost volumn融合的那里，产生的初始视差，类似于输出2
            # 得到初始视差图，根据初始视差图计算不确定性，调整每个像素的视差搜索范围
            pred2_s4 = torch.squeeze(pred2_s4, 1)  # (1,256,512)

            cost0_s3 = self.confidence_classif0_s3(cost0_s3).squeeze(1)  # (1,16,64,128)
            cost0_s3 = F.softmax(cost0_s3, dim=1)  # (1,16,64,128)
            pred0_s3 = torch.sum(cost0_s3 * disparity_samples_s3, dim=1, keepdim=True)  # (1,1,64,128)
            pred0_s3 = F.upsample(pred0_s3 * 4, [left.size()[2], left.size()[3]], mode='bilinear',
                                  align_corners=True)  # (1,1,256,512)
            # 输出4，特征提取中1/4特征，并且经过什么置信度处理得到
            # 根据 1/4 的特征以及新的平面假设深度构造 cost volume（应该是右图特征根据新的平面假设深度与左图特征对齐），
            # 根据 cost volume 和 新的平面假设深度得到 1/4 的视差图，再根据 1/4 的视差图计算不确定性，调整每个像素的视差搜索范围
            pred0_s3 = torch.squeeze(pred0_s3, 1)  # (1,256,512)

            costmid_s3 = self.confidence_classifmid_s3(out1_s3).squeeze(1)  # (1,16,64,128)
            costmid_s3 = F.softmax(costmid_s3, dim=1)  # (1,16,64,128)
            predmid_s3 = torch.sum(costmid_s3 * disparity_samples_s3, dim=1, keepdim=True)  # (1,1,64,128)
            predmid_s3 = F.upsample(predmid_s3 * 4, [left.size()[2], left.size()[3]], mode='bilinear',
                                    align_corners=True)  # (1,1,256,512)
            # 输出5，跟输出4貌似有什么关联，很接近
            predmid_s3 = torch.squeeze(predmid_s3, 1)  # (1,256,512)

            pred1_s3_up = F.upsample(pred1_s3 * 4, [left.size()[2], left.size()[3]], mode='bilinear',
                                     align_corners=True)  # (1,1,256,512)
            # 输出6，跟输出5貌似有什么关联，很接近
            pred1_s3_up = torch.squeeze(pred1_s3_up, 1)  # (1,256,512)

            cost0_s2 = self.confidence_classif0_s2(cost0_s2).squeeze(1)  # (1,12,128,256)
            cost0_s2 = F.softmax(cost0_s2, dim=1)  # (1,12,128,256)
            pred0_s2 = torch.sum(cost0_s2 * disparity_samples_s2, dim=1, keepdim=True)  # (1,1,128,256)
            pred0_s2 = F.upsample(pred0_s2 * 2, [left.size()[2], left.size()[3]], mode='bilinear',
                                  align_corners=True)  # (1,1,256,512)
            # 输出7，特征提取中1/2特征，并且经过什么置信度处理得到
            # 根据1/2的特征以及新的平面假设深度构造cost volume，根据cost volume和新的平面假设深度得到1/2的视差图
            pred0_s2 = torch.squeeze(pred0_s2, 1)  # (1,256,512)

            costmid_s2 = self.confidence_classifmid_s2(out1_s2).squeeze(1)  # (1,12,128,256)
            costmid_s2 = F.softmax(costmid_s2, dim=1)  # (1,12,128,256)
            predmid_s2 = torch.sum(costmid_s2 * disparity_samples_s2, dim=1, keepdim=True)  # (1,1,128,256)
            predmid_s2 = F.upsample(predmid_s2 * 2, [left.size()[2], left.size()[3]], mode='bilinear',
                                    align_corners=True)  # (1,1,256,512)
            # 输出8，和输出7有什么关联
            predmid_s2 = torch.squeeze(predmid_s2, 1)  # (1,256,512)

            pred1_s2 = F.upsample(pred1_s2 * 2, [left.size()[2], left.size()[3]], mode='bilinear',
                                  align_corners=True)  # (1,1,256,512)
            # 输出9，和输出8有什么关联
            pred1_s2 = torch.squeeze(pred1_s2, 1)  # (1,256,512)

            # 总结一下，一共是9个输出。其实输出1和2是只针对1/8分辨率特征图的一个结果，
            # 输出3是融合cost volumn的输出，
            # 4,5,6针对1/4;
            # 6,7,8针对1/2;

            return [pred0_4, pred1_4, pred2_s4, pred0_s3, predmid_s3, pred1_s3_up, pred0_s2, predmid_s2, pred1_s2]

        else:
            pred2_s4 = F.upsample(pred2_s4 * 8, [left.size()[2], left.size()[3]], mode='bilinear', align_corners=True)
            pred2_s4 = torch.squeeze(pred2_s4, 1)

            pred1_s3_up = F.upsample(pred1_s3 * 4, [left.size()[2], left.size()[3]], mode='bilinear',
                                     align_corners=True)
            pred1_s3_up = torch.squeeze(pred1_s3_up, 1)

            pred1_s2 = F.upsample(pred1_s2 * 2, [left.size()[2], left.size()[3]], mode='bilinear', align_corners=True)
            pred1_s2 = torch.squeeze(pred1_s2, 1)

            return [pred1_s2], [pred1_s3_up], [pred2_s4]


def CFNet_yk(d):
    return cfnet(d, use_concat_volume=True)
