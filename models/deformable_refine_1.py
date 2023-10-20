import torch
import torch.nn as nn
import torch.nn.functional as F
from models.U_net import U_Net, U_Net_F, U_Net_F_v2
from models.mv2_yk import MobileNetV2


class OffsetConv(nn.Module):
    def __init__(self, inc, node_num, modulation):
        super(OffsetConv, self).__init__()
        self.modulation = modulation

        self.p_conv = nn.Conv2d(inc, 2 * node_num, kernel_size=1, padding=0, stride=1)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)

        if modulation:
            self.m_conv = nn.Conv2d(inc, node_num, kernel_size=1, padding=0, stride=1)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

        self.lr_ratio = 1e-2

    def _set_lr(self, module, grad_input, grad_output):
        # print('grad input:', grad_input)
        new_grad_input = []

        for i in range(len(grad_input)):
            if grad_input[i] is not None:
                new_grad_input.append(grad_input[i] * self.lr_ratio)
            else:
                new_grad_input.append(grad_input[i])

        new_grad_input = tuple(new_grad_input)
        # print('new grad input:', new_grad_input)
        return new_grad_input

    def forward(self, x):
        offset = self.p_conv(x)
        B, N, H, W = offset.size()

        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))
        else:
            m = torch.ones(B, N // 2, H, W).cuda()

        return offset, m


class GetValueV2(nn.Module):
    def __init__(self, stride):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(GetValueV2, self).__init__()

        self.stride = stride

    def forward(self, x, offset):
        b, _, h, w = x.size()

        dtype = offset.data.type()
        N = offset.size(1) // 2

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)

        # clip p
        # torch.clamp(input, min, max, out=None) → Tensor
        # clamp（）函数的功能将输入input张量每个元素的值压缩到区间 [min,max]，并返回结果到一个新张量
        p_y = torch.clamp(p[..., :N], 0, h - 1) / (h - 1) * 2 - 1
        p_x = torch.clamp(p[..., N:], 0, w - 1) / (w - 1) * 2 - 1

        x_offset = []
        # torch.nn.functional.grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=None)
        # 这个函数的作用就是给定输入input和网格grid，根据grid中的像素位置从input中取出对应位置的值（可能需要插值）得到输出output
        for i in range(N):
            get_x = F.grid_sample(x, torch.stack((p_x[:, :, :, i], p_y[:, :, :, i]), dim=3), mode='bilinear')
            x_offset.append(get_x)

        x_offset = torch.stack(x_offset, dim=4)

        return x_offset

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        # torch.meshgrid（）的功能是生成网格，可以用于生成坐标。函数输入两个数据类型相同的一维张量，两个输出张量的行数为第一个输入张量的元素个数，列数为第二个输入张量的元素个数，
        # 当两个输入张量数据类型不同或维度不是一维时会报错。
        # 其中第一个输出张量填充第一个输入张量中的元素，各行元素相同；第二个输出张量填充第二个输入张量中的元素各列元素相同
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, N, ks)],
                             dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)

        return x_offset


class DeformableRefine(nn.Module):
    def __init__(self, feature_c, node_n, modulation, cost=False):
        super(DeformableRefine, self).__init__()
        self.refine_cost = cost

        self.feature_net = U_Net(img_ch=3, output_ch=feature_c)

        # self.feature_net = U_Net_v2(img_ch=3, output_ch=feature_c)
        #
        self.offset_conv = OffsetConv(inc=feature_c, node_num=node_n, modulation=modulation)
        self.get_value = GetValueV2(stride=1)

    def forward(self, img, depth):
        if not self.refine_cost:
            depth = depth.unsqueeze(1)

        feature = self.feature_net(img)
        offset, m = self.offset_conv(feature)

        # B, 1, H, W, N or B, D, H, W, N
        depth_offset = self.get_value(depth, offset)

        m = m.unsqueeze(4).transpose(1, 4)
        interpolated_depth = torch.sum(m * depth_offset, dim=4) / (torch.sum(m, dim=4) + 1e-8)

        return interpolated_depth, offset


class DeformableRefineF(nn.Module):
    def __init__(self, feature_c, node_n, modulation, cost=False):
        super(DeformableRefineF, self).__init__()
        self.refine_cost = cost

        # self.feature_net = U_Net_F(img_ch=3, output_ch=feature_c)
        self.feature_net = U_Net_F_v2(img_ch=3, output_ch=feature_c)
        self.mobilenetv2 = MobileNetV2()
        #
        self.offset_conv = OffsetConv(inc=feature_c, node_num=node_n, modulation=modulation)
        self.get_value = GetValueV2(stride=1)

    def forward(self, img, depth):
        if not self.refine_cost:
            depth = depth.unsqueeze(1)

        # feature = self.feature_net(img)  # (1,64,256,512)
        feature = self.mobilenetv2(img)
        offset, m = self.offset_conv(feature)  # 生成一个位移图offset和一个调制图m

        # B, 1, H, W, N or B, D, H, W, N
        # 位移图被用于和输入的深度图去生成视差(深度)的偏移
        depth_offset = self.get_value(depth, offset)  # 这一步也很关键

        m = m.unsqueeze(4)
        m = m.transpose(1, 4)
        # 最终，根据调制图m和深度偏移值depth_offset，模块会计算出一个经过插值处理的深度图，并将其作为模块的输出返回
        interpolated_depth = torch.sum(m * depth_offset, dim=4) / (torch.sum(m, dim=4) + 1e-8)

        return interpolated_depth
