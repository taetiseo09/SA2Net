import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule, build_activation_layer, Scale
from mmcv.cnn.bricks import DropPath

from mmseg.core import add_prefix
from mmseg.ops import resize

from ..losses import accuracy
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from .psp_head import PPM

try:
    from mmcv.ops import CrissCrossAttention
except ModuleNotFoundError:
    CrissCrossAttention = None


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_cfg=dict(type='GELU'), drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = build_activation_layer(act_cfg)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, out_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads  # number of attention heads
        self.dim = dim
        self.out_dim = out_dim
        head_dim = out_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        # linear layers to project the input vectors to q k v, respectively
        self.q_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.k_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.v_map = nn.Linear(dim, out_dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(out_dim, out_dim)  # linear layer for the final output projection
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, v):
        B, N, _ = q.shape  # q:Query vectors of shape (B,N,dim)
        C = self.out_dim
        k = v  # v:Key and Value vectors of shape (B,NK,dim)
        NK = k.size(1)  # NK:number of key-value tokens

        q = self.q_map(q).view(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_map(k).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_map(v).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B,num_heads,N,NK)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, dim_q=None, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_cfg=dict(type='GELU'), norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.self_attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        dim_q = dim_q or dim
        self.norm_q = norm_layer(dim_q)
        self.norm_v = norm_layer(dim)
        self.attn = CrossAttention(
            dim, dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)  # dimx4
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_cfg=act_cfg, drop=drop)

    def forward(self, q, v):
        # q = q + self.drop_path(self.self_attn(self.norm1(q)))
        norm_q = self.norm1(q)
        q_1 = self.self_attn(norm_q)
        q = q + self.drop_path(q_1)

        norm_q = self.norm_q(q)
        norm_v = self.norm_v(v)
        q_2 = self.attn(norm_q, norm_v)
        q = q + self.drop_path(q_2)
        # q = q + self.drop_path(self.attn(self.norm_q(q), self.norm_v(v)))

        q = q + self.drop_path(self.mlp(self.norm2(q)))
        return q


class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim=128, depth=4, num_heads=8,
                 mlp_ratio=4, qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path=0.):
        super().__init__()
        self.decoder = nn.ModuleList([
            DecoderBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path)
            for _ in range(depth)])  # 4层DecoderBlock layers(each head 128/6=16)

    def forward(self, x, q):
        qs = []
        for blk in self.decoder:
            q = blk(q, x)
            qs.append(q)
        return qs


class TransformerPredictor(nn.Module):
    def __init__(self, embed_dim=128, depth=6, num_heads=8, mask_dim=128,
                 drop_rate=0.1, attn_drop_rate=0.1, drop_path=0.1,
                 in_channels_2d=512, num_classes=4):
        super(TransformerPredictor, self).__init__()

        self.num_classes = num_classes
        self.hidden_dim = embed_dim

        self.dec_2d = TransformerDecoder(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path=drop_path
        )

        self.proj_2d = nn.Conv1d(in_channels_2d, self.hidden_dim, kernel_size=1)  # C -> hidden_dim
        c2_xavier_fill(self.proj_2d)

        self.query_embed = nn.Embedding(num_classes, self.hidden_dim)
        self.mask_embed = FFN(self.hidden_dim, self.hidden_dim, mask_dim, 3)

    def forward(self, x_2d):  # B,C,H,W
        tB, C, H, W = x_2d.size()  # tB: actual batch size
        # fB = int(tB // self.num_view)  # self.num_view = ViewNum = TBD
        # tokens = self.num_view * H * W  # total number of 2D tokens
        srcs_2d = x_2d.view(tB, C, H * W)  # tB,C,H*W
        trans_feat_2d = self.proj_2d(srcs_2d).transpose(1, 2).contiguous()  # tB,H*W,hidden_dim
        # trans_input_2d = trans_input_2d.view(self.num_view, fB, H * W, self.hidden_dim)
        # trans_input_2d = trans_input_2d.transpose(0, 1).contiguous()  # fB,ViewNum,H*W,hidden_dim
        # trans_input_2d = trans_input_2d.view(fB, tokens, self.hidden_dim)

        # pos_input = self.pos_embed_2d.expand(tB, -1, -1)  # tB,tokens,hidden_dim
        # pos_input = pos_input.view(self.num_view, fB, H * W, self.hidden_dim)
        # pos_input = pos_input.transpose(0, 1).contiguous()
        # pos_input = pos_input.view(fB, tokens, self.hidden_dim)

        # trans_feat_2d = self.enc_share(trans_input_2d, pos_input)  # fB,tokens,hidden_dim
        # trans_feat_2d = trans_feat_2d.view(fB, self.num_view, H * W, self.hidden_dim)
        # trans_feat_2d = trans_feat_2d.transpose(0, 1).contiguous()
        # trans_feat_2d = trans_feat_2d.view(tB, H * W, self.hidden_dim)

        # decoder: B,H*W,hidden_dim  B,num_classes,hidden_dim
        hs_adain_2d = self.dec_2d(trans_feat_2d, self.query_embed.weight.unsqueeze(dim=0).expand(tB, -1, -1))
        # unsqueeze(dim=0) add an extra dimension at the specified position [1,num_classes,hidden_dim]
        mask_embed = self.mask_embed(hs_adain_2d[-1])  # process the last layer of 'hs_adain_2d'
        return mask_embed  # [tB, num_classes, mask_dim]


def c2_xavier_fill(module: nn.Module) -> None:
    """
    Initialize `module.weight` using the "XavierFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.
    """
    # Caffe2 implementation of XavierFill in fact
    # corresponds to kaiming_uniform_ in PyTorch
    nn.init.kaiming_uniform_(module.weight, a=1)
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)


class FFN(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class CAM(nn.Module):
    def __init__(self, in_channels):
        super(CAM, self).__init__()
        # self.in_channels = in_channels
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels // 8, in_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.GELU())
        self.gamma = Scale(0)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()  # batch_size=4
        query = self.query_conv(x)  # B*C'*H*W, C'=C/8
        key = self.key_conv(x)  # B*C'*H*W
        value = self.value_conv(x)  # B*C'*H*W
        proj_query = query.view(batch_size, channels//8, -1)  # B*C'*HW
        proj_key = key.view(batch_size, channels//8, -1).permute(0, 2, 1)  # B*HW*C'
        energy = torch.bmm(proj_query, proj_key)  # 4*C'*C'
        energy_new = torch.max(
            energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)  # 4*C'*C'
        proj_value = value.view(batch_size, channels//8, -1)  # 4*C'*HW

        out = torch.bmm(attention, proj_value)  # 4*C'*HW
        out = out.view(batch_size, channels//8, height, width)
        out = self.proj(out)

        out = self.gamma(out) + x
        return out


class SASelfAttention(nn.Module):
    def __init__(self,
                 in_channels,
                 basis,
                 ratio=4,
                 embed_dim=128, depth=6, num_heads=8, mask_dim=128
                 ):
        super(SASelfAttention, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = int(in_channels/ratio)
        self.basis = basis

        self.maskformer = TransformerPredictor(
            embed_dim=embed_dim,
            mask_dim=mask_dim,
            depth=depth,
            num_heads=num_heads,
            in_channels_2d=self.in_channels,
            num_classes=self.basis
        )
        self.proj = nn.Conv1d(self.basis, self.inter_channels, kernel_size=1)

        self.theta = ConvModule(  # f -> k
            self.in_channels,
            self.basis,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None
        )
        self.phi = ConvModule(  # f -> q
            self.in_channels,
            self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None
        )
        self.delta = ConvModule(
            self.inter_channels,
            self.in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=None,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=None
        )
        self.ro = ConvModule(
            self.in_channels * 2,
            self.in_channels,
            kernel_size=3,
            stride=1,
            padding=2,
            conv_cfg=None,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU')
        )
        self.gamma = Scale(0)

    def forward(self, x):  # B*C*H*W
        batch_size = x.size(0)  # B=4
        mask = self.maskformer(x)  # B*4*mask_dim
        mask_k = mask.transpose(1, 2).contiguous()  # B*mask_dim*4
        mask_q = self.proj(mask)  # B*C'*mask_dim

        q = self.phi(x)  # C change to C'
        q = q.view(batch_size, self.inter_channels, -1)  # B*C'*(HW)

        k = self.theta(x)  # C change to 4
        lo = k.view(batch_size, self.basis, x.size(2), x.size(3))  # B*4*H*W
        k = k.view(batch_size, self.basis, -1)  # B*4*(HW)
        k_prime = nn.functional.softmax(k, dim=1).permute(0, 2, 1)  # B*(HW)*4

        k_prime = torch.cat([k_prime, mask_k], dim=1)  # B* (HW)+mask_dim *4
        q_prime = torch.cat([q, mask_q], dim=2)  # B*C'* (HW)+mask_dim

        s = torch.matmul(q_prime, k_prime)  # B*C'*4
        s = nn.functional.softmax(s, dim=2)  # B*C'*4

        f = torch.matmul(s, nn.functional.softmax(k, dim=1))  # B*C'*(HW)
        f = f.view(x.size(0), self.inter_channels, x.size(2), x.size(3))  # B*C'*H*W
        x_ = self.delta(f)  # B*C*H*W

        # y = self.gamma(x_) + x
        y = torch.cat((x, x_), dim=1)  # B*2C*H*W
        y = self.ro(y)  # B*C*H*W

        return y, lo


@HEADS.register_module()
class AffiCDUPerHead(BaseDecodeHead):
    """
    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self,
                 pool_scales=(1, 2, 3, 6),
                 recurrence=2,
                 **kwargs):
        super(AffiCDUPerHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],  # the last layer of the network
            self.channels,  # 512
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        # Channel Attention Module
        self.cam = CAM(self.channels)
        self.cam_out_conv = ConvModule(
            self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.cam_conv_seg = nn.Conv2d(
            self.channels, self.num_classes, kernel_size=1)
        # Position Attention Module
        self.recurrence = recurrence
        self.pam = CrissCrossAttention(self.channels)
        self.pam_out_conv = ConvModule(
            self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.pam_conv_seg = nn.Conv2d(
            self.channels, self.num_classes, kernel_size=1)
        # Structure-Affinity Module
        self.semantic_aug = SASelfAttention(in_channels=self.channels, basis=self.num_classes, ratio=4)

    def pam_cls_seg(self, feat):
        """PAM feature classification."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.pam_conv_seg(feat)
        return output

    def cam_cls_seg(self, feat):
        """CAM feature classification."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.cam_conv_seg(feat)
        return output

    def psp_forward(self, inputs):  # Pyramid Pooling Module
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def _forward_feature(self, inputs):
        """Forward function for feature maps before
         classifying each pixel with ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        inputs = self._transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)  # combine with the output of PPM
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        feats = self.fpn_bottleneck(fpn_outs)
        return feats

    def forward(self, inputs):
        """Forward function."""
        out = self._forward_feature(inputs)

        cam_feat = self.cam(out)
        cam_feat = self.cam_out_conv(cam_feat)
        cam_out = self.cam_cls_seg(cam_feat)

        for _ in range(self.recurrence):
            out = self.pam(out)
        pam_feat = self.pam_out_conv(out)
        pam_out = self.pam_cls_seg(pam_feat)

        output = pam_feat + cam_feat
        # output = self.cls_seg(output)
        semantic_aug, seg_lo = self.semantic_aug(output)
        output = self.cls_seg(semantic_aug)
        return output, seg_lo, cam_out, pam_out

    def forward_test(self, inputs, img_metas, test_cfg):
        """
        Forward function for testing, only "output" is used.
        """
        return self.forward(inputs)[0]

    def losses(self, seg_logit, seg_label):
        """Compute segmentation loss."""
        affi_seg_logit = seg_logit[0]
        seg_lo = seg_logit[1]
        cam_seg_logit = seg_logit[2]
        pam_seg_logit = seg_logit[3]

        loss = dict()
        affi_seg_logit = resize(
            input=affi_seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        seg_lo = resize(
            input=seg_lo,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        cam_seg_logit = resize(
            input=cam_seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        pam_seg_logit = resize(
            input=pam_seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)

        loss['loss_seg'] = self.loss_decode(
            affi_seg_logit,
            seg_label,
            weight=seg_weight,
            ignore_index=self.ignore_index) + 0.4 * self.loss_decode(
            seg_lo,
            seg_label,
            weight=seg_weight,
            ignore_index=self.ignore_index) + 0.5 * self.loss_decode(
            cam_seg_logit,
            seg_label,
            weight=seg_weight,
            ignore_index=self.ignore_index) + 0.5 * self.loss_decode(
            pam_seg_logit,
            seg_label,
            weight=seg_weight,
            ignore_index=self.ignore_index)

        loss['acc_seg'] = accuracy(affi_seg_logit, seg_label)

        return loss
