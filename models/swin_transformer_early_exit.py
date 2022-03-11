from collections import OrderedDict
import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from models.swin_transformer import BasicLayer, PatchEmbed, PatchMerging, SwinTransformer, SwinTransformerBlock


class SwinTransformer_s1(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=128, depths=[2, 2, 10, 0], num_heads=[4, 8, 16, 32],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.5,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.transfer_weights()

    def transfer_weights(self):
        pretrained_model = torch.load('/home/slzhang/projects/Swin-Transformer/swin_base_patch4_window7_224.pth')
        # dict_old = pretrained_model['state_dict']
        # dict_new = OrderedDict()
        # for k, v in dict_old.items():
        #     print(k)
        #     dict_new[k.replace('module.', '')] = v
        self.load_state_dict(pretrained_model['model'], strict=False)


    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        # x = self.norm(x)  # B L C
        # x = self.avgpool(x.transpose(1, 2))  # B C 1
        # x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)
        return x    


class SwinTransformer_s2(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=128, depths=[0, 0, 8, 2], num_heads=[4, 8, 16, 32],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.5,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            if i_layer < 2:
                continue
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.transfer_weights()

    def transfer_weights(self):
        pretrained_model = torch.load('/home/slzhang/projects/Swin-Transformer/swin_base_patch4_window7_224.pth')
        dict_old = pretrained_model['model']
        dict_new = OrderedDict()
        for k, v in self.named_parameters():
            key_new = k.split('.')
            key_new[1] = str(int(key_new[1])+2)
            key_new[3] = str(int(key_new[3])+2)
            key_new = '.'.join(key_new)
            dict_new[k] = dict_old[key_new]
        # dict_new = OrderedDict()
        # for k, v in dict_old.items():
        #     print(k)
        #     dict_new[k.replace('module.', '')] = v
        self.load_state_dict(pretrained_model['model'], strict=False)


    def forward_features(self, x):
        # x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x    

class SwinTransformer_multi_exits(nn.Module):

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, exit_list=[], **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.exit_list = exit_list

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        ori_backbone = SwinTransformer(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32], drop_path_rate=0.5)
        self.backbone = nn.ModuleList()
        self.bridge = nn.ModuleList()
        self.head = nn.ModuleList()

        # for k, v in ori_backbone.named_parameters():
        #     print(k)

        # print('--------------------')

        i_layer = 2
        input_resolution = (patches_resolution[0] // (2 ** i_layer), patches_resolution[1] // (2 ** i_layer))
        dim=int(embed_dim * 2 ** i_layer)
        drop=drop_rate
        attn_drop=attn_drop_rate
        drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])]

        for i in range(len(exit_list)+1):
            backbone = nn.Sequential()
            if i == 0:
                for layer in ori_backbone.layers.named_children():
                    if layer[0] == '0' or layer[0] == '1':
                        backbone.add_module(layer[0], layer[1])

            for j in range(2):
                backbone.add_module('blocks-'+str(j), SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads[i_layer], window_size=window_size,
                                 shift_size=0 if (j % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[j+2*i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer))
            # for layer in ori_backbone.layers[2].blocks.named_children():
            #     if int(layer[0]) < 2*(i+1) and int(layer[0]) >= 2*i:
            #         backbone.add_module('blocks-'+layer[0], layer[1])
            self.backbone.append(backbone)

        for i in range(len(exit_list)+1):
            bridge = nn.Sequential()
            bridge.add_module('downsample', PatchMerging(input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                                            patches_resolution[1] // (2 ** i_layer)),
                                                        dim=int(embed_dim * 2 ** i_layer),
                                                        norm_layer=norm_layer))
            # for layer in ori_backbone.layers[2].downsample.named_children():
            #     print(layer[0])
            #     bridge.add_module(layer[0], layer[1])
            for layer in ori_backbone.layers.named_children():
                if layer[0] == '3':
                    bridge.add_module(layer[0], layer[1])
            self.bridge.append(bridge)

        # for k, v in self.backbone[0].named_parameters():
        #     print(k, v.size())

        # print('--------------------')
        # for k, v in ori_backbone.named_parameters():
        #     print(k, v.size())

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        for i in range(len(exit_list)+1):
            self.head.append(nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity())

        self.transfer_weights()
        del ori_backbone


    def transfer_weights(self):
        pretrained_model = torch.load('/home/slzhang/projects/Swin-Transformer/swin_base_patch4_window7_224.pth')
        dict_old = pretrained_model['model']
        dict_new = OrderedDict()

        # for k,v in self.state_dict().items():
        #     print(k)

        # print('--------------------')

        for k,v in self.state_dict().items():
            split_layer_name = k.split('.')
            if split_layer_name[0] == 'patch_embed' or split_layer_name[0] == 'norm':
                dict_new[k] = dict_old[k]
            elif split_layer_name[0] == 'head':
                dict_new[k] = dict_old[split_layer_name[0]+'.'+split_layer_name[2]]
            elif split_layer_name[2] == '0' or split_layer_name[2] == '1' or split_layer_name[2] == '3':
                dict_new[k] = dict_old['layers.'+split_layer_name[2]+'.'+'.'.join(split_layer_name[3:])]
            elif split_layer_name[3] == 'reduction' or split_layer_name[3] == 'norm':
                dict_new[k] = dict_old['layers.2.downsample.'+'.'.join(split_layer_name[3:])]
            else:
                dict_new[k] = dict_old['layers.2.blocks.'+ str(int(split_layer_name[2].split('-')[1])+2*int(split_layer_name[1])) +'.'+'.'.join(split_layer_name[3:])]

        self.load_state_dict(dict_new, strict=False)
            

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x):
        output = []
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for i in range(len(self.exit_list)+1):
            # print('1', x.size())
            x = self.backbone[i](x)
            # print('2', x.size())
            x_exit = self.bridge[i](x)
            # print('3', x.size())
            x_exit = self.norm(x_exit)  # B L C
            x_exit = self.avgpool(x_exit.transpose(1, 2))  # B C 1
            x_exit = torch.flatten(x_exit, 1)
            x_exit = self.head[i](x_exit)
            output.append(x_exit)

        return output


class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops