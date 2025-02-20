# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange

# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         try:
#             nn.init.xavier_uniform_(m.weight.data)
#             m.bias.data.fill_(0)
#         except AttributeError:
#             print("Skipping initialization of ", classname)


# class GatedActivation(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x):
#         x, y = x.chunk(2, dim=1)
#         return F.tanh(x) * F.sigmoid(y)


# class GatedMaskedConv2d(nn.Module):
#     def __init__(self, mask_type, dim, kernel, residual=True, n_classes=10, bh_model=False):
#         super().__init__()
#         assert kernel % 2 == 1, print("Kernel size must be odd")
#         self.mask_type = mask_type
#         self.residual = residual
#         self.bh_model = bh_model

#         self.class_cond_embedding = nn.Embedding(
#             n_classes, 2 * dim
#         )

#         kernel_shp = (kernel // 2 + 1, 3 if self.bh_model else 1)  # (ceil(n/2), n)
#         padding_shp = (kernel // 2, 1 if self.bh_model else 0)
#         self.vert_stack = nn.Conv2d(
#             dim, dim * 2,
#             kernel_shp, 1, padding_shp
#         )

#         self.vert_to_horiz = nn.Conv2d(2 * dim, 2 * dim, 1)

#         kernel_shp = (1, 2)
#         padding_shp = (0, 1)
#         self.horiz_stack = nn.Conv2d(
#             dim, dim * 2,
#             kernel_shp, 1, padding_shp
#         )

#         self.horiz_resid = nn.Conv2d(dim, dim, 1)

#         self.gate = GatedActivation()

#     def make_causal(self):
#         self.vert_stack.weight.data[:, :, -1].zero_()  # Mask final row
#         self.horiz_stack.weight.data[:, :, :, -1].zero_()  # Mask final column

#     def forward(self, x_v, x_h, h):
#         if self.mask_type == 'A':
#             self.make_causal()

#         h = self.class_cond_embedding(h)
#         h_vert = self.vert_stack(x_v)
#         h_vert = h_vert[:, :, :x_v.size(-2), :]
#         out_v = self.gate(h_vert + h[:, :, None, None])

#         if self.bh_model:
#             h_horiz = self.horiz_stack(x_h)
#             h_horiz = h_horiz[:, :, :, :x_h.size(-1)]
#             v2h = self.vert_to_horiz(h_vert)

#             out = self.gate(v2h + h_horiz + h[:, :, None, None])
#             if self.residual:
#                 out_h = self.horiz_resid(out) + x_h
#             else:
#                 out_h = self.horiz_resid(out)
#         else:
#             if self.residual:
#                 out_v = self.horiz_resid(out_v) + x_v
#             else:
#                 out_v = self.horiz_resid(out_v)
#             out_h = out_v

#         return out_v, out_h

# class CrossAttention(nn.Module):
#     def __init__(self, in_channels, emb_dim, att_dropout=0.0, aropout=0.0):
#         super(CrossAttention, self).__init__()
#         self.emb_dim = emb_dim
#         self.scale = emb_dim ** -0.5

#         self.proj_in = nn.Conv2d(in_channels, emb_dim, kernel_size=1, stride=1, padding=0)

#         self.Wq = nn.Linear(emb_dim, emb_dim)
#         self.Wk = nn.Linear(emb_dim, emb_dim)
#         self.Wv = nn.Linear(emb_dim, emb_dim)

#         self.proj_out = nn.Conv2d(emb_dim, in_channels, kernel_size=1, stride=1, padding=0)

#     def forward(self, x, context, pad_mask=None):
#         '''

#         :param x: [batch_size, c, h, w]
#         :param context: [batch_szie, seq_len, emb_dim]
#         :param pad_mask: [batch_size, seq_len, seq_len]
#         :return:
#         '''
#         b, c, h, w = x.shape

#         x = self.proj_in(x)   # [batch_size, c, h, w] = [3, 512, 512, 512]
#         x = rearrange(x, 'b c h w -> b (h w) c')   # [batch_size, h*w, c] = [3, 262144, 512]

#         Q = self.Wq(x)  # [batch_size, h*w, emb_dim] = [3, 262144, 512]
#         K = self.Wk(context)  # [batch_szie, seq_len, emb_dim] = [3, 5, 512]
#         V = self.Wv(context)

#         # [batch_size, h*w, seq_len]
#         att_weights = torch.einsum('bid,bjd -> bij', Q, K)
#         att_weights = att_weights * self.scale

#         if pad_mask is not None:
#             # [batch_size, h*w, seq_len]
#             att_weights = att_weights.masked_fill(pad_mask, -1e9)

#         att_weights = F.softmax(att_weights, dim=-1)
#         out = torch.einsum('bij, bjd -> bid', att_weights, V)   # [batch_size, h*w, emb_dim]

#         out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)   # [batch_size, c, h, w]
#         out = self.proj_out(out)   # [batch_size, c, h, w]

#         print(out.shape)

#         return out, att_weights

# class GatedPixelCNN(nn.Module):
#     def __init__(self, input_dim=256, dim=64, n_layers=15, n_classes=10, audio=False, bh_model=False):
#         super().__init__()
#         self.dim = dim
#         self.audio = audio
#         self.bh_model = bh_model

#         if self.audio:
#             self.embedding_aud = nn.Conv2d(256, dim, 1, 1, padding=0)
#             self.fusion_v = nn.Conv2d(dim * 2, dim, 1, 1, padding=0)
#             self.fusion_h = nn.Conv2d(dim * 2, dim, 1, 1, padding=0)

#         # Create embedding layer to embed input
#         self.embedding = nn.Embedding(input_dim, dim)

#         # Building the PixelCNN layer by layer
#         self.layers = nn.ModuleList()

#         # Initial block with Mask-A convolution
#         # Rest with Mask-B convolutions
#         for i in range(n_layers):
#             mask_type = 'A' if i == 0 else 'B'
#             kernel = 7 if i == 0 else 3
#             residual = False if i == 0 else True

#             self.layers.append(
#                 GatedMaskedConv2d(mask_type, dim, kernel, residual, n_classes, bh_model)
#             )

#         # Add the output layer
#         self.output_conv = nn.Sequential(
#             nn.Conv2d(dim, 512, 1),
#             nn.ReLU(True),
#             nn.Conv2d(512, input_dim, 1)
#         )

#         self.cross_attn = CrossAttention(in_channels=dim, emb_dim=256)  # 添加交叉注意力模块

#         self.apply(weights_init)

#         self.dp = nn.Dropout(0.1)

#     def forward(self, x, label, aud=None):
#         shp = x.size() + (-1,)
#         x = self.embedding(x.view(-1)).view(shp)  # (B, H, W, C)
#         x = x.permute(0, 3, 1, 2)  # (B, C, W, W)

#         x_v, x_h = (x, x)
#         for i, layer in enumerate(self.layers):
#             if i == 1 and self.audio is True:
#                 aud = self.embedding_aud(aud)
#                 a = torch.ones(aud.shape[-2]).to(aud.device)
#                 a = self.dp(a)
#                 aud = (aud.transpose(-1, -2) * a).transpose(-1, -2)
#                 x_v = self.fusion_v(torch.cat([x_v, aud], dim=1))
#                 if self.bh_model:
#                     x_h = self.fusion_h(torch.cat([x_h, aud], dim=1))
#             x_v, x_h = layer(x_v, x_h, label)

#         # 在这里将 label 和 aud_feat 输入到交叉注意力模块中
#         if self.audio:
#             aud_feat = aud_feat.view(aud_feat.size(0), aud_feat.size(1), -1)
#             x_v, _ = self.cross_attn(x_v, aud_feat)
#             x_h, _ = self.cross_attn(x_h, aud_feat)

#         if self.bh_model:
#             return self.output_conv(x_h)
#         else:
#             return self.output_conv(x_v)

#     def generate(self, label, shape=(8, 8), batch_size=64, aud_feat=None, pre_latents=None, pre_audio=None):
#         param = next(self.parameters())
#         x = torch.zeros(
#             (batch_size, *shape),
#             dtype=torch.int64, device=param.device
#         )
#         if pre_latents is not None:
#             x = torch.cat([pre_latents, x], dim=1)
#             aud_feat = torch.cat([pre_audio, aud_feat], dim=2)
#             h0 = pre_latents.shape[1]
#             h = h0 + shape[0]
#         else:
#             h0 = 0
#             h = shape[0]

#         for i in range(h0, h):
#             for j in range(shape[1]):
#                 if self.audio:
#                     logits = self.forward(x, label, aud_feat)
#                 else:
#                     logits = self.forward(x, label)
#                 probs = F.softmax(logits[:, :, i, j], -1)
#                 x.data[:, i, j].copy_(
#                     probs.multinomial(1).squeeze().data
#                 )
#         return x[:, h0:h]




import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, in_channels, emb_dim, att_dropout=0.0, aropout=0.0):
        super(CrossAttention, self).__init__()
        self.emb_dim = emb_dim
        self.scale = emb_dim ** -0.5

        self.proj_in = nn.Conv2d(in_channels, emb_dim, kernel_size=1, stride=1, padding=0)

        self.Wq = nn.Linear(emb_dim, emb_dim)
        self.Wk = nn.Linear(emb_dim, emb_dim)
        self.Wv = nn.Linear(emb_dim, emb_dim)

        self.proj_out = nn.Conv2d(emb_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, context, pad_mask=None):
        '''

        :param x: [batch_size, c, h, w]
        :param context: [batch_szie, seq_len, emb_dim]
        :param pad_mask: [batch_size, seq_len, seq_len]
        :return:
        '''
        b, c, h, w = x.shape

        x = self.proj_in(x)   # [batch_size, c, h, w] = [3, 512, 512, 512]
        x = rearrange(x, 'b c h w -> b (h w) c')   # [batch_size, h*w, c] = [3, 262144, 512]

        Q = self.Wq(x)  # [batch_size, h*w, emb_dim] = [3, 262144, 512]
        K = self.Wk(context)  # [batch_szie, seq_len, emb_dim] = [3, 5, 512]
        V = self.Wv(context)

        # [batch_size, h*w, seq_len]
        att_weights = torch.einsum('bid,bjd -> bij', Q, K)
        att_weights = att_weights * self.scale

        if pad_mask is not None:
            # [batch_size, h*w, seq_len]
            att_weights = att_weights.masked_fill(pad_mask, -1e9)

        att_weights = F.softmax(att_weights, dim=-1)
        out = torch.einsum('bij, bjd -> bid', att_weights, V)   # [batch_size, h*w, emb_dim]

        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)   # [batch_size, c, h, w]
        out = self.proj_out(out)   # [batch_size, c, h, w]

        print(out.shape)

        return out, att_weights


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)


class GatedActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x, y = x.chunk(2, dim=1)
        return F.tanh(x) * F.sigmoid(y)


class GatedMaskedConv2d(nn.Module):
    def __init__(self, mask_type, dim, kernel, residual=True, n_classes=10, bh_model=False):
        super().__init__()
        assert kernel % 2 == 1, print("Kernel size must be odd")
        self.mask_type = mask_type
        self.residual = residual
        self.bh_model = bh_model

        self.class_cond_embedding = nn.Embedding(
            n_classes, 2 * dim
        )

        kernel_shp = (kernel // 2 + 1, 3 if self.bh_model else 1)  # (ceil(n/2), n)
        padding_shp = (kernel // 2, 1 if self.bh_model else 0)
        self.vert_stack = nn.Conv2d(
            dim, dim * 2,
            kernel_shp, 1, padding_shp
        )

        self.vert_to_horiz = nn.Conv2d(2 * dim, 2 * dim, 1)

        kernel_shp = (1, 2)
        padding_shp = (0, 1)
        self.horiz_stack = nn.Conv2d(
            dim, dim * 2,
            kernel_shp, 1, padding_shp
        )

        self.horiz_resid = nn.Conv2d(dim, dim, 1)

        self.gate = GatedActivation()

    def make_causal(self):
        self.vert_stack.weight.data[:, :, -1].zero_()  # Mask final row
        self.horiz_stack.weight.data[:, :, :, -1].zero_()  # Mask final column

    def forward(self, x_v, x_h, h, aud_feat):
        if self.mask_type == 'A':
            self.make_causal()

        h = self.class_cond_embedding(h)
        h_vert = self.vert_stack(x_v)
        h_vert = h_vert[:, :, :x_v.size(-2), :]
        out_v = self.gate(h_vert + h[:, :, None, None])

        if self.bh_model:
            h_horiz = self.horiz_stack(x_h)
            h_horiz = h_horiz[:, :, :, :x_h.size(-1)]
            v2h = self.vert_to_horiz(h_vert)

            out = self.gate(v2h + h_horiz + h[:, :, None, None])
            if self.residual:
                out_h = self.horiz_resid(out) + x_h
            else:
                out_h = self.horiz_resid(out)
        else:
            if self.residual:
                out_v = self.horiz_resid(out_v) + x_v
            else:
                out_v = self.horiz_resid(out_v)
            out_h = out_v

        return out_v, out_h



class GatedPixelCNN(nn.Module):
    def __init__(self, input_dim=256, dim=64, n_layers=15, n_classes=10, audio=False, bh_model=False):
        super().__init__()
        self.dim = dim
        self.audio = audio
        self.bh_model = bh_model

        if self.audio:
            self.embedding_aud = nn.Conv2d(256, dim, 1, 1, padding=0)
            self.fusion_v = nn.Conv2d(dim * 2, dim, 1, 1, padding=0)
            self.fusion_h = nn.Conv2d(dim * 2, dim, 1, 1, padding=0)

        # 创建用于嵌入输入的嵌入层
        self.embedding = nn.Embedding(input_dim, dim)

        # 逐层构建 PixelCNN
        self.layers = nn.ModuleList()

        # 使用 Mask-A 卷积的初始块
        # 使用 Mask-B 卷积的其余部分
        for i in range(n_layers):
            mask_type = 'A' if i == 0 else 'B'
            kernel = 7 if i == 0 else 3
            residual = False if i == 0 else True

            self.layers.append(
                GatedMaskedConv2d(mask_type, dim, kernel, residual, n_classes, bh_model)
            )

        # 添加输出层
        self.output_conv = nn.Sequential(
            nn.Conv2d(dim, 512, 1),
            nn.ReLU(True),
            nn.Conv2d(512, input_dim, 1)
        )

        self.apply(weights_init)

        self.dp = nn.Dropout(0.1)

        # 添加交叉注意力机制
        self.cross_attn = CrossAttention(in_channels=dim, emb_dim=128)  # 根据需要调整 emb_dim 的值

    def forward(self, x, label, aud_feat=None):
        shp = x.size() + (-1,)
        x = self.embedding(x.view(-1)).view(shp)  # (B, H, W, C)
        x = x.permute(0, 3, 1, 2)  # (B, C, W, W)

        x_v, x_h = (x, x)
        for i, layer in enumerate(self.layers):
            if i == 1 and self.audio is True:
                aud = self.embedding_aud(aud_feat)
                a = torch.ones(aud.shape[-2]).to(aud.device)
                a = self.dp(a)
                aud = (aud.transpose(-1, -2) * a).transpose(-1, -2)
                x_v = self.fusion_v(torch.cat([x_v, aud], dim=1))
                if self.bh_model:
                    x_h = self.fusion_h(torch.cat([x_h, aud], dim=1))
            # 在每一层的前向传播中，将label和aud_feat传入交叉注意力机制
            x_v, x_h = layer(x_v, x_h, label, aud_feat)

        if self.bh_model:
            return self.output_conv(x_h)
        else:
            return self.output_conv(x_v)



    def generate(self, label, shape=(8, 8), batch_size=64, aud_feat=None, pre_latents=None, pre_audio=None):
        param = next(self.parameters())
        x = torch.zeros(
            (batch_size, *shape),
            dtype=torch.int64, device=param.device
        )
        if pre_latents is not None:
            x = torch.cat([pre_latents, x], dim=1)
            aud_feat = torch.cat([pre_audio, aud_feat], dim=2)
            h0 = pre_latents.shape[1]
            h = h0 + shape[0]
        else:
            h0 = 0
            h = shape[0]

        for i in range(h0, h):
            for j in range(shape[1]):
                if self.audio:
                    logits = self.forward(x, label, aud_feat)
                else:
                    logits = self.forward(x, label)
                probs = F.softmax(logits[:, :, i, j], -1)
                x.data[:, i, j].copy_(
                    probs.multinomial(1).squeeze().data
                )
        return x[:, h0:h]










####################################################################################3
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         try:
#             nn.init.xavier_uniform_(m.weight.data)
#             m.bias.data.fill_(0)
#         except AttributeError:
#             print("Skipping initialization of ", classname)

# class GatedActivation(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x):
#         x, y = x.chunk(2, dim=1)
#         return F.tanh(x) * F.sigmoid(y)

# class ConditionalBatchNorm2d(nn.Module):
#     def __init__(self, num_features, num_classes):
#         super().__init__()
#         self.num_features = num_features
#         self.bn = nn.BatchNorm2d(num_features, affine=False)
#         self.embed = nn.Embedding(num_classes, num_features * 2)
#         self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialize scale at N(1, 0.02)
#         self.embed.weight.data[:, num_features:].zero_()  # Initialize bias at 0

#     def forward(self, x, y):
#         out = self.bn(x)
#         gamma, beta = self.embed(y).chunk(2, 1)
#         out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
#         return out

# class CrossAttention(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(CrossAttention, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x, y):
#         # x: (B, C1, H, W), y: (B, C2)
#         # Project y to match x's channels
#         proj_y = self.conv1(y.unsqueeze(-1).unsqueeze(-1))  # (B, C1, 1, 1)
#         proj_y = proj_y.expand_as(x)  # (B, C1, H, W)

#         # Concatenate x and projected y
#         concat = torch.cat([x, proj_y], dim=1)  # (B, C1+C2, H, W)

#         # Apply convolutions
#         out = self.conv2(concat)  # (B, C_out, H, W)

#         # Spatial softmax to compute attention weights
#         attention_weights = self.softmax(out)

#         # Apply attention weights to x
#         attended_x = x * attention_weights

#         return attended_x

# class GatedMaskedConv2d(nn.Module):
#     def __init__(self, mask_type, dim, kernel, residual=True, n_classes=10, bh_model=False):
#         super().__init__()
#         assert kernel % 2 == 1, print("Kernel size must be odd")
#         self.mask_type = mask_type
#         self.residual = residual
#         self.bh_model = bh_model
#         # self.conv = nn.Conv2d(256, 256, 1)

#         # Define attention mechanism
#         self.cross_attention = CrossAttention(in_channels=dim, out_channels=dim)

#         # self.conv = nn.Conv2d(256, 1280, 1)
#         self.conv = nn.Conv2d(256, 256, 1)




#         self.bn = ConditionalBatchNorm2d(dim * 2, n_classes)

#         kernel_shp = (kernel // 2 + 1, 3 if self.bh_model else 1)  # (ceil(n/2), n)
#         padding_shp = (kernel // 2, 1 if self.bh_model else 0)
#         self.vert_stack = nn.Conv2d(
#             768, dim * 2,
#             kernel_shp, 1, padding_shp
#         )

#         self.vert_to_horiz = nn.Conv2d(2 * dim, 2 * dim, 1)

#         kernel_shp = (1, 2)
#         padding_shp = (0, 1)
#         self.horiz_stack = nn.Conv2d(
#             dim, dim * 2,
#             kernel_shp, 1, padding_shp
#         )

#         self.horiz_resid = nn.Conv2d(dim, dim, 1)

#         self.gate = GatedActivation()

#         # Add a new conv layer to change the channel number of x_h
#         self.conv = nn.Conv2d(1280, 256, 1)

#     def make_causal(self):
#         self.vert_stack.weight.data[:, :, -1].zero_()  # Mask final row
#         self.horiz_stack.weight.data[:, :, :, -1].zero_()  # Mask final column

#     def forward(self, x_v, x_h, h, aud_feat):
#         if self.mask_type == 'A':
#             self.make_causal()

#         # Change the channel number of x_h from 1280 to 256
#         x_h = self.conv(x_h)

#         # Apply cross-attention mechanism
#         attended_x_v = self.cross_attention(x_v, aud_feat)

#         h_vert = self.vert_stack(attended_x_v)
#         h_vert = self.bn(h_vert, h)  # Apply conditional batch norm
#         h_vert = h_vert[:, :, :x_v.size(-2), :]
#         out_v = self.gate(h_vert)

#         if self.bh_model:
#             h_horiz = self.horiz_stack(x_h)
#             h_horiz = h_horiz[:, :, :, :x_h.size(-1)]
#             v2h = self.vert_to_horiz(h_vert)

#             out = self.gate(v2h + h_horiz)
#             if self.residual:
#                 out_h = self.horiz_resid(out) + x_h
#             else:
#                 out_h = self.horiz_resid(out)
#         else:
#             if self.residual:
#                 out_v = self.horiz_resid(out_v) + x_v
#             else:
#                 out_v = self.horiz_resid(out_v)
#             out_h = out_v

#         return out_v, out_h

# class GatedPixelCNN(nn.Module):
#     def __init__(self, input_dim=256, dim=64, n_layers=15, n_classes=10, audio=False, bh_model=False):
#         super().__init__()
#         self.dim = dim
#         self.audio = audio
#         self.bh_model = bh_model
#         self.embedding = nn.Embedding(input_dim, dim)

#         if self.audio:
#             self.embedding_aud = nn.Conv2d(256, dim, 1, 1, padding=0)
#             self.fusion_v = nn.Conv2d(dim * 2, dim, 1, 1, padding=0)
#             self.fusion_h = nn.Conv2d(dim * 2, dim, 1, 1, padding=0)

#         # Create embedding layer to embed input
#         self.embedding = nn.Embedding(input_dim, dim)

#         # Building the PixelCNN layer by layer
#         self.layers = nn.ModuleList()

#         # Initial block with Mask-A convolution
#         # Rest with Mask-B convolutions
#         for i in range(n_layers):
#             mask_type = 'A' if i == 0 else 'B'
#             kernel = 7 if i == 0 else 3
#             residual = False if i == 0 else True

#             self.layers.append(
#                 GatedMaskedConv2d(mask_type, dim, kernel, residual, n_classes, bh_model)
#             )

#         # Add the output layer
#         self.output_conv = nn.Sequential(
#             nn.Conv2d(dim, 512, 1),
#             nn.ReLU(True),
#             nn.Conv2d(512, input_dim, 1)
#         )

#         self.apply(weights_init)

#         self.dp = nn.Dropout(0.1)

#     def forward(self, x, label, aud=None):
#         shp = x.size() + (-1,)
#         x = self.embedding(x.view(-1)).view(shp)  # (B, H, W, C)
#         x = x.permute(0, 3, 1, 2)  # (B, C, W, W)

#         x_v, x_h = (x, x)
#         for i, layer in enumerate(self.layers):
#             if i == 1 and self.audio is True:
#                 aud = self.embedding_aud(aud)
#                 a = torch.ones(aud.shape[-2]).to(aud.device)
#                 a = self.dp(a)
#                 aud = (aud.transpose(-1, -2) * a).transpose(-1, -2)
#                 x_v = self.fusion_v(torch.cat([x_v, aud], dim=1))
#                 if self.bh_model:
#                     x_h = self.fusion_h(torch.cat([x_h, aud], dim=1))
#             x_v, x_h = layer(x_v, x_h, label, aud_feat=aud)

#         if self.bh_model:
#             return self.output_conv(x_h)
#         else:
#             return self.output_conv(x_v)

#     def generate(self, label, shape=(8, 8), batch_size=64, aud_feat=None, pre_latents=None, pre_audio=None):
#         param = next(self.parameters())
#         x = torch.zeros(
#             (batch_size, *shape),
#             dtype=torch.int64, device=param.device
#         )
#         if pre_latents is not None:
#             x = torch.cat([pre_latents, x], dim=1)
#             aud_feat = torch.cat([pre_audio, aud_feat], dim=2)
#             h0 = pre_latents.shape[1]
#             h = h0 + shape[0]
#         else:
#             h0 = 0
#             h = shape[0]

#         for i in range(h0, h):
#             for j in range(shape[1]):
#                 if self.audio:
#                     logits = self.forward(x, label, aud_feat)
#                 else:
#                     logits = self.forward(x, label)
#                 probs = F.softmax(logits[:, :, i, j], -1)
#                 x.data[:, i, j].copy_(
#                     probs.multinomial(1).squeeze().data
#                 )
#         return x[:, h0:h]
########################################################训练
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         try:
#             nn.init.xavier_uniform_(m.weight.data)
#             m.bias.data.fill_(0)
#         except AttributeError:
#             print("Skipping initialization of ", classname)

# class CrossAttention(nn.Module):

#     def __init__(self, dim, heads=8):
#         super().__init__()
#         self.fc_aud_feat = nn.Linear(512, 256)
#         self.fc_label = nn.Linear(22528, 256)
#         self.attn = nn.MultiheadAttention(dim, heads)

#     def forward(self, aud_feat, label):
#         # print(aud_feat.shape, label.shape)
#         aud_feat = aud_feat.permute(2, 0, 1, 3).contiguous()
#         aud_feat = aud_feat.view(22, 128, -1)  # reshape to [seq_len, batch, embed_dim]
#         label = label.reshape(1, 128, -1)  # reshape label to [seq_len, batch, embed_dim]  # reshape label to [seq_len, batch, embed_dim]
#         label = label.expand(22, -1, -1)  # make sure label has the same seq_len as aud_feat
#         aud_feat = self.fc_aud_feat(aud_feat)
#         label = self.fc_label(label)
#         # print(aud_feat.shape, label.shape) 
#         out, _ = self.attn(label, aud_feat, aud_feat)
#         return out.reshape(128, 256, -1) # reshape back to original shape


# class GatedActivation(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x):
#         x, y = x.chunk(2, dim=1)
#         return F.tanh(x) * F.sigmoid(y)

# class GatedMaskedConv2d(nn.Module):
#     def __init__(self, mask_type, dim, kernel, residual=True, n_classes=10, bh_model=False):
#         super().__init__()
#         assert kernel % 2 == 1, print("Kernel size must be odd")
#         self.mask_type = mask_type
#         self.residual = residual
#         self.bh_model = bh_model
#         self.cross_attn = CrossAttention(dim)
#         self.class_cond_embedding = nn.Embedding(
#             n_classes, 2 * dim
#         )

#         kernel_shp = (kernel // 2 + 1, 3 if self.bh_model else 1)  # (ceil(n/2), n)
#         padding_shp = (kernel // 2, 1 if self.bh_model else 0)
#         self.vert_stack = nn.Conv2d(
#             dim, dim * 2,
#             kernel_shp, 1, padding_shp
#         )

#         self.vert_to_horiz = nn.Conv2d(2 * dim, 2 * dim, 1)

#         kernel_shp = (1, 2)
#         padding_shp = (0, 1)
#         self.horiz_stack = nn.Conv2d(
#             dim, dim * 2,
#             kernel_shp, 1, padding_shp
#         )

#         self.horiz_resid = nn.Conv2d(dim, dim, 1)

#         self.gate = GatedActivation()

#     def make_causal(self):
#         self.vert_stack.weight.data[:, :, -1].zero_()  # Mask final row
#         self.horiz_stack.weight.data[:, :, :, -1].zero_()  # Mask final column

#     def forward(self, x_v, x_h, h, aud_feat):
#         if self.mask_type == 'A':
#             self.make_causal()

#         h = self.class_cond_embedding(h)
#         h = h.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x_v.size(-2), x_v.size(-1))
#         h_vert = self.vert_stack(x_v)
#         h_vert = h_vert[:, :, :x_v.size(-2), :]
#         out_v = self.gate(h_vert + h)
#         aud_feat = self.cross_attn(aud_feat, h)
#         # print(aud_feat.shape, out_v.shape)
#         if self.bh_model:
#             h_horiz = self.horiz_stack(x_h)
#             h_horiz = h_horiz[:, :, :, :x_h.size(-1)]
#             v2h = self.vert_to_horiz(h_vert)

#             out = self.gate(v2h + h_horiz + h)
#             if self.residual:
#                 out_h = self.horiz_resid(out) + x_h
#             else:
#                 out_h = self.horiz_resid(out)
#         else:
#             if self.residual:
#                 out_v = self.horiz_resid(out_v) + x_v
#             else:
#                 out_v = self.horiz_resid(out_v)
#             out_h = out_v

#         return out_v, out_h

# class GatedPixelCNN(nn.Module):
#     def __init__(self, input_dim=256, dim=64, n_layers=15, n_classes=10, audio=False, bh_model=False):
#         super().__init__()
#         self.dim = dim
#         self.audio = audio
#         self.bh_model = bh_model

#         if self.audio:
#             self.embedding_aud = nn.Conv2d(256, dim, 1, 1, padding=0)
#             self.fusion_v = nn.Conv2d(dim * 2, dim, 1, 1, padding=0)
#             self.fusion_h = nn.Conv2d(dim * 2, dim, 1, 1, padding=0)

#         # Create embedding layer to embed input
#         self.embedding = nn.Embedding(input_dim, dim)

#         # Building the PixelCNN layer by layer
#         self.layers = nn.ModuleList()

#         # Initial block with Mask-A convolution
#         # Rest with Mask-B convolutions
#         for i in range(n_layers):
#             mask_type = 'A' if i == 0 else 'B'
#             kernel = 7 if i == 0 else 3
#             residual = False if i == 0 else True

#             self.layers.append(
#                 GatedMaskedConv2d(mask_type, dim, kernel, residual, n_classes, bh_model)
#             )

#         # Add the output layer
#         self.output_conv = nn.Sequential(
#             nn.Conv2d(dim, 512, 1),
#             nn.ReLU(True),
#             nn.Conv2d(512, input_dim, 1)
#         )

#         self.apply(weights_init)

#         self.dp = nn.Dropout(0.1)

#     def forward(self, x, label, aud_feat=None):
#         shp = x.size() + (-1,)
#         x = self.embedding(x.view(-1)).view(shp)  # (B, H, W, C)
#         x = x.permute(0, 3, 1, 2)  # (B, C, W, W)

#         x_v, x_h = (x, x)
#         for i, layer in enumerate(self.layers):
#             x_v, x_h = layer(x_v, x_h, label, aud_feat)
#             # if i == 1 and self.audio is True:
#             #     aud = self.embedding_aud(aud_feat)
#             #     a = torch.ones(aud.shape[-2]).to(aud.device)
#             #     a = self.dp(a)
#             #     aud = (aud.transpose(-1, -2) * a).transpose(-1, -2)
#             #     x_v = self.fusion_v(torch.cat([x_v, aud], dim=1))
#             #     if self.bh_model:
#             #         x_h = self.fusion_h(torch.cat([x_h, aud], dim=1))
#             # x_v, x_h = layer(x_v, x_h, label, aud_feat)

#         if self.bh_model:
#             return self.output_conv(x_h)
#         else:
#             return self.output_conv(x_v)

#     def generate(self, label, shape=(8, 8), batch_size=64, aud_feat=None, pre_latents=None, pre_audio=None):
#         param = next(self.parameters())
#         x = torch.zeros(
#             (batch_size, *shape),
#             dtype=torch.int64, device=param.device
#         )
#         if pre_latents is not None:
#             x = torch.cat([pre_latents, x], dim=1)
#             aud_feat = torch.cat([pre_audio, aud_feat], dim=2)
#             h0 = pre_latents.shape[1]
#             h = h0 + shape[0]
#         else:
#             h0 = 0
#             h = shape[0]

#         for i in range(h0, h):
#             for j in range(shape[1]):
#                 if self.audio:
#                     logits = self.forward(x, label, aud_feat)
#                 else:
#                     logits = self.forward(x, label)
#                 probs = F.softmax(logits[:, :, i, j], -1)
#                 x.data[:, i, j].copy_(
#                     probs.multinomial(1).squeeze().data
#                 )
#         return x[:, h0:h]