import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_conv, out_conv, kernel_size=3, padding=1):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_conv, out_conv, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_conv),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_conv, out_conv):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_conv, out_conv, kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        return self.deconv(x)


class UNETR(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_size = 256
        self.num_channels = 3
        self.num_layers = 12
        self.hidden_dim = 784
        self.mlp_dim = 2048
        self.num_heads = 16
        self.num_patches = 256
        self.patch_size = 16

        # Patch embeddings
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.patch_dim = self.patch_size * self.patch_size * self.num_channels
        self.patch_embed = nn.Linear(self.patch_dim, self.hidden_dim)

        # Position embeddings
        self.positions = torch.arange(start=0, end=self.num_patches, step=1, dtype=torch.int32)
        self.pos_embed = nn.Embedding(self.num_patches, self.hidden_dim)

        # Transformer encoder layers
        self.trans_encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=self.num_heads,
                dim_feedforward=self.mlp_dim,
                activation=nn.GELU(),
                batch_first=True
            ) for _ in range(self.num_layers)
        ])

        # Decoders (d), Skip Connections (s) and Convolutions (c)
        self.d1 = DeconvBlock(self.hidden_dim, 512)
        self.s1 = nn.Sequential(
            DeconvBlock(self.hidden_dim, 512),
            ConvBlock(512, 512))
        self.c1 = nn.Sequential(
            ConvBlock(512*2, 512),
            ConvBlock(512, 512))

        self.d2 = DeconvBlock(512, 256)
        self.s2 = nn.Sequential(
            DeconvBlock(self.hidden_dim, 256),
            ConvBlock(256, 256),
            DeconvBlock(256, 256),
            ConvBlock(256, 256))
        self.c2 = nn.Sequential(
            ConvBlock(256*2, 256),
            ConvBlock(256, 256))

        self.d3 = DeconvBlock(256, 128)
        self.s3 = nn.Sequential(
            DeconvBlock(self.hidden_dim, 128),
            ConvBlock(128, 128),
            DeconvBlock(128, 128),
            ConvBlock(128, 128),
            DeconvBlock(128, 128),
            ConvBlock(128, 128))
        self.c3 = nn.Sequential(
            ConvBlock(128*2, 128),
            ConvBlock(128, 128))

        self.d4 = DeconvBlock(128, 64)
        self.s4 = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 64))
        self.c4 = nn.Sequential(
            ConvBlock(64*2, 64),
            ConvBlock(64, 64))

        # Output
        self.output = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        # Extract patches
        b, c, h, w = input.size()

        patches = input.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(b, c, -1, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous()
        patches = patches.view(b, self.num_patches, -1)

        # Embed patches
        patch_embed = self.patch_embed(patches)

        # Positional embeddings
        pos_embed = self.pos_embed(self.positions.to(input.device))
        x = patch_embed + pos_embed

        # Transformer encoder
        skip_connection_index = [3, 6, 9, 12]
        skip_connections = []

        for i in range(self.num_layers):
            layer = self.trans_encoder_layers[i]
            x = layer(x)

            if (i+1) in skip_connection_index:
                skip_connections.append(x)

        z3, z6, z9, z12 = skip_connections

        ## Reshaping for skip connections
        z0 = input.view((b, self.num_channels, self.image_size, self.image_size))
        z3 = z3.view((b, self.hidden_dim, self.patch_size, self.patch_size))
        z6 = z6.view((b, self.hidden_dim, self.patch_size, self.patch_size))
        z9 = z9.view((b, self.hidden_dim, self.patch_size, self.patch_size))
        z12 = z12.view((b, self.hidden_dim, self.patch_size, self.patch_size))

        ## Decoding
        x = self.d1(z12)
        s = self.s1(z9)
        x = torch.cat([x, s], dim=1)
        x = self.c1(x)

        x = self.d2(x)
        s = self.s2(z6)
        x = torch.cat([x, s], dim=1)
        x = self.c2(x)

        x = self.d3(x)
        s = self.s3(z3)
        x = torch.cat([x, s], dim=1)
        x = self.c3(x)

        x = self.d4(x)
        s = self.s4(z0)
        x = torch.cat([x, s], dim=1)
        x = self.c4(x)

        # Output
        x = self.output(x)
        output = self.sigmoid(x)
        return output
