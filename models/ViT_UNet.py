from torch import nn
from models.UNet import Down, DoubleConv, Up, OutConv
from torchvision.models.vision_transformer import VisionTransformer

class ViT_UNet(nn.Module):
    def __init__(self):
        super(ViT_UNet, self).__init__()
        
        self.vit = VisionTransformer(
            image_size=256,
            patch_size=16,
            num_layers=4,
            num_heads=16,
            hidden_dim=1024,
            mlp_dim=2048,
            num_classes=784
        )
        self._initialize_vit_weights()
        self.linear = nn.Linear(784, 1024 * 16 * 16)

        # Encoder
        self.in_conv = DoubleConv(3, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # Decoder
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.out_conv = OutConv(64, 1)
        self.sigmoid = nn.Sigmoid()

    def _initialize_vit_weights(self):
        for name, param in self.vit.named_parameters():
            if 'weight' in name and param.dim() > 1:  # Weights of layers (e.g., Linear, Conv)
                nn.init.xavier_uniform_(param)

    def forward(self, x):
        b = x.size(0)
        z = self.vit(x)
        z = self.linear(z)
        z = z.view(b, 1024, 16, 16)

        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(z, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out_conv(x)
        x = self.sigmoid(x)
        return x

