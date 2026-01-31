import torch
import torch.nn as nn

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        """
        Simple UNet for Image-to-Image translation.
        """
        super().__init__()
        
        def down_block(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2))
            return nn.Sequential(*layers)
            
        def up_block(in_feat, out_feat, dropout=0.0):
            layers = [
                nn.ConvTranspose2d(in_feat, out_feat, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_feat),
                nn.ReLU()
            ]
            if dropout:
                layers.append(nn.Dropout(dropout))
            return nn.Sequential(*layers)
            
        # Encoder
        self.down1 = down_block(in_channels, 64, normalize=False)
        self.down2 = down_block(64, 128)
        self.down3 = down_block(128, 256)
        self.down4 = down_block(256, 512)
        
        # Decoder
        self.up1 = up_block(512, 256)
        self.up2 = up_block(512, 128) # Cat 256+256 = 512
        self.up3 = up_block(256, 64)  # Cat 128+128 = 256
        
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2), 
            nn.Conv2d(128, out_channels, 3, 1, 1), # Cat 64+64 = 128
            nn.Tanh()
        )
        
    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        
        u1 = self.up1(d4)
        u2 = self.up2(torch.cat([u1, d3], 1))
        u3 = self.up3(torch.cat([u2, d2], 1))
        
        out = self.final(torch.cat([u3, d1], 1))
        return out

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3):
        """
        PatchGAN Discriminator.
        """
        super(NLayerDiscriminator, self).__init__()
        
        sequence = [nn.Conv2d(input_nc, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
            
        sequence += [nn.Conv2d(ndf * nf_mult, 1, 4, 1, 1)] # Output 1 channel prediction map
        self.model = nn.Sequential(*sequence)
        
    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    gen = UNetGenerator()
    disc = NLayerDiscriminator()
    x = torch.randn(1, 3, 256, 256)
    print(f"Gen Output: {gen(x).shape}")
    print(f"Disc Output: {disc(x).shape}")
