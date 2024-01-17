import torch
import torch.nn as nn
class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
            #nn.Dropout2d(p=0.2)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to modify the number of channels
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                DoubleConv(in_channels, out_channels),
                #nn.Dropout2d(p=0.2)
            )
        else:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                #nn.Dropout2d(p=0.2)
            )

    def forward(self, x):
        return self.up(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class CompressConv(nn.Module):
    def __init__(self, in_channels, compress_channel):
        super(CompressConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, compress_channel, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class DeCompressConv(nn.Module):
    def __init__(self, compress_channel, out_channels):
        super(DeCompressConv, self).__init__()
        self.conv = nn.Conv2d(compress_channel, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
        
class AE_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, init_channel, compress_channel, bilinear=True):
        super(AE_Conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_channel = init_channel
        self.compress_channel = compress_channel
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, init_channel)
        self.down1 = Down(init_channel, init_channel*2)
        self.down2 = Down(init_channel*2, init_channel*4)
        self.down3 = Down(init_channel*4, init_channel*8)
        self.down4 = Down(init_channel*8, init_channel*16)
        self.compress = CompressConv(init_channel*16, compress_channel)
        self.decompress = DeCompressConv(compress_channel, init_channel*16)
        self.up4 = Up(init_channel*16, init_channel*8, bilinear)
        self.up3 = Up(init_channel*8, init_channel*4, bilinear)
        self.up2 = Up(init_channel*4, init_channel*2, bilinear)
        self.up1 = Up(init_channel*2, init_channel, bilinear)
        self.outc = OutConv(init_channel, out_channels)

        self._initialize_weights()

    def forward(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.compress(x)
        x = self.decompress(x)
        x = self.up4(x)
        x = self.up3(x)
        x = self.up2(x)
        x = self.up1(x)
        x = self.outc(x)
        x = torch.sigmoid(x)
        return x

    def _initialize_weights(self):
       for m in self.modules():
           if isinstance(m, nn.Conv2d):
               nn.init.kaiming_normal_(m.weight)
               if m.bias is not None:
                   nn.init.zeros_(m.bias)
           elif isinstance(m, nn.BatchNorm2d):
               nn.init.constant_(m.weight, 1)
               nn.init.constant_(m.bias, 0)
               
class Encoder(nn.Module):
    def __init__(self, in_channels, init_channel, compress_channel, bilinear=True):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.init_channel = init_channel
        self.compress_channel = compress_channel
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, init_channel)
        self.down1 = Down(init_channel, init_channel*2)
        self.down2 = Down(init_channel*2, init_channel*4)
        self.down3 = Down(init_channel*4, init_channel*8)
        self.down4 = Down(init_channel*8, init_channel*16)
        self.compress = CompressConv(init_channel*16, compress_channel)

        self._initialize_weights()

    def forward(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.compress(x)
        return x
        
    def _initialize_weights(self):
       for m in self.modules():
           if isinstance(m, nn.Conv2d):
               nn.init.kaiming_normal_(m.weight)
               if m.bias is not None:
                   nn.init.zeros_(m.bias)
           elif isinstance(m, nn.BatchNorm2d):
               nn.init.constant_(m.weight, 1)
               nn.init.constant_(m.bias, 0)