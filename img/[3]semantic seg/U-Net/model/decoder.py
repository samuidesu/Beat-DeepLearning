"""U-Net decoder: expanding path that mirrors the ResNet encoder.

This replaces the FCN project's FPN neck. Both fuse deep semantics into
shallow, high-resolution maps top-down, but the recipe differs in three ways
(the classic U-Net choices):

  1. CONCAT, not add: the skip feature map is concatenated channel-wise with
     the upsampled decoder map, so the network can freely mix "what" (deep)
     and "where" (shallow) instead of being forced to sum them.
  2. LEARNED upsampling: ConvTranspose2d(kernel=2, stride=2) doubles H,W with
     trainable weights (the FPN neck used parameter-free nearest-neighbor).
  3. Channels SHRINK as resolution grows (512 -> 256 -> 128 -> 64 -> 64),
     mirroring the encoder, instead of a constant 256-wide pyramid.

One extra step vs. the FCN neck: we also fuse C1 (the stride-2 stem output),
so the decoder climbs all the way back to stride 2. The head then needs only
a single 2x upsample to reach full input resolution.

Data flow (H, W = input size, ResNet-18/34 channels):

    c5 [B,512,H/32] --up--> +c4 [B,256,H/16] -> p4 [B,256,H/16]
    p4              --up--> +c3 [B,128,H/8 ] -> p3 [B,128,H/8 ]
    p3              --up--> +c2 [B, 64,H/4 ] -> p2 [B, 64,H/4 ]
    p2              --up--> +c1 [B, 64,H/2 ] -> p1 [B, 64,H/2 ]   <- returned
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """The U-Net workhorse block: (Conv3x3 -> BN -> ReLU) x 2.

    Args:
        in_ch, out_ch: input / output channel counts.

    Input:  [B, in_ch,  H, W]
    Output: [B, out_ch, H, W]  (padding=1 keeps H,W unchanged)

    Two stacked 3x3 convs see a 5x5 neighborhood with fewer parameters than
    one 5x5 conv, and get two nonlinearities instead of one. bias=False
    because each conv is followed by a BatchNorm that has its own shift term.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_ch,
                out_ch,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """One decoder step: 2x learned upsample -> concat skip -> DoubleConv.

    Args:
        decoder_ch: channels of the incoming (deeper) decoder map.
        skip_ch:    channels of the encoder skip map at the target resolution.
        out_ch:     channels of this block's output.

    Input:
        x:    [B, decoder_ch, h,  w ]  (deeper decoder feature map)
        skip: [B, skip_ch,    2h, 2w]  (encoder tap, twice the resolution)
    Output:
        [B, out_ch, 2h, 2w]

    Step by step:
        1. ConvTranspose2d(kernel=2, stride=2) doubles h,w AND projects
           decoder_ch -> out_ch (learned upsampling; each output 2x2 tile is
           a learned function of one input pixel, so no overlap artifacts).
        2. torch.cat glues the skip map on the channel axis: the upsampled
           map contributes semantics, the skip map contributes precise
           spatial detail lost to downsampling.
        3. DoubleConv mixes the two sources and sets the output width.
    """

    def __init__(
        self,
        decoder_ch: int,
        skip_ch: int,
        out_ch: int,
    ):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            decoder_ch,
            out_ch,
            kernel_size=2,
            stride=2,
        )

        self.fuse = DoubleConv(
            out_ch + skip_ch,
            out_ch,
        )

    def forward(self, x, skip):
        x = self.up(x)                  # [B, out_ch, 2h, 2w]

        # Requires matching H,W -- guaranteed here because the input image is
        # padded to a multiple of 32, so every downsample halves exactly.
        x = torch.cat(
            (x, skip),
            dim=1,
        )                               # [B, out_ch + skip_ch, 2h, 2w]

        return self.fuse(x)             # [B, out_ch, 2h, 2w]


class UNetDecoder(nn.Module):
    """The full expanding path: 4 UpBlocks from stride 32 back to stride 2.

    Args:
        in_channels: channel counts of the encoder taps (c1, c2, c3, c4, c5).
            Defaults to ResNet-18/34's (64, 64, 128, 256, 512). c5 is the
            "bottleneck" input; c1..c4 are the skip connections.

    Attributes:
        out_channels (int): channels of the returned p1 map (= c1's count);
            the head reads this to size its classifier conv.
    """

    def __init__(self, in_channels=(64, 64, 128, 256, 512)):
        super().__init__()
        c1, c2, c3, c4, c5 = in_channels
        self.out_channels = c1

        # Each block halves the channel count while doubling the resolution,
        # mirroring the encoder. (No extra bottleneck below c5: the ResNet's
        # layer4 already plays the role of the classic U-Net bottom block.)
        self.decoder4 = UpBlock(decoder_ch=c5, skip_ch=c4, out_ch=c4)  # 512 -> 256, H/32 -> H/16
        self.decoder3 = UpBlock(decoder_ch=c4, skip_ch=c3, out_ch=c3)  # 256 -> 128, H/16 -> H/8
        self.decoder2 = UpBlock(decoder_ch=c3, skip_ch=c2, out_ch=c2)  # 128 ->  64, H/8  -> H/4
        self.decoder1 = UpBlock(decoder_ch=c2, skip_ch=c1, out_ch=c1)  #  64 ->  64, H/4  -> H/2

    def forward(self, feats):
        """Fuse the 5 encoder maps bottom-up into one stride-2 map.

        Input:
            feats: tuple (c1, c2, c3, c4, c5) from the encoder
                c1: [B, 64,  H/2,  W/2]   (stride 2, stem output)
                c2: [B, 64,  H/4,  W/4]   (stride 4)
                c3: [B, 128, H/8,  W/8]   (stride 8)
                c4: [B, 256, H/16, W/16]  (stride 16)
                c5: [B, 512, H/32, W/32]  (stride 32, the bottleneck)

        Output:
            p1: [B, out_channels, H/2, W/2] -- the single fused stride-2
                feature map. The head turns it into per-pixel class logits
                and upsamples 2x back to input resolution.
        """
        c1, c2, c3, c4, c5 = feats

        p4 = self.decoder4(c5, c4)      # [B, 256, H/16, W/16]
        p3 = self.decoder3(p4, c3)      # [B, 128, H/8,  W/8]
        p2 = self.decoder2(p3, c2)      # [B, 64,  H/4,  W/4]
        p1 = self.decoder1(p2, c1)      # [B, 64,  H/2,  W/2]

        return p1


# ---- Quick self-test: run this file directly to verify shapes ---------------
# python model/decoder.py
if __name__ == "__main__":
    # Fake encoder outputs for a 416x416 input (batch of 2).
    c1 = torch.randn(2, 64, 208, 208)
    c2 = torch.randn(2, 64, 104, 104)
    c3 = torch.randn(2, 128, 52, 52)
    c4 = torch.randn(2, 256, 26, 26)
    c5 = torch.randn(2, 512, 13, 13)

    decoder = UNetDecoder(in_channels=(64, 64, 128, 256, 512))
    p1 = decoder((c1, c2, c3, c4, c5))

    print("p1:", tuple(p1.shape), "(expected (2, 64, 208, 208))")
    print("out_channels:", decoder.out_channels)
