"""
EXP-007: three ways to build a "two-input" model.

Every model gets the same two pictures, both 448 x 128 and lined up:
    ref   the clean reference logo
    cand  the logo from a photo, rectified onto the reference (EXP-001 style)
and gives two answers:
    tampered   one number (a logit): > 0 leans "tampered"
    kind       8 numbers, one per class in TAMPER_TYPES ("which tamper?")

The three models differ in WHERE the two pictures are compared:

  SiameseDiff   (recommended) the same pretrained ResNet-18 looks at each
                picture separately; we compare the two feature MAPS location
                by location, so the model keeps track of WHERE they differ.
  EarlyFusion   the two pictures are stacked into one 6-channel picture and
                a single ResNet-18 looks at both at once. Simplest baseline.
  DinoFrozen    a frozen DINOv2 vision transformer turns each picture into a
                grid of patch features; only a small comparison head is
                trained. Least likely to over-fit our synthetic data, but its
                14 px patches are coarse for tiny shifts.

Licences: torchvision ResNet-18 weights (BSD-3), DINOv2 weights (Apache 2.0) via timm (Apache 2.0).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

TAMPER_TYPES = ["none", "stretch", "recolour", "respace", "glyph_swap", "font_swap", "overlap", "cut"]
INPUT_H, INPUT_W = 128, 448


def resnet18(pretrained):
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    return torchvision.models.resnet18(weights=weights)


class Heads(nn.Module):
    """The two outputs every model ends with."""

    def __init__(self, in_dim):
        super().__init__()
        self.tampered = nn.Linear(in_dim, 1)
        self.kind = nn.Linear(in_dim, len(TAMPER_TYPES))

    def forward(self, x):
        return self.tampered(x).squeeze(1), self.kind(x)


def compare_block(in_ch):
    """Two small conv layers that look at the 'difference' features."""
    return nn.Sequential(
        nn.Conv2d(in_ch, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
    )


def pool(x):
    """Max (strongest local difference) and mean (overall difference) over the map."""
    return torch.cat([x.amax(dim=(2, 3)), x.mean(dim=(2, 3))], dim=1)


class SiameseDiff(nn.Module):
    def __init__(self, pretrained=True, depth=2, dropout=0.0):
        super().__init__()
        r = resnet18(pretrained)
        # depth = how many ResNet stages we keep:
        #   1 -> 64 channels at 1/4 of the input (32 x 112): finest positions
        #   2 -> 128 channels at 1/8 (16 x 56): the default
        #   3 -> 256 channels at 1/16 (8 x 28): more "meaning", coarser positions
        # Small shifts and swapped letters need fine position information,
        # which is why we stop early rather than using the whole network.
        stages = [r.layer1, r.layer2, r.layer3][:depth]
        self.encoder = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, *stages)
        channels = {1: 64, 2: 128, 3: 256}[depth]
        self.compare = compare_block(2 * channels)
        self.dropout = nn.Dropout(dropout)
        self.heads = Heads(256)

    def forward(self, ref, cand):
        a, b = self.encoder(ref), self.encoder(cand)            # same weights for both
        x = self.compare(torch.cat([(a - b).abs(), a * b], 1))  # where and how they differ
        return self.heads(self.dropout(pool(x)))

    @torch.no_grad()
    def heatmap(self, ref, cand):
        """Where the two feature maps differ most (for showing the panel)."""
        return (self.encoder(ref) - self.encoder(cand)).abs().mean(1)


class EarlyFusion(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        r = resnet18(pretrained)
        old = r.conv1
        r.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            # Re-use the pretrained RGB filters for both pictures (halved so the
            # first layer's output keeps roughly the same scale).
            r.conv1.weight.copy_(torch.cat([old.weight, old.weight], 1) / 2)
        r.fc = nn.Identity()
        self.net = r
        self.heads = Heads(512)

    def forward(self, ref, cand):
        return self.heads(self.net(torch.cat([ref, cand], 1)))


class DinoFrozen(nn.Module):
    VIT_H, VIT_W = 126, 448          # must be multiples of the 14 px patch size

    def __init__(self, pretrained=True):
        super().__init__()
        import timm  # only needed for this model
        self.vit = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=pretrained,
                                     img_size=(self.VIT_H, self.VIT_W), num_classes=0)
        for p in self.vit.parameters():
            p.requires_grad = False
        self.vit.eval()
        dim = self.vit.embed_dim                   # 384 for ViT-S
        self.grid = (self.VIT_H // 14, self.VIT_W // 14)   # 9 x 32 patches
        self.compare = compare_block(2 * dim)
        self.heads = Heads(256)

    def train(self, mode=True):
        super().train(mode)
        self.vit.eval()                            # the frozen part never switches to training mode
        return self

    @torch.no_grad()
    def patches(self, x):
        x = F.interpolate(x, size=(self.VIT_H, self.VIT_W), mode="bilinear", align_corners=False)
        tokens = self.vit.forward_features(x)[:, self.vit.num_prefix_tokens:]   # drop the CLS token
        b, n, c = tokens.shape
        return tokens.transpose(1, 2).reshape(b, c, *self.grid)

    def forward(self, ref, cand):
        a, b = self.patches(ref), self.patches(cand)
        x = self.compare(torch.cat([(a - b).abs(), a * b], 1))
        return self.heads(pool(x))


ARCHS = {"siamese_diff": SiameseDiff, "early_fusion": EarlyFusion, "dino_frozen": DinoFrozen}


def build(arch, pretrained=True, **options):
    """options (depth, dropout) only apply to siamese_diff."""
    if arch == "siamese_diff":
        return SiameseDiff(pretrained=pretrained, **options)
    return ARCHS[arch](pretrained=pretrained)
