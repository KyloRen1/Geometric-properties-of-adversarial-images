import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, nz, feature_map=64, n_channels=1):
        super(Generator, self).__init__()
        self.conv_model = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=nz,
                out_channels=feature_map * 8,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(num_features=feature_map * 8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                in_channels=feature_map * 8,
                out_channels=feature_map * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=feature_map * 4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                in_channels=feature_map * 4,
                out_channels=feature_map * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=feature_map * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                in_channels=feature_map * 2,
                out_channels=feature_map,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=feature_map),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                in_channels=feature_map,
                out_channels=n_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv_model(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, in_channels, feature_map=64):
        super(Discriminator, self).__init__()
        self.conv_model = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=feature_map,
                kernel_size=4,
                stride=2, padding=1,
                bias=False
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(
                in_channels=feature_map,
                out_channels=feature_map * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=feature_map * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(
                in_channels=feature_map * 2,
                out_channels=feature_map * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=feature_map * 4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(
                in_channels=feature_map * 4,
                out_channels=feature_map * 8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=feature_map * 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(
                in_channels=feature_map * 8,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_model(x)
        return x


def weights_init(x):
    classname = x.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(x.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(x.weight.data, 1.0, 0.02)
        nn.init.constant_(x.bias.data, 0)


def get_models(nz_size, in_channels, feature_map, device):
    generator = Generator(
        nz_size, feature_map, in_channels
    ).to(device)
    generator.apply(weights_init)

    discriminator = Discriminator(
        in_channels, feature_map
    ).to(device)
    discriminator.apply(weights_init)
    return generator, discriminator
