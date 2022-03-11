import torch
import torch.nn as nn

class CDCGAN_Generator(nn.Module):
    def __init__(self, num_class, depth = 128):
        super(CDCGAN_Generator, self).__init__()

        self.z_deconv = nn.Sequential(
            nn.ConvTranspose2d(num_class * num_class, depth * 2, 4, 1, 0, bias = False),
            nn.BatchNorm2d(depth * 2),
            nn.ReLU(True),
        )

        self.y_deconv = nn.Sequential(
            nn.ConvTranspose2d(num_class, depth * 2, 4, 1, 0, bias = False),
            nn.BatchNorm2d(depth * 2),
            nn.ReLU(True),
        )

        self.main = torch.nn.Sequential(
            nn.ConvTranspose2d(depth * 4, depth * 2, 4, 2, 1, bias = False),
            nn.BatchNorm2d(depth * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(depth * 2, depth, 4, 2, 1, bias = False),
            nn.BatchNorm2d(depth),
            nn.ReLU(True),

            nn.ConvTranspose2d(depth, 1, 4, 2, 1, bias = False),
            nn.Tanh()
        )

    def forward(self, input, label):
        z = self.z_deconv(input)
        y = self.y_deconv(label)
        output = torch.cat([z, y], 1)
        output = self.main(output)

        return output

    def init_weight(self, mean, std):
        for m in self._modules:
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(mean, std)
                m.bias.data.zero_()

class CDCGAN_Discriminator(nn.Module):
    def __init__(self, num_class, depth = 128):
        super(CDCGAN_Discriminator, self).__init__()

        self.x_conv = nn.Sequential(
            nn.Conv2d(1, depth / 2, 4, 2, 1, bias = False),
            nn.LeakyReLU(0.2, inplace = True),
        )

        self.y_conv = nn.Sequential(
            nn.Conv2d(num_class, depth / 2, 4, 2, 1, bias = False),
            nn.LeakyReLU(0.2, inplace = True),
        )

        self.main = nn.Sequential(
            nn.Conv2d(depth, depth * 2, 4, 2, 1, bias = False),
            nn.BatchNorm2d(depth * 2),
            nn.LeakyReLU(0.2, inplace = True),

            nn.Conv2d(depth * 2, depth * 4, 4, 2, 1, bias = False),
            nn.BatchNorm2d(depth * 4),
            nn.LeakyReLU(0.2, inplace = True),

            nn.Conv2d(depth * 4, 1, 4, 1, 0, bias = False),
            nn.Sigmoid()
        )

    def forward(self, input, label):
        x = self.x_conv(input)
        y = self.y_conv(label)
        output = torch.cat([x, y], 1)
        output = self.main(output)

        return output

    def init_weight(self, mean, std):
        for m in self._modules:
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(mean, std)
                m.bias.data.zero_()
