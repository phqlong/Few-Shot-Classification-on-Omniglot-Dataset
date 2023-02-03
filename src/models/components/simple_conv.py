from torch import nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class SimpleConvNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        kernel_size: int = 3,
        hidden_channel: int = 64,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, hidden_channel, kernel_size),
            nn.BatchNorm2d(hidden_channel, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(hidden_channel, hidden_channel, kernel_size),
            nn.BatchNorm2d(hidden_channel, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(hidden_channel, hidden_channel, kernel_size),
            nn.BatchNorm2d(hidden_channel, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            Flatten(),
            nn.Linear(hidden_channel, num_classes))

    def forward(self, x):
        batch_size, channels, width, height = x.size()

        # (batch, 1, width, height) -> (batch, 1*width*height)
        # x = x.view(batch_size, -1)

        return self.model(x)


if __name__ == "__main__":
    _ = SimpleConvNet()
