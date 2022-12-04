class Net(Module):
    def __init__(selfself, in_channels, out_channels):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(in_channels, 64, kernel_size=12, stride=4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=1),
            # Defining another 2D convolution layer
            Conv2d(64, 112, kernel_size=4, stride=1),
            ReLU(inplace=True),
            # Defining another 2D convolution layer
            Conv2d(1, 112, kernel_size=3, stride=1),
            ReLU(inplace=True),
            # Defining another 2D convolution layer
            Linear(112, 4096)
        )

        self.linear_layers = Sequential(
            Linear(4 * 7 * 7, 10)
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x