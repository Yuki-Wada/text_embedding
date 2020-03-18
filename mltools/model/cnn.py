import torch.nn as nn

class TempCNN(nn.Module):
    def __init__(self, label_count: int):
        super(TempCNN, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, 5),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 8, 5),
            nn.MaxPool2d(2, 2),
        )
        self.output_layer = nn.Linear(200, label_count)

    def forward(self, *input_data):
        out, = input_data
        for layer in self.layers:
            out = layer(out)
        out = out.view(out.shape[0], -1)
        outputs = self.output_layer(out)

        return outputs

    def calculate_loss(self, images, labels):
        self.train()
        outputs = self(images)
        return self.loss_func(outputs, labels)

    def predict(self, images):
        self.eval()
        return self(images)
