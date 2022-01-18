import torch
import torch.nn as nn
from torchvision import models


def main():
    pytorch_model = models.resnet18(pretrained=True, progress=True)
    for param in pytorch_model.parameters():
        param.requires_grad = False

    num_ftrs = pytorch_model.fc.in_features

    pytorch_model.fc = nn.Linear(num_ftrs, 26)

    pytorch_model.load_state_dict(torch.load('model_weights_usign.pth', map_location=torch.device('cpu')))
    pytorch_model.eval()
    dummy_input = torch.zeros(64, 3, 7, 7)
    torch.onnx.export(pytorch_model, dummy_input, 'onyx_model.onnx', verbose=True)


if __name__ == '__main__':
    main()
