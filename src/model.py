import torch
import torch.nn as nn

from torchvision.models import (
    resnet18,
    ResNet18_Weights,
    efficientnet_b0,
    EfficientNet_B0_Weights
)


def create_model(model_name="resnet18"):

    # --------------------------------------------------
    # ResNet18
    # --------------------------------------------------

    if model_name == "resnet18":

        weights = ResNet18_Weights.DEFAULT

        model = resnet18(
            weights=weights
        )

        # Original final layer:
        # 512 features -> 1000 ImageNet classes
        #
        # Replace it with:
        # 512 features -> 2 classes
        num_features = model.fc.in_features

        model.fc = nn.Linear(
            num_features,
            2
        )

        return model


    # --------------------------------------------------
    # EfficientNet-B0
    # --------------------------------------------------

    elif model_name == "efficientnet_b0":

        weights = EfficientNet_B0_Weights.DEFAULT

        model = efficientnet_b0(
            weights=weights
        )

        # EfficientNet stores its final linear layer
        # inside model.classifier.
        num_features = model.classifier[1].in_features

        model.classifier[1] = nn.Linear(
            num_features,
            2
        )

        return model


    # --------------------------------------------------
    # Invalid model name
    # --------------------------------------------------

    else:

        raise ValueError(
            f"Unknown model name: {model_name}"
        )


# --------------------------------------------------
# Quick test
# --------------------------------------------------

if __name__ == "__main__":

    fake_images = torch.randn(
        4,
        3,
        224,
        224
    )


    # Test ResNet18
    resnet_model = create_model(
        "resnet18"
    )

    resnet_output = resnet_model(
        fake_images
    )

    print("ResNet18 output shape:")
    print(resnet_output.shape)


    # Test EfficientNet-B0
    efficientnet_model = create_model(
        "efficientnet_b0"
    )

    efficientnet_output = efficientnet_model(
        fake_images
    )

    print("\nEfficientNet-B0 output shape:")
    print(efficientnet_output.shape)