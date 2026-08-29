
from contextlib import nullcontext

import numpy as np
import timm
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class DINOFeatureExtractor:
    """
    Extract patch-mean DINOv3 embeddings from pasture images.
    """

    def __init__(
        self,
        backbone_name,
        image_height,
        image_width,
        device=None
    ):
        self.backbone_name = backbone_name
        self.image_height = image_height
        self.image_width = image_width

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)

        # Exact preprocessing used during training
        self.transform = transforms.Compose([
            transforms.Resize(
                (
                    self.image_height,
                    self.image_width
                ),
                interpolation=InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Frozen pretrained DINO backbone
        self.model = timm.create_model(
            self.backbone_name,
            pretrained=True,
            num_classes=0,
            img_size=(
                self.image_height,
                self.image_width
            )
        )

        self.model = self.model.to(
            self.device
        )

        self.model.eval()

        for parameter in self.model.parameters():
            parameter.requires_grad = False

        self.embedding_dim = (
            self.model.num_features
        )

    def extract(self, image_path):
        """
        Convert one image into a 2D numpy embedding
        with shape (1, embedding_dim).
        """

        image = Image.open(
            image_path
        ).convert("RGB")

        image_tensor = (
            self.transform(image)
            .unsqueeze(0)
            .to(self.device)
        )

        # FP16 only when CUDA is available
        autocast_context = (
            torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            )
            if self.device.type == "cuda"
            else nullcontext()
        )

        with torch.inference_mode():

            with autocast_context:

                tokens = self.model.forward_features(
                    image_tensor
                )

                patch_tokens = tokens[
                    :,
                    self.model.num_prefix_tokens:,
                    :
                ]

                embedding = patch_tokens.mean(
                    dim=1
                )

        embedding = (
            embedding
            .float()
            .cpu()
            .numpy()
        )

        return embedding
