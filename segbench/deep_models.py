"""Shared neural architectures for the segmentation benchmark.

The model factory is intentionally independent of paths, devices, datasets and
training policy.  Every network emits raw logits.  Thresholding and mask
post-processing live in the shared training/inference utilities so a method
cannot silently use a more favourable evaluator.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

from segbench.artifact_identity import git_blob_sha1, sha256_file

import torch
from torch import Tensor, nn
import torch.nn.functional as functional


SegmentationOutput = Union[Tensor, Sequence[Tensor]]


def parameter_counts(model: nn.Module) -> Dict[str, int]:
    """Return total and trainable scalar parameter counts."""
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }


class ConvNormAct(nn.Sequential):
    """Convolution followed by batch normalisation, ReLU and optional dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        padding = kernel_size // 2
        layers: List[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        super().__init__(*layers)


class DoubleConv(nn.Sequential):
    """The two-convolution block used by U-Net and UNet++."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            ConvNormAct(in_channels, out_channels),
            ConvNormAct(out_channels, out_channels),
        )


class UpBlock(nn.Module):
    """Bilinear upsampling, projection, skip concatenation and convolution."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.project = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.fuse = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, value: Tensor, skip: Tensor) -> Tensor:
        value = functional.interpolate(
            value, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )
        value = self.project(value)
        return self.fuse(torch.cat((skip, value), dim=1))


class UNet(nn.Module):
    """Four-level Ronneberger U-Net with BatchNorm and bilinear upsampling."""

    def __init__(self, in_channels: int = 1, base_channels: int = 32, levels: int = 4) -> None:
        super().__init__()
        if levels != 4:
            raise ValueError("the benchmark U-Net is frozen at four down/up levels")
        channels = [base_channels * (2**index) for index in range(levels + 1)]
        self.encoders = nn.ModuleList(
            [DoubleConv(in_channels, channels[0])]
            + [DoubleConv(channels[index - 1], channels[index]) for index in range(1, 5)]
        )
        self.pool = nn.MaxPool2d(2)
        self.decoders = nn.ModuleList(
            [
                UpBlock(channels[index], channels[index - 1], channels[index - 1])
                for index in range(4, 0, -1)
            ]
        )
        self.head = nn.Conv2d(channels[0], 1, kernel_size=1)

    def forward(self, image: Tensor) -> Tensor:
        skips: List[Tensor] = []
        value = image
        for index, encoder in enumerate(self.encoders):
            value = encoder(value)
            if index < len(self.encoders) - 1:
                skips.append(value)
                value = self.pool(value)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            value = decoder(value, skip)
        return self.head(value)


class UNetPP(nn.Module):
    """Native four-level UNet++ with four deep-supervision heads."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        levels: int = 4,
        deep_supervision: bool = True,
    ) -> None:
        super().__init__()
        if levels != 4:
            raise ValueError("the benchmark UNet++ is frozen at four down/up levels")
        self.deep_supervision = deep_supervision
        channels = [base_channels * (2**index) for index in range(5)]
        self.pool = nn.MaxPool2d(2)
        self.base = nn.ModuleList(
            [DoubleConv(in_channels, channels[0])]
            + [DoubleConv(channels[index - 1], channels[index]) for index in range(1, 5)]
        )
        self.nested = nn.ModuleDict()
        for depth in range(1, 5):
            for level in range(0, 5 - depth):
                in_count = depth * channels[level] + channels[level + 1]
                self.nested[f"{level}_{depth}"] = DoubleConv(in_count, channels[level])
        self.heads = nn.ModuleList(
            [nn.Conv2d(channels[0], 1, kernel_size=1) for _ in range(4)]
        )

    @staticmethod
    def _upsample(value: Tensor, reference: Tensor) -> Tensor:
        return functional.interpolate(
            value, size=reference.shape[-2:], mode="bilinear", align_corners=False
        )

    def forward(self, image: Tensor) -> SegmentationOutput:
        nodes: Dict[Tuple[int, int], Tensor] = {}
        nodes[(0, 0)] = self.base[0](image)
        for level in range(1, 5):
            nodes[(level, 0)] = self.base[level](self.pool(nodes[(level - 1, 0)]))
        for depth in range(1, 5):
            for level in range(0, 5 - depth):
                same_level = [nodes[(level, prior)] for prior in range(depth)]
                deeper = self._upsample(nodes[(level + 1, depth - 1)], same_level[0])
                nodes[(level, depth)] = self.nested[f"{level}_{depth}"](
                    torch.cat((*same_level, deeper), dim=1)
                )
        outputs = [head(nodes[(0, index)]) for index, head in enumerate(self.heads, 1)]
        if self.training and self.deep_supervision:
            return outputs
        return outputs[-1]


class RecurrentUnit(nn.Module):
    """A convolutional unit recurrently refining a fixed input feature map."""

    def __init__(self, channels: int, steps: int, dropout: float) -> None:
        super().__init__()
        if steps < 1:
            raise ValueError("recurrent_steps must be positive")
        self.steps = steps
        self.block = ConvNormAct(channels, channels, dropout=dropout)

    def forward(self, value: Tensor) -> Tensor:
        state: Tensor | None = None
        for _ in range(self.steps):
            state = self.block(value if state is None else value + state)
        if state is None:  # defensive; steps is validated above
            raise RuntimeError("recurrent unit produced no state")
        return state


class RecurrentResidualBlock(nn.Module):
    """R2U-Net recurrent residual block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        steps: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.recurrent_1 = RecurrentUnit(out_channels, steps, dropout)
        self.recurrent_2 = RecurrentUnit(out_channels, steps, dropout)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, value: Tensor) -> Tensor:
        residual = self.project(value)
        update = self.recurrent_2(self.recurrent_1(residual))
        return self.activation(residual + update)


class RecurrentUpBlock(nn.Module):
    """Decoder block for recurrent U-Nets."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        *,
        steps: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.project = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.fuse = RecurrentResidualBlock(
            out_channels + skip_channels,
            out_channels,
            steps=steps,
            dropout=dropout,
        )

    def forward(self, value: Tensor, skip: Tensor) -> Tensor:
        value = functional.interpolate(
            value, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.fuse(torch.cat((skip, self.project(value)), dim=1))


class RecurrentUNet(nn.Module):
    """Shared recurrent segmentation backbone used by DRUnet and RA-RUnet."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        levels: int = 4,
        recurrent_steps: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if levels != 4:
            raise ValueError("the benchmark recurrent U-Net is frozen at four levels")
        channels = [base_channels * (2**index) for index in range(5)]
        self.bottleneck_channels = channels[-1]
        self.encoders = nn.ModuleList(
            [
                RecurrentResidualBlock(
                    in_channels if index == 0 else channels[index - 1],
                    channels[index],
                    steps=recurrent_steps,
                    dropout=dropout,
                )
                for index in range(5)
            ]
        )
        self.pool = nn.MaxPool2d(2)
        self.decoders = nn.ModuleList(
            [
                RecurrentUpBlock(
                    channels[index],
                    channels[index - 1],
                    channels[index - 1],
                    steps=recurrent_steps,
                    dropout=dropout,
                )
                for index in range(4, 0, -1)
            ]
        )
        self.head = nn.Conv2d(channels[0], 1, kernel_size=1)

    def forward_features(self, image: Tensor) -> Tuple[Tensor, Tensor]:
        skips: List[Tensor] = []
        value = image
        for index, encoder in enumerate(self.encoders):
            value = encoder(value)
            if index < len(self.encoders) - 1:
                skips.append(value)
                value = self.pool(value)
        bottleneck = value
        for decoder, skip in zip(self.decoders, reversed(skips)):
            value = decoder(value, skip)
        return self.head(value), bottleneck

    def forward(self, image: Tensor) -> Tensor:
        logits, _ = self.forward_features(image)
        return logits


class GradientReversalFunction(torch.autograd.Function):
    """Identity forward pass with a scaled, sign-reversed backward pass."""

    @staticmethod
    def forward(ctx: Any, value: Tensor, weight: float) -> Tensor:
        ctx.weight = float(weight)
        return value.view_as(value)

    @staticmethod
    def backward(ctx: Any, gradient: Tensor) -> Tuple[Tensor, None]:
        return -ctx.weight * gradient, None


def gradient_reverse(value: Tensor, weight: float) -> Tensor:
    """Apply gradient reversal to ``value``."""
    return GradientReversalFunction.apply(value, weight)


class RARUNet(nn.Module):
    """Recurrent U-Net with an optional bottleneck domain discriminator."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        levels: int = 4,
        recurrent_steps: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = RecurrentUNet(
            in_channels=in_channels,
            base_channels=base_channels,
            levels=levels,
            recurrent_steps=recurrent_steps,
            dropout=dropout,
        )
        bottleneck_channels = self.backbone.bottleneck_channels
        self.domain_discriminator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(bottleneck_channels, max(16, bottleneck_channels // 2)),
            nn.ReLU(inplace=True),
            nn.Linear(max(16, bottleneck_channels // 2), 1),
        )

    def forward(
        self,
        image: Tensor,
        *,
        return_domain: bool = False,
        gradient_reversal_weight: float = 1.0,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        logits, bottleneck = self.backbone.forward_features(image)
        if not return_domain:
            return logits
        domain_logits = self.domain_discriminator(
            gradient_reverse(bottleneck, gradient_reversal_weight)
        ).squeeze(1)
        return logits, domain_logits


class LeNetPatchClassifier(nn.Module):
    """LeNet-5-style binary classifier for 32 x 32 context patches."""

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 1),
        )

    def forward(self, patch: Tensor) -> Tensor:
        if patch.shape[-2:] != (32, 32):
            raise ValueError(f"LeNet expects 32 x 32 patches, got {tuple(patch.shape[-2:])}")
        return self.classifier(self.features(patch)).squeeze(1)


class AlexNetPatchClassifier(nn.Module):
    """Five-convolution, three-linear AlexNet adaptation for 64 x 64 patches."""

    def __init__(self, in_channels: int = 1, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256 * 2 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, patch: Tensor) -> Tensor:
        if patch.shape[-2:] != (64, 64):
            raise ValueError(f"AlexNet expects 64 x 64 patches, got {tuple(patch.shape[-2:])}")
        return self.classifier(self.features(patch)).squeeze(1)


class ConvAutoencoder(nn.Module):
    """Stacked convolutional autoencoder used for train-only pretraining."""

    def __init__(self, in_channels: int = 1, latent_channels: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, latent_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, patch: Tensor) -> Tensor:
        return self.decoder(self.encoder(patch))


class CNNAEClassifier(nn.Module):
    """Autoencoder encoder followed by a binary patch-classification head."""

    def __init__(
        self,
        in_channels: int = 1,
        latent_channels: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        autoencoder = ConvAutoencoder(in_channels, latent_channels)
        self.encoder = autoencoder.encoder
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(latent_channels, 1),
        )

    def forward(self, patch: Tensor) -> Tensor:
        if patch.shape[-2:] != (64, 64):
            raise ValueError(f"CNN+AE expects 64 x 64 patches, got {tuple(patch.shape[-2:])}")
        return self.classifier(self.encoder(patch)).squeeze(1)


def verify_pinned_pretrained_snapshot(
    root: Path, declared: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """Verify every declared local upstream file before model construction."""
    if not root.is_dir():
        raise FileNotFoundError(f"pinned pretrained directory is absent: {root}")
    observed: Dict[str, Dict[str, Any]] = {}
    if not declared:
        raise ValueError("pretrained_files must be a non-empty mapping")
    for relative, identity in declared.items():
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError(f"unsafe pretrained filename: {relative!r}")
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing pinned pretrained file: {relative}")
        record: Dict[str, Any] = {"bytes": path.stat().st_size}
        if "bytes" in identity and record["bytes"] != int(identity["bytes"]):
            raise ValueError(f"{relative}: byte-size mismatch")
        if "sha256" in identity:
            record["sha256"] = sha256_file(path)
            if record["sha256"] != identity["sha256"]:
                raise ValueError(f"{relative}: SHA-256 mismatch")
        if "git_blob_sha1" in identity:
            record["git_blob_sha1"] = git_blob_sha1(path)
            if record["git_blob_sha1"] != identity["git_blob_sha1"]:
                raise ValueError(f"{relative}: Git-blob identity mismatch")
        observed[relative] = record
    if "config.json" not in observed or "pytorch_model.bin" not in observed:
        raise ValueError("pinned SegFormer snapshot lacks config or PyTorch weights")
    return observed


class SegFormerBinary(nn.Module):
    """Hugging Face SegFormer-B2 with a deterministic one-channel adapter."""

    def __init__(
        self,
        pretrained_model: str,
        *,
        pretrained: bool = True,
        local_files_only: bool = False,
        pretrained_revision: str = "",
        pretrained_path_env: str = "",
        pretrained_files: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        super().__init__()
        try:
            from transformers import SegformerConfig, SegformerForSemanticSegmentation
        except ImportError as exc:  # pragma: no cover - depends on optional environment
            raise RuntimeError("SegFormer requires the transformers package") from exc

        if pretrained:
            if not local_files_only:
                raise ValueError("pretrained benchmark initialization must be local-only")
            if len(pretrained_revision) != 40 or any(
                value not in "0123456789abcdef" for value in pretrained_revision
            ):
                raise ValueError("pretrained_revision must be an immutable 40-hex commit")
            if not pretrained_path_env:
                raise ValueError("pretrained_path_env is required")
            local_path_value = os.environ.get(pretrained_path_env)
            if not local_path_value:
                raise RuntimeError(f"{pretrained_path_env} is not set")
            local_path = Path(local_path_value)
            observed_files = verify_pinned_pretrained_snapshot(
                local_path, pretrained_files or {}
            )
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                str(local_path),
                num_labels=1,
                ignore_mismatched_sizes=True,
                local_files_only=True,
            )
            self.pretrained_provenance = {
                "model_id": pretrained_model,
                "revision": pretrained_revision,
                "local_files_only": True,
                "observed_files": observed_files,
                "classification_head_discarded": True,
                "segmentation_head_initialized_for_benchmark": True,
            }
        else:
            configuration = SegformerConfig(
                num_labels=1,
                num_channels=1,
                depths=[3, 4, 6, 3],
                hidden_sizes=[64, 128, 320, 512],
                decoder_hidden_size=768,
                num_attention_heads=[1, 2, 5, 8],
            )
            self.model = SegformerForSemanticSegmentation(configuration)

        projection = self.model.segformer.encoder.patch_embeddings[0].proj
        if projection.in_channels != 1:
            replacement = nn.Conv2d(
                1,
                projection.out_channels,
                kernel_size=projection.kernel_size,
                stride=projection.stride,
                padding=projection.padding,
                dilation=projection.dilation,
                bias=projection.bias is not None,
            )
            with torch.no_grad():
                replacement.weight.copy_(projection.weight.mean(dim=1, keepdim=True))
                if projection.bias is not None and replacement.bias is not None:
                    replacement.bias.copy_(projection.bias)
            self.model.segformer.encoder.patch_embeddings[0].proj = replacement

    def forward(self, image: Tensor) -> Tensor:
        logits = self.model(pixel_values=image).logits
        return functional.interpolate(
            logits, size=image.shape[-2:], mode="bilinear", align_corners=False
        )


def build_model(method: str, model_config: Mapping[str, Any]) -> nn.Module:
    """Build one declared benchmark model from its YAML ``model`` section."""
    config = dict(model_config)
    in_channels = int(config.get("in_channels", 1))
    if in_channels != 1:
        raise ValueError("the benchmark requires one-channel grayscale inputs")
    if method == "unet":
        return UNet(
            in_channels=in_channels,
            base_channels=int(config.get("base_channels", 32)),
            levels=int(config.get("levels", 4)),
        )
    if method == "unetpp":
        return UNetPP(
            in_channels=in_channels,
            base_channels=int(config.get("base_channels", 32)),
            levels=int(config.get("levels", 4)),
            deep_supervision=bool(config.get("deep_supervision", True)),
        )
    if method == "drunet":
        return RecurrentUNet(
            in_channels=in_channels,
            base_channels=int(config.get("base_channels", 32)),
            levels=int(config.get("levels", 4)),
            recurrent_steps=int(config.get("recurrent_steps", 2)),
            dropout=float(config.get("dropout", 0.2)),
        )
    if method == "ra_runet":
        alignment = config.get("alignment", {})
        if not bool(alignment.get("enabled", False)):
            # Dataset 1 is explicitly the recurrent backbone without domain
            # alignment.  Do not carry an unused, randomly initialised domain
            # head into its checkpoint or parameter-count column.
            return RecurrentUNet(
                in_channels=in_channels,
                base_channels=int(config.get("base_channels", 32)),
                levels=int(config.get("levels", 4)),
                recurrent_steps=int(config.get("recurrent_steps", 2)),
                dropout=float(config.get("dropout", 0.0)),
            )
        return RARUNet(
            in_channels=in_channels,
            base_channels=int(config.get("base_channels", 32)),
            levels=int(config.get("levels", 4)),
            recurrent_steps=int(config.get("recurrent_steps", 2)),
            dropout=float(config.get("dropout", 0.0)),
        )
    if method == "segformer_sub":
        return SegFormerBinary(
            str(config.get("pretrained_model", "nvidia/mit-b2")),
            pretrained=bool(config.get("pretrained", True)),
            local_files_only=bool(config.get("local_files_only", False)),
            pretrained_revision=str(config.get("pretrained_revision", "")),
            pretrained_path_env=str(config.get("pretrained_path_env", "")),
            pretrained_files=config.get("pretrained_files", {}),
        )
    if method == "lenet":
        return LeNetPatchClassifier(in_channels=in_channels)
    if method == "alexnet":
        return AlexNetPatchClassifier(
            in_channels=in_channels, dropout=float(config.get("dropout", 0.5))
        )
    if method == "cnn_ae":
        return CNNAEClassifier(
            in_channels=in_channels,
            latent_channels=int(config.get("latent_channels", 128)),
            dropout=float(config.get("dropout", 0.2)),
        )
    raise ValueError(f"unknown deep model method: {method}")
