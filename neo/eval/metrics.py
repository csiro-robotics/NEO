"""
This file contains NeRF metrics with masking capabilities.
This is a part of NeRFiller repository (https://github.com/ethanweber/nerfiller) modified for our own use.
"""

from abc import abstractmethod
from typing import Optional

import torch
from torch import nn
from torchmetrics.functional import structural_similarity_index_measure
from torchtyping import TensorType

from PIL import Image
import io
from kornia.feature import LoFTR
from kornia.color import rgb_to_grayscale

from neo.eval.lpips_utils import LearnedPerceptualImagePatchSimilarityWithMasking

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImageMetricModule(nn.Module):
    """Computes image metrics with masking capabilities.
    We assume that the pred and target inputs are in the range [0, 1].
    """

    def __init__(self, name: Optional[str] = "None"):
        super().__init__()
        self.populate_modules()
        self.name = name

    def populate_modules(self):
        """Populates the modules that will be used to compute the metric."""

    @abstractmethod
    def forward(
        self,
        preds: TensorType["bs", 3, "H", "W"],
        target: TensorType["bs", 3, "H", "W"],
        mask: Optional[TensorType["bs", 1, "H", "W"]] = None,
    ) -> TensorType["bs"]:
        """Computes the metric.
        Args:
            preds: Predictions.
            target: Ground truth.
            mask: Mask to use to only compute the metrics where the mask is True.
        Returns:
            Metric value.
        """


class PSNRModule(ImageMetricModule):
    """Computes PSNR with masking capabilities."""

    def __init__(self):
        super().__init__("PSNR")

    def forward(
        self,
        preds: TensorType["bs", 3, "H", "W"],
        target: TensorType["bs", 3, "H", "W"],
        mask: Optional[TensorType["bs", 1, "H", "W"]] = None,
    ) -> TensorType["bs"]:
        bs, h, w = preds.shape[0], preds.shape[2], preds.shape[3]
        hw = h * w

        preds_reshaped = preds.view(bs, 3, hw)
        target_reshaped = target.view(bs, 3, hw)
        num = (preds_reshaped - target_reshaped) ** 2
        # The non-masked version
        if mask is None:
            den = hw
        else:
            mask_reshaped = mask.view(bs, 1, hw)
            num = num * mask_reshaped
            den = mask_reshaped.sum(-1)
        mse = num.sum(-1) / den
        psnr = 10 * torch.log10(1.0 / (mse + 1e-6))
        psnr = psnr.mean(-1)

        return psnr
    

class MAEModule(ImageMetricModule):
    """Computes MAE with masking capabilities."""

    def __init__(self):
        super().__init__("MAE")

    def forward(
        self,
        preds: TensorType["bs", 3, "H", "W"],
        target: TensorType["bs", 3, "H", "W"],
        mask: Optional[TensorType["bs", 1, "H", "W"]] = None,
    ) -> TensorType["bs"]:
        bs, h, w = preds.shape[0], preds.shape[1], preds.shape[2]
        hw = h * w

        preds_reshaped = preds.view(bs, hw)
        target_reshaped = target.view(bs, hw)
        num = torch.abs(preds_reshaped - target_reshaped)

        # the non-masked version
        if mask is None:
            den = torch.count_nonzero(target_reshaped, dim=-1)

        else:
            mask_reshaped = mask.view(bs, hw)
            num = num * mask_reshaped
            den =  torch.count_nonzero(target_reshaped * mask_reshaped, dim=-1)
        mae = num.sum(-1) / (den + 1e-9)

        return mae


class SSIMModule(ImageMetricModule):
    """Computes PSNR with masking capabilities."""

    def __init__(self):
        super().__init__("SSIM")

    def forward(
        self,
        preds: TensorType["bs", 3, "H", "W"],
        target: TensorType["bs", 3, "H", "W"],
        mask: Optional[TensorType["bs", 1, "H", "W"]] = None,
    ) -> TensorType["bs"]:
        bs, h, w = preds.shape[0], preds.shape[2], preds.shape[3]
        hw = h * w

        _, ssim_image = structural_similarity_index_measure(
            preds=preds,
            target=target,
            reduction="none",
            data_range=1.0,
            return_full_image=True,
        )
        ssim_image = ssim_image.mean(1)  # average over the channels
        assert ssim_image.shape == (bs, h, w)

        # the non-masked version
        if mask is None:
            ssim = ssim_image.view(bs, hw).mean(1)

            return ssim

        # the masked version
        ssim_reshaped = ssim_image.view(bs, hw)
        mask_reshaped = mask.view(bs, hw)
        den = mask_reshaped.sum(-1, keepdim=True)
        ssim = (ssim_reshaped * mask_reshaped / den).sum(-1)

        return ssim


class LPIPSModule(ImageMetricModule):
    """Computes LPIPS with masking capabilities."""

    def __init__(self):
        super().__init__("LPIPS")

    def populate_modules(self):
        # by setting normalize=True, we assume that the pred and target inputs are in the range [0, 1]
        self.lpips_with_masking = LearnedPerceptualImagePatchSimilarityWithMasking(normalize=True).to(device)

    def forward(
        self,
        preds: TensorType["bs", 3, "H", "W"],
        target: TensorType["bs", 3, "H", "W"],
        mask: Optional[TensorType["bs", 1, "H", "W"]] = None,
    ) -> TensorType["bs"]:
        bs, h, w = preds.shape[0], preds.shape[2], preds.shape[3]
        hw = h * w

        with torch.no_grad():
            lpips_image = self.lpips_with_masking(preds, target)
        lpips_image = lpips_image.mean(1)  # average over the channels
        assert lpips_image.shape == (bs, h, w)

        # the non-masked version
        if mask is None:
            lpips = lpips_image.view(bs, hw).mean(1)

            return lpips

        # the masked version
        lpips_reshaped = lpips_image.view(bs, hw)
        mask_reshaped = mask.view(bs, hw)
        den = mask_reshaped.sum(-1, keepdim=True)
        lpips = (lpips_reshaped * mask_reshaped / den).sum(-1)

        return lpips


class CorrsModule(ImageMetricModule):
    """Compute the number of corrs between images with a confidence greater than the threshold."""

    def __init__(self):
        super().__init__("Corrs")

    def populate_modules(self):
        self.matcher = LoFTR(pretrained="indoor").to(device)
        self.confidence_thresh = 0.8

    def forward(
        self,
        preds: TensorType["bs", 3, "H", "W"],
        target: TensorType["bs", 3, "H", "W"],
        mask: Optional[TensorType["bs", 1, "H", "W"]] = None,
    ) -> TensorType["bs"]:
        bs, h, w = preds.shape[0], preds.shape[2], preds.shape[3]
        hw = h * w

        # assert mask is None

        with torch.no_grad():
            scores = []
            for i in range(bs):
                image0 = rgb_to_grayscale(preds[i : i + 1])
                image1 = rgb_to_grayscale(target[i : i + 1])

                if mask is None:
                    input_ = {"image0": image0, "image1": image1}

                else:
                    mask_ = mask[i : i + 1]
                    input_ = {"image0": image0 * mask_, "image1": image1 * mask_}


                correspondences_dict = self.matcher(input_)
                num_corrs = float((correspondences_dict["confidence"] >= self.confidence_thresh).sum())
                scores.append(num_corrs)
            scores = torch.tensor(scores)

        return scores
