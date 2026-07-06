"""
MC-Dropout uncertainty scoring for YOLO detection, bridging BaaL to an object
detector. Implements the design doc's "Active learning: ... Select high-
uncertainty vehicles (MC Dropout)" (Sec. 4) and "Uncertainty-based escalation
(MC Dropout) to human review" (Sec. 6).

THE BRIDGING PROBLEM:
BaaL's heuristics (BALD, Entropy, ...) are built for classification, where
each sample has ONE fixed-size probability vector that sums to 1 across
classes. YOLO detection doesn't look like that: each image produces a dense
grid of thousands of anchors, and each anchor has an INDEPENDENT (multi-label,
sigmoid - not softmax) class-probability vector, since more than one damage
type can be present in the same region. Stock YOLO11 also has zero Dropout
layers anywhere (it relies on augmentation for regularization, not dropout),
so there's nothing for BaaL's `patch_module` to convert - MC-Dropout has to be
added, not just switched on.

THE BRIDGE, concretely:
  1. Insert BaaL's `Dropout2d` (an MC-Dropout module that stays stochastic in
     eval() mode, unlike plain `nn.Dropout`) into the Detect head's
     classification branch (`cv3`) only. The box-regression branch (`cv2`) is
     left untouched - we want class uncertainty, not localization jitter.
  2. Run T stochastic forward passes on the SAME preprocessed image tensor,
     reading the raw, pre-NMS, per-anchor class-score grid (shape [nc, A])
     each time, so anchor order is identical pass to pass. NMS'd output
     (the normal `model.predict()` path) would keep a different, differently-
     ordered, differently-SIZED subset of boxes each pass - there'd be nothing
     to compare anchor-by-anchor across passes.
  3. For each class independently, build a proper 2-way distribution
     [p_c, 1-p_c] per anchor per pass and feed it to BaaL's actual
     `BALD().compute_score(...)`. Summing to 1 satisfies BaaL's internal
     `to_prob` check, so this runs BaaL's real BALD computation rather than
     reimplementing the formula, and avoids BaaL silently softmax-
     renormalizing probabilities across classes that aren't mutually
     exclusive (which is what would happen if we handed it the raw
     [nc, A, T] multi-label tensor directly).
  4. Reduce per-anchor-per-class BALD scores to one scalar per image: max
     over classes (worst-case class uncertainty at that anchor), then mean of
     the top-K anchors (the K most plausible damage regions), so thousands of
     confidently-empty background anchors don't dilute the signal.

LIMITATION worth being upfront about: adding dropout to cv3 changes the
network slightly from the plain fine-tuned baseline, so the checkpoint used
for uncertainty scoring should be re-validated (not assumed identical to the
non-dropout baseline's mAP) if it's also the checkpoint used for real
detections.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from baal.active.heuristics import BALD
from baal.bayesian.dropout import Dropout2d as MCDropout2d
from ultralytics import YOLO
from ultralytics.models.yolo.detect.predict import DetectionPredictor


def inject_mc_dropout(yolo_model: YOLO, p: float = 0.1) -> YOLO:
    """Insert MC-Dropout into the Detect head's classification branch (cv3) only."""
    detect_head = yolo_model.model.model[-1]
    for branch in detect_head.cv3:
        branch.insert(len(branch) - 1, MCDropout2d(p=p))
    return yolo_model


class MCDropoutYOLO:
    """Wraps a fine-tuned YOLO checkpoint to produce per-image epistemic
    uncertainty scores via MC-Dropout + BaaL's BALD heuristic."""

    def __init__(
        self,
        weights: str,
        dropout_p: float = 0.1,
        n_passes: int = 10,
        device: str = "cuda",
        imgsz: int = 640,
    ):
        self.yolo = YOLO(weights)
        inject_mc_dropout(self.yolo, p=dropout_p)
        self.model = self.yolo.model.to(device).eval()
        self.n_passes = n_passes
        self.device = device
        self.nc = len(self.yolo.names)
        self.names = self.yolo.names

        # Reuse Ultralytics' own preprocessing (letterbox resize, BGR->RGB,
        # normalize) instead of reimplementing it, to avoid subtle mismatches
        # with how the model was trained.
        self.predictor = DetectionPredictor(overrides={"imgsz": imgsz, "device": device, "verbose": False})
        self.predictor.setup_model(self.model)
        # setup_model doesn't set this (it's normally computed later, in the full
        # stream_inference pipeline we're bypassing) - pre_transform needs it directly.
        self.predictor.imgsz = (imgsz, imgsz)

    def _dense_class_probs(self, im_tensor: torch.Tensor) -> torch.Tensor:
        """One stochastic forward pass -> [nc, num_anchors] sigmoid class scores."""
        with torch.no_grad():
            raw = self.model(im_tensor)[0]  # [1, 4+nc, A], cls channels already sigmoid-activated
        return raw[0, 4:4 + self.nc, :]

    def image_uncertainty(self, image_path: str, top_k: int = 20) -> dict:
        im = cv2.imread(image_path)
        if im is None:
            raise FileNotFoundError(image_path)
        im_tensor = self.predictor.preprocess([im])

        passes = [self._dense_class_probs(im_tensor) for _ in range(self.n_passes)]
        stacked = torch.stack(passes, dim=-1).cpu().numpy()  # [nc, A, T]

        bald = BALD()
        per_class_scores = []
        for c in range(self.nc):
            p_c = stacked[c]  # [A, T]
            probs_2way = np.stack([p_c, 1.0 - p_c], axis=1)  # [A, 2, T] - sums to 1, so
            per_class_scores.append(bald.compute_score(probs_2way))  # BaaL treats it as a real distribution
        per_anchor = np.max(np.stack(per_class_scores, axis=0), axis=0)  # [A], worst class per anchor

        k = min(top_k, per_anchor.shape[0])
        top_idx = np.argpartition(per_anchor, -k)[-k:]
        image_score = float(per_anchor[top_idx].mean())

        mean_probs = stacked.mean(axis=-1)  # [nc, A]
        per_class_peak_confidence = mean_probs.max(axis=1)

        return {
            "image": str(image_path),
            "uncertainty": image_score,
            "per_class_peak_confidence": {
                name: float(per_class_peak_confidence[i]) for i, name in self.names.items()
            },
        }

    def rank_pool(self, image_paths: list, top_k: int = 20) -> list:
        """Score a pool of unlabeled images, most-uncertain first."""
        scored = [self.image_uncertainty(p, top_k=top_k) for p in image_paths]
        return sorted(scored, key=lambda r: r["uncertainty"], reverse=True)
