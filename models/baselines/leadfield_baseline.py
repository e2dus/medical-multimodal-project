# -*- coding: utf-8 -*-
"""
Baseline for LeadFieldECGRecon ablation study.

Same U-Net + source_head(→257) architecture, but the last projection
is a LEARNED Conv1d(257, 12) instead of the fixed ECGSIM transfer matrix.

This isolates the effect of physics vs learned projection.
"""

from __future__ import annotations
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# 동일한 빌딩 블록 재사용
from leadfield_1to12 import (
    ConvBlock, DownBlock, UpBlock,
    LEADS, parse_target_leads,
    reconstruction_loss, lead_consistency_loss,
)


class LeadFieldBaseline(nn.Module):
    """
    LeadFieldECGRecon과 동일한 구조, 마지막 레이어만 다름:
    - LeadFieldECGRecon: source_head → L (ECGSIM 고정) → 12-lead
    - LeadFieldBaseline: source_head → Conv1d (학습 가능)  → 12-lead
    """

    def __init__(
        self,
        in_ch: int = 1,
        base_ch: int = 48,
        n_sources: int = 257,
        output_indices: Optional[List[int]] = None,
    ):
        super().__init__()
        if output_indices is None:
            output_indices = list(range(12))
        self.output_leads = [LEADS[i] for i in output_indices]
        self.out_ch = len(output_indices)
        self.n_sources = n_sources

        # ── 동일한 U-Net backbone ──
        self.enc1 = ConvBlock(in_ch, base_ch)
        self.enc2 = DownBlock(base_ch, base_ch * 2)
        self.enc3 = DownBlock(base_ch * 2, base_ch * 4)
        self.enc4 = DownBlock(base_ch * 4, base_ch * 8)
        self.bottleneck = ConvBlock(base_ch * 8, base_ch * 8)
        self.up3 = UpBlock(base_ch * 8, base_ch * 4, base_ch * 4)
        self.up2 = UpBlock(base_ch * 4, base_ch * 2, base_ch * 2)
        self.up1 = UpBlock(base_ch * 2, base_ch, base_ch)

        # ── 동일한 source_head (→257 채널) ──
        self.source_head = nn.Sequential(
            nn.Conv1d(base_ch, base_ch, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(base_ch, self.n_sources, 1),
        )

        # ── 차이점: 학습 가능한 마지막 레이어 (vs ECGSIM 고정) ──
        self.projection = nn.Conv1d(self.n_sources, self.out_ch, 1, bias=False)

    def forward(self, x: torch.Tensor, return_explain: bool = False):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.bottleneck(e4)
        d3 = self.up3(b, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        source = self.source_head(d1)   # [B, 257, T]
        pred = self.projection(source)  # [B, 12, T] ← 학습된 가중치

        if not return_explain:
            return pred
        explain = {
            "source_activation": source,
            "physics_ecg": pred,            # baseline에서는 physics=pred
            "unet_residual": torch.zeros_like(pred),
            "leadfield_matrix": self.projection.weight.data.squeeze(-1).unsqueeze(0).expand(x.size(0), -1, -1),
            "leadfield_base": self.projection.weight.data.squeeze(-1),
            "output_leads": self.output_leads,
        }
        return pred, explain
