from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class VisionZipOutput:
    image_embeds: torch.Tensor
    keep_indices: torch.Tensor
    metadata: dict[str, Any]


def _target_count(num_tokens: int, retention_ratio: float) -> int:
    if num_tokens <= 0:
        raise ValueError("VisionZip requires at least one visual token.")
    ratio = min(max(float(retention_ratio), 0.0), 1.0)
    return min(num_tokens, max(1, int(round(num_tokens * ratio))))


def _sorted_unique(indices: torch.Tensor, num_tokens: int) -> torch.Tensor:
    indices = indices.to(dtype=torch.long).flatten().unique(sorted=True)
    if indices.numel() == 0:
        raise ValueError("VisionZip must keep at least one visual token.")
    if int(indices.min().item()) < 0 or int(indices.max().item()) >= int(num_tokens):
        raise IndexError("VisionZip produced out-of-range keep indices.")
    return indices


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1, eps=1e-6)


class VisionZipPruner:
    """VisionZip selector/merger for pre-LLM visual tokens.

    The official Qwen2.5-VL VisionZip release selects dominant tokens by
    last-layer visual attention, keeps about 5% of the original visual tokens as
    a contextual subset from the non-dominant pool, and merges dropped
    contextual tokens into their nearest kept contextual token. This class keeps
    that behavior but exposes the total retention ratio as the experiment knob.
    """

    def __init__(
        self,
        contextual_fraction: float = 0.05,
        allow_embedding_fallback: bool = False,
    ) -> None:
        self.contextual_fraction = float(contextual_fraction)
        self.allow_embedding_fallback = bool(allow_embedding_fallback)

    def select_and_merge(
        self,
        vision_embeds: torch.Tensor,
        grid_thw: torch.Tensor | None,
        retention_ratio: float,
        question: str | None = None,
        metadata: dict[str, Any] | None = None,
        attn_scores: torch.Tensor | None = None,
        attn_key: torch.Tensor | None = None,
    ) -> VisionZipOutput:
        del grid_thw, question
        if vision_embeds.ndim != 2:
            raise ValueError(f"vision_embeds must be [N, D], got {tuple(vision_embeds.shape)}.")
        num_tokens = int(vision_embeds.shape[0])
        target = _target_count(num_tokens, retention_ratio)
        if target >= num_tokens:
            keep = torch.arange(num_tokens, device=vision_embeds.device, dtype=torch.long)
            return VisionZipOutput(
                image_embeds=vision_embeds,
                keep_indices=keep,
                metadata={
                    "visionzip_exact_metrics": attn_scores is not None and attn_key is not None,
                    "visionzip_target_tokens": target,
                    "visionzip_dominant_tokens": num_tokens,
                    "visionzip_contextual_tokens": 0,
                    "visionzip_merged_tokens": 0,
                    **(metadata or {}),
                },
            )

        if attn_scores is None:
            if not self.allow_embedding_fallback:
                raise RuntimeError(
                    "VisionZip attention scores are unavailable. Use a Qwen2.5-VL visual module compatible with "
                    "VisionZip metrics or set allow_embedding_fallback=true for a documented smoke-test fallback."
                )
            attn_scores = vision_embeds.float().norm(dim=-1)
        attn_scores = attn_scores.to(device=vision_embeds.device, dtype=torch.float32).flatten()
        if int(attn_scores.numel()) != num_tokens:
            raise ValueError(f"VisionZip attention scores length {int(attn_scores.numel())} != tokens {num_tokens}.")

        contextual_count = int(num_tokens * self.contextual_fraction)
        if self.contextual_fraction > 0:
            contextual_count = max(contextual_count, 1)
        contextual_count = min(max(contextual_count, 0), target)
        dominant_count = target - contextual_count

        if dominant_count > 0:
            dominant_indices = torch.topk(attn_scores, k=dominant_count, dim=0).indices
        else:
            dominant_indices = torch.empty(0, device=vision_embeds.device, dtype=torch.long)
        dominant_mask = torch.zeros(num_tokens, dtype=torch.bool, device=vision_embeds.device)
        dominant_mask[dominant_indices] = True
        contextual_pool = (~dominant_mask).nonzero(as_tuple=True)[0]

        if contextual_count > int(contextual_pool.numel()):
            contextual_count = int(contextual_pool.numel())
        if contextual_count > 0:
            step = max(1, int(contextual_pool.numel()) // contextual_count)
            pool_positions = torch.arange(0, int(contextual_pool.numel()), step, device=vision_embeds.device)[
                :contextual_count
            ]
            contextual_indices = contextual_pool[pool_positions]
        else:
            contextual_indices = torch.empty(0, device=vision_embeds.device, dtype=torch.long)

        keep = _sorted_unique(torch.cat([dominant_indices, contextual_indices]), num_tokens)
        if int(keep.numel()) < target:
            keep_mask = torch.zeros(num_tokens, dtype=torch.bool, device=vision_embeds.device)
            keep_mask[keep] = True
            fill = (~keep_mask).nonzero(as_tuple=True)[0][: target - int(keep.numel())]
            keep = _sorted_unique(torch.cat([keep, fill]), num_tokens)

        merged_embeds = vision_embeds.clone()
        merged_count = 0
        if contextual_indices.numel() > 0:
            dropped_contextual = contextual_pool[~torch.isin(contextual_pool, contextual_indices)]
            if dropped_contextual.numel() > 0:
                metric = attn_key
                if metric is None:
                    if not self.allow_embedding_fallback:
                        raise RuntimeError("VisionZip contextual merge requires visual keys or embedding fallback.")
                    metric = vision_embeds
                metric = metric.squeeze(0) if metric.ndim == 3 and int(metric.shape[0]) == 1 else metric
                metric = metric.to(device=vision_embeds.device)
                if metric.ndim != 2 or int(metric.shape[0]) != num_tokens:
                    raise ValueError(f"VisionZip merge metric must be [N, D], got {tuple(metric.shape)}.")
                metric_norm = _normalize(metric)
                dropped_metric = metric_norm[dropped_contextual]
                target_metric = metric_norm[contextual_indices]
                assign = torch.matmul(dropped_metric, target_metric.T).argmax(dim=-1)
                one_hot = F.one_hot(assign, num_classes=int(contextual_indices.numel())).to(dtype=vision_embeds.dtype)
                counts = one_hot.sum(dim=0).clamp_min(1).to(dtype=vision_embeds.dtype).unsqueeze(-1)
                aggregated = torch.matmul(one_hot.T, vision_embeds[dropped_contextual]) / counts
                merged_embeds[contextual_indices] = vision_embeds[contextual_indices] + aggregated
                merged_count = int(dropped_contextual.numel())

        return VisionZipOutput(
            image_embeds=merged_embeds,
            keep_indices=keep,
            metadata={
                "visionzip_exact_metrics": attn_key is not None,
                "visionzip_target_tokens": int(target),
                "visionzip_dominant_tokens": int(dominant_count),
                "visionzip_contextual_tokens": int(contextual_indices.numel()),
                "visionzip_merged_tokens": int(merged_count),
                "visionzip_contextual_fraction": self.contextual_fraction,
                **(metadata or {}),
            },
        )


def compute_kl_teacher_forward(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    if teacher_logits.shape != student_logits.shape:
        raise ValueError(f"KL logits shape mismatch: {tuple(teacher_logits.shape)} vs {tuple(student_logits.shape)}")
    temperature = float(temperature)
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    teacher_probs = F.softmax(teacher_logits.float() / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits.float() / temperature, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature**2)


def grpo_group_advantages(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rewards = rewards.float().flatten()
    if rewards.numel() == 0:
        raise ValueError("GRPO rewards must be non-empty.")
    return (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp_min(float(eps))
