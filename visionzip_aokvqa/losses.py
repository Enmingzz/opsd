from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_forward_kl(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    if teacher_logits.ndim != 2 or student_logits.ndim != 2:
        raise ValueError("KL logits must be [T, vocab].")
    if teacher_logits.shape != student_logits.shape:
        raise ValueError(f"Teacher/student logits must align by token index: {teacher_logits.shape} vs {student_logits.shape}")
    temperature = float(temperature)
    teacher_probs = F.softmax(teacher_logits.float() / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits.float() / temperature, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature**2)


def compute_generalized_jsd(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    beta: float = 0.0,
    temperature: float = 1.0,
    top_k: int | None = None,
    token_clip: float | None = None,
    clip_mode: str = "token",
) -> torch.Tensor:
    """Official OPSD-style generalized JSD over generated token positions.

    With beta=0 this is forward KL from teacher to student, matching the main
    OPSD setting.  With beta=1 it becomes reverse KL.  Intermediate beta values
    use the generalized Jensen-Shannon mixture from the official trainer.
    """

    if teacher_logits.shape != student_logits.shape:
        raise ValueError(f"Teacher/student logits must align: {teacher_logits.shape} vs {student_logits.shape}")
    if teacher_logits.ndim not in {2, 3}:
        raise ValueError("Generalized JSD logits must be [T, vocab] or [B, T, vocab].")
    if int(teacher_logits.shape[-2]) <= 0:
        raise ValueError("Generalized JSD requires at least one token position.")

    temperature = float(temperature)
    beta = float(beta)
    if beta < 0.0 or beta > 1.0:
        raise ValueError(f"OPSD beta must be in [0, 1], got {beta}.")

    teacher_logits = teacher_logits.float() / temperature
    student_logits = student_logits.float() / temperature

    if top_k is not None and int(top_k) > 0:
        k = min(int(top_k), int(teacher_logits.shape[-1]))
        _, top_k_indices = torch.topk(teacher_logits, k=k, dim=-1)
        teacher_logits = torch.gather(teacher_logits, dim=-1, index=top_k_indices)
        student_logits = torch.gather(student_logits, dim=-1, index=top_k_indices)

    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
    student_log_probs = F.log_softmax(student_logits, dim=-1)

    if beta == 0.0:
        jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
    elif beta == 1.0:
        jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
    else:
        beta_t = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
        mixture_log_probs = torch.logsumexp(
            torch.stack(
                [
                    student_log_probs + torch.log1p(-beta_t),
                    teacher_log_probs + torch.log(beta_t),
                ]
            ),
            dim=0,
        )
        kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
        kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)
        jsd = beta_t * kl_teacher + (1.0 - beta_t) * kl_student

    normalized_clip_mode = str(clip_mode).lower()
    if token_clip is not None and float(token_clip) > 0.0 and normalized_clip_mode in {"official", "vocab", "vocab_element"}:
        jsd = jsd.clamp(max=float(token_clip))
        return jsd.sum(dim=-1).mean()

    token_loss = jsd.sum(dim=-1).clamp_min(0.0)
    if token_clip is not None and float(token_clip) > 0.0 and normalized_clip_mode in {"token", "token_sum"}:
        token_loss = token_loss.clamp(max=float(token_clip))
    elif normalized_clip_mode not in {"token", "token_sum"}:
        raise ValueError(f"Unsupported JSD clip_mode={clip_mode!r}. Use 'token' or 'official'.")
    return token_loss.mean()


def compute_token_ce(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError("CE logits must be [T, vocab].")
    target_ids = target_ids.to(device=logits.device, dtype=torch.long).reshape(-1)
    if int(target_ids.numel()) != int(logits.shape[0]):
        raise ValueError(f"CE target length {int(target_ids.numel())} != logits length {int(logits.shape[0])}.")
    return F.cross_entropy(logits.float(), target_ids)


def compute_sequence_logprob(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    token_ids = token_ids.to(device=logits.device, dtype=torch.long).reshape(-1)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return log_probs.gather(dim=-1, index=token_ids[:, None]).squeeze(-1).sum()


def grpo_policy_loss(sequence_logprobs: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
    if sequence_logprobs.shape != advantages.shape:
        raise ValueError("GRPO sequence_logprobs and advantages must have matching shape.")
    return -(sequence_logprobs.float() * advantages.float().detach()).mean()
