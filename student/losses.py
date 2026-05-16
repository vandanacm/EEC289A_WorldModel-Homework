"""Student one-step plus rollout loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .rollout import open_loop_rollout


def _regression(pred: torch.Tensor, target: torch.Tensor, *, loss_type: str) -> torch.Tensor:
    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(pred, target, beta=0.05)
    if loss_type == "mse":
        return F.mse_loss(pred, target)
    raise ValueError(f"Unknown loss_type={loss_type!r}; expected 'mse' or 'smooth_l1'.")


def one_step_delta_loss(
    model,
    states: torch.Tensor,
    actions: torch.Tensor,
    normalizer,
    *,
    loss_type: str = "mse",
) -> torch.Tensor:
    obs = states[:, :-1].reshape(-1, states.shape[-1])
    act = actions.reshape(-1, actions.shape[-1])
    target_delta = (states[:, 1:] - states[:, :-1]).reshape(-1, states.shape[-1])
    obs_norm = normalizer.normalize_obs(obs)
    act_norm = normalizer.normalize_act(act)
    target_norm = normalizer.normalize_delta(target_delta)
    pred_norm, _ = model(obs_norm, act_norm, None)
    return _regression(pred_norm, target_norm, loss_type=loss_type)


def rollout_loss(
    model,
    states: torch.Tensor,
    actions: torch.Tensor,
    normalizer,
    warmup_steps: int,
    horizon: int,
    *,
    milestones: list[int] | None = None,
    loss_type: str = "mse",
) -> torch.Tensor:
    needed_states = int(warmup_steps) + int(horizon) + 1
    if states.shape[1] < needed_states:
        raise ValueError(
            "training.train_sequence_length is too short for rollout loss: "
            f"need at least {needed_states - 1} actions for warmup={warmup_steps}, horizon={horizon}."
        )
    max_start = states.shape[1] - needed_states
    if max_start > 0:
        start = int(torch.randint(0, max_start + 1, (), device=states.device).item())
    else:
        start = 0
    sub_states = states[:, start : start + needed_states]
    sub_actions = actions[:, start : start + int(warmup_steps) + int(horizon)]
    preds = open_loop_rollout(model, sub_states, sub_actions, normalizer, warmup_steps=warmup_steps, horizon=horizon)
    targets = sub_states[:, warmup_steps + 1 : warmup_steps + 1 + horizon]
    pred_norm = normalizer.normalize_obs(preds)
    target_norm = normalizer.normalize_obs(targets)

    if milestones:
        caps = sorted({int(m) for m in milestones if 1 <= int(m) <= int(horizon)})
        if not caps:
            caps = [int(horizon)]
        parts = []
        for h in caps:
            parts.append(_regression(pred_norm[:, :h], target_norm[:, :h], loss_type=loss_type))
        return torch.stack(parts, dim=0).mean()

    return _regression(pred_norm, target_norm, loss_type=loss_type)


def compute_loss(model, batch: dict[str, torch.Tensor], normalizer, cfg: dict):
    loss_cfg = cfg["loss"]
    states = batch["states"]
    actions = batch["actions"]
    one_type = str(loss_cfg.get("one_step_loss", "mse"))
    roll_type = str(loss_cfg.get("rollout_loss", "mse"))
    one = one_step_delta_loss(model, states, actions, normalizer, loss_type=one_type)
    horizon = int(loss_cfg.get("rollout_train_horizon", 5))
    warmup = int(cfg["eval"].get("warmup_steps", 5))
    milestones = loss_cfg.get("rollout_milestones")
    if milestones is not None:
        milestones = [int(m) for m in milestones]
    roll = rollout_loss(
        model,
        states,
        actions,
        normalizer,
        warmup_steps=warmup,
        horizon=horizon,
        milestones=milestones,
        loss_type=roll_type,
    )
    total = float(loss_cfg.get("one_step_weight", 1.0)) * one + float(loss_cfg.get("rollout_weight", 0.3)) * roll
    return total, {
        "loss/total": float(total.detach().cpu()),
        "loss/one_step": float(one.detach().cpu()),
        "loss/rollout": float(roll.detach().cpu()),
    }
