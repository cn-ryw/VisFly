import inspect
import math
import textwrap

import torch

from VisFly.utils.algorithms.PPO import PPO


def clipped_policy_loss(log_ratio, advantages, clip_range):
    lower = math.log1p(-clip_range)
    upper = math.log1p(clip_range)
    # Evaluate only the ratio selected by the surrogate. The discarded exp()
    # branch otherwise contributes inf * 0 during backpropagation.
    selected = torch.where(
        advantages >= 0, log_ratio.clamp_max(upper), log_ratio.clamp_min(lower),
    )
    log_scale = max(0.0, selected.detach().max().item() - 30.0)
    loss = -(advantages * (selected - log_scale).exp()).mean()
    fraction = ((log_ratio < lower) | (log_ratio > upper)).float().mean().item()
    return loss, fraction, log_scale


def clip_scaled_gradients(parameters, max_norm, log_scale):
    if log_scale == 0.0:
        try:
            return torch.nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True)
        except RuntimeError:
            # The iterator is materialized by the caller for the double-precision retry.
            pass
    gradients = [param.grad for param in parameters if param.grad is not None]
    if not gradients:
        return
    with torch.no_grad():
        norms = torch.stack([torch.linalg.vector_norm(grad, dtype=torch.float64) for grad in gradients])
        norm = torch.linalg.vector_norm(norms).item()
        if not math.isfinite(norm):
            raise FloatingPointError("Nonfinite gradients even after scaling")
        if norm == 0.0:
            return
        # The true gradient is exp(log_scale) times the stored gradient.
        # Fold the unscale and the usual norm clip into one finite multiplier.
        denominator = norm + 1e-6 * math.exp(-log_scale)
        factor = math.exp(min(log_scale, math.log(max_norm) - math.log(denominator)))
        for gradient in gradients:
            gradient.mul_(factor)


def restore_metric(value, log_scale):
    raw = value.detach().item()
    if raw == 0.0 or log_scale == 0.0:
        return raw
    magnitude = math.log(abs(raw)) + log_scale
    if magnitude > math.log(float.fromhex("0x1.fffffffffffffp+1023")):
        return math.copysign(float("inf"), raw)
    return math.copysign(math.exp(magnitude), raw)


def build_train():
    source = textwrap.dedent(inspect.getsource(PPO.train))
    start = source.index("            # ratio between old and new policy")
    end = source.index("            if self.clip_range_vf is None:", start)
    replacement = """            log_ratio = log_prob - rollout_data.old_log_prob
            policy_loss, clip_fraction, log_scale = clipped_policy_loss(log_ratio, advantages, clip_range)
            pg_losses.append(restore_metric(policy_loss, log_scale))
            clip_fractions.append(clip_fraction)

"""
    source = source[:start] + replacement + source[end:]
    original_penalty = "+ self.ent_coef * entropy_loss\n                    + self.vf_coef * value_loss"
    assert source.count(original_penalty) == 1
    source = source.replace(
        original_penalty, "+ inv_scale(log_scale) * (self.ent_coef * entropy_loss + self.vf_coef * value_loss)"
    )
    start = source.index("            th.nn.utils.clip_grad_norm_(")
    end = source.index("            self.policy.optimizer.step()", start)
    source = (
        source[:start]
        + "            clip_scaled_gradients(list(self.policy.parameters()), self.max_grad_norm, log_scale)\n\n"
        + source[end:]
    )
    assert source.count("th.exp(log_ratio)") == 1
    source = source.replace("th.exp(log_ratio)", "th.exp(log_ratio.double())")
    source = source.replace('self.logger.record("train/loss", loss.item())', 'self.logger.record("train/loss", restore_metric(loss, log_scale))')
    scope = dict(
        PPO.train.__globals__,
        clipped_policy_loss=clipped_policy_loss,
        clip_scaled_gradients=clip_scaled_gradients,
        restore_metric=restore_metric,
        inv_scale=lambda scale: math.exp(-scale),
    )
    exec(compile(source, "<VisFly PPO train with stable clipping and gradient norm>", "exec"), scope)
    return scope["train"]


stable_train = build_train()
