"""Probe: structural and behavioural facts about the released checkpoints (handoff §3, §5.1, §9).

    uv run python scripts/probes/model_facts.py              # 5M
    uv run python scripts/probes/model_facts.py --size 28M

Prints: module naming, attention counts, head size, gradients of a target-only loss with respect
to covariate inputs, the role firewall under noise, and the single-row variate-attention result.
Runs on CPU. Needs the Hub the first time. Expected output is recorded in docs/VERIFICATION.md.
"""

from __future__ import annotations

import argparse
import copy

import torch
from _common import PATCH, ROLE_KNOWN_COV, ROLE_PAST_COV, ROLE_TARGET, load


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", default="5M", choices=["5M", "28M", "123M", "209M"])
    args = parser.parse_args()
    torch.manual_seed(0)
    model = load(args.size).model
    d_model = model.config.d_model

    print(f"== Timestral-{args.size} ==")
    names = [name for name, _ in model.named_modules()]
    legacy = [n for n in names if "q_proj" in n or "v_proj" in n or "variate_attn" in n]
    print(f"[names] modules matching q_proj / v_proj / variate_attn: {legacy}")
    attention = [(n, m.is_time) for n, m in model.named_modules() if type(m).__name__ == "Attention"]
    variate = [n for n, is_time in attention if not is_time]
    n_all, n_variate = len(attention), len(variate)
    print(f"[names] attention modules: {n_all} (temporal {n_all - n_variate}, variate {n_variate})")
    print(f"[names] variate attention at: {variate}")
    first_variate = model.get_submodule(variate[0])
    print(f"[names] attention parameters: {[n for n, _ in first_variate.named_parameters()]}")
    print(f"[names] wQKV weight shape: {tuple(first_variate.wQKV.weight.shape)} (d_model = {d_model})")
    total = sum(p.numel() for p in model.parameters())
    head = sum(p.numel() for p in model.head.parameters())
    print(f"[size]  parameters: total {total:,} | head {head:,} ({100 * head / total:.1f}%) | formula 4D^2+3360D = "
          f"{4 * d_model**2 + 3360 * d_model:,}")  # fmt: skip

    # One series: target, past-only covariate, known-future covariate.
    rows, context, horizon = 3, 8 * PATCH, 2
    roles = torch.tensor([ROLE_TARGET, ROLE_PAST_COV, ROLE_KNOWN_COV])
    x = torch.randn(rows, context).cumsum(-1).requires_grad_(True)
    obs = torch.ones(rows, context, dtype=torch.bool)
    gid = torch.zeros(rows, dtype=torch.long)
    kf = torch.zeros(rows, horizon * PATCH)
    kf[2] = torch.randn(horizon * PATCH)
    kf.requires_grad_(True)
    kfm = torch.zeros(rows, horizon * PATCH, dtype=torch.bool)
    kfm[2] = True

    quantiles = model.predict_block(x, obs, gid, horizon, roles, kf, kfm)
    quantiles[0].abs().mean().backward()  # loss on the target row only
    grads = (x.grad[0], x.grad[1], x.grad[2], kf.grad[2])
    labels = ("target context", "past-only context", "known-future context", "known-future horizon")
    print("[grad]  |d(target-only loss)/d(input)| summed: " + ", ".join(
        f"{label} {g.abs().sum():.2e}" for label, g in zip(labels, grads, strict=True)))  # fmt: skip
    model.zero_grad()

    x, kf = x.detach(), kf.detach()
    with torch.no_grad():
        base = model.predict_block(x, obs, gid, horizon, roles, kf, kfm)
        for name, row in (("target", 0), ("past-only", 1), ("known-future", 2)):
            noisy = x.clone()
            noisy[row, 40:200] += 3 * torch.randn(160)
            out = model.predict_block(noisy, obs, gid, horizon, roles, kf, kfm)
            changes = [f"{(out[i] - base[i]).abs().max():.2e}" for i in range(3)]
            print(f"[wall]  noise on {name:12s} context -> max change in (target, past-only, known-future): {changes}")

        # Single-row series: query and key rows of variate attention have no effect.
        xu = torch.randn(1, context).cumsum(-1)
        ou = torch.ones(1, context, dtype=torch.bool)
        gu = torch.zeros(1, dtype=torch.long)
        base = model.predict_block(xu, ou, gu, horizon)
        for label, rows_slice in (("query+key rows", slice(0, 2 * d_model)), ("value rows", slice(2 * d_model, None))):
            perturbed = copy.deepcopy(model)
            for module in perturbed.modules():
                if type(module).__name__ == "Attention" and not module.is_time:
                    weight = module.wQKV.weight
                    weight[rows_slice] += 0.5 * torch.randn_like(weight[rows_slice])
            change = (perturbed.predict_block(xu, ou, gu, horizon) - base).abs().max()
            print(f"[1row]  univariate input, perturb variate {label:14s} -> max output change {change:.2e}")


if __name__ == "__main__":
    main()
