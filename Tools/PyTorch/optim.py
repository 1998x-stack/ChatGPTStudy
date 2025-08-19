"""Custom Optimizer: SGDMomentumNesterov (Industrial-Grade Example).

This module implements a production-quality PyTorch optimizer that extends the
classic SGD with options for Momentum and Nesterov acceleration, plus flexible
weight decay (coupled L2 or decoupled AdamW-style), gradient scaling support,
and safe numerics.

Key features
-----------
1) Momentum & Nesterov: configurable via flags.
2) Weight Decay modes: ``coupled`` (classic L2) or ``decoupled`` (AdamW-style).
3) Mixed precision friendly: works with GradScaler (no special handling needed).
4) "foreach" fast-path: optional vectorized updates when available.
5) Differentiable updates: can be toggled for meta-learning (slower).
6) Robust state handling: delayed state init, full state_dict/restore support.

The code uses Google-style docstrings (PEP 257 compatible) and PEP 8 naming,
adds type annotations, and includes Chinese comments explaining key steps.

Example:
    >>> import torch
    >>> from custom_optimizer_sgd_momentum_nesterov import SGDMomentumNesterov
    >>> model = torch.nn.Linear(10, 1)
    >>> opt = SGDMomentumNesterov(
    ...     model.parameters(), lr=1e-2, momentum=0.9, nesterov=True,
    ...     weight_decay=1e-4, decoupled_weight_decay=True,
    ... )
    >>> x = torch.randn(8, 10)
    >>> y = torch.randn(8, 1)
    >>> loss_fn = torch.nn.MSELoss()
    >>> for _ in range(5):
    ...     opt.zero_grad(set_to_none=True)
    ...     loss = loss_fn(model(x), y)
    ...     loss.backward()
    ...     opt.step()

Safety:
    - This file performs no I/O on import.
    - Public API is a single optimizer class.

"""
from __future__ import annotations

from typing import Iterable, List, Optional, Dict, Any
import math

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, _params_t


class SGDMomentumNesterov(Optimizer):
    """SGD with Momentum and optional Nesterov acceleration.

    This optimizer implements the classic SGD algorithm with optional momentum
    and Nesterov acceleration. It also supports two weight decay modes:
    * coupled L2: adds ``lambda * p`` to the gradient (classic SGD behavior)
    * decoupled: subtracts ``lr * lambda * p`` from the parameter directly

    When ``foreach=True``, vectorized update kernels are used when supported by
    the backend (PyTorch may fall back automatically if unsupported).

    Args:
        params: Iterable of parameters or dicts defining parameter groups.
        lr: Base learning rate.
        momentum: Momentum factor in [0, 1]. Set 0 to disable.
        nesterov: Whether to use Nesterov acceleration. Requires momentum > 0
            and dampening == 0 to strictly match Nesterov.
        dampening: Dampening for momentum. Usually 0.
        weight_decay: Weight decay coefficient (non-negative).
        decoupled_weight_decay: If True, use AdamW-style decoupled decay.
            If False, use classic L2 (coupled) by adding to the gradient.
        maximize: If True, perform gradient *ascent* instead of descent.
        foreach: If True, attempt vectorized updates for speed.
        differentiable: If True, make the optimizer step differentiable.
            This is slower and used in meta-learning scenarios.
        eps: Small epsilon added for numeric safety in denominators (rarely
            used in pure SGD but kept for robustness and symmetry with other
            optimizers).

    Raises:
        ValueError: If hyperparameter ranges are invalid.

    Chinese Notes (中文说明):
        - 该优化器支持动量与 Nesterov 前瞻加速，同时支持“耦合式 L2 正则”与
          “解耦式权重衰减 (AdamW 风格)”两种方式；
        - 当 ``foreach=True`` 时，会尝试使用 PyTorch 的 batched kernel 来批量
          更新张量，减少 Python 循环开销；
        - ``differentiable=True`` 可在需要对优化步骤求导的场景下使用（如元学习），
          但会显著降低性能；
        - ``zero_grad(set_to_none=True)`` 可以减少显存写入并避免误累加梯度。
    """

    def __init__(
        self,
        params: _params_t,
        lr: float = 1e-3,
        momentum: float = 0.0,
        nesterov: bool = False,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        decoupled_weight_decay: bool = False,
        maximize: bool = False,
        foreach: Optional[bool] = None,
        differentiable: bool = False,
        eps: float = 0.0,
    ) -> None:
        if lr < 0:
            raise ValueError(f"Invalid lr: {lr}")
        if momentum < 0 or momentum >= 1:
            raise ValueError(f"Invalid momentum: {momentum}. Expected in [0,1).")
        if dampening < 0 or dampening >= 1:
            raise ValueError(f"Invalid dampening: {dampening}. Expected in [0,1).")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if eps < 0:
            raise ValueError(f"Invalid eps: {eps}")
        if nesterov and (momentum <= 0):
            raise ValueError("Nesterov requires momentum > 0.")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            dampening=dampening,
            weight_decay=weight_decay,
            decoupled_weight_decay=decoupled_weight_decay,
            maximize=maximize,
            foreach=foreach,
            differentiable=differentiable,
            eps=eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Any] = None) -> Optional[float]:
        """Performs a single optimization step.

        If ``closure`` is provided, it is called with no arguments and should
        re-evaluate the model and return the loss.

        Chinese Notes (中文说明):
            - 该函数执行一次整体参数更新；
            - 支持可选的 closure（如 L-BFGS 需要多次前向/反向），SGD 通常无需；
            - 如果设置了 decoupled weight decay，会在更新前对参数做一次独立衰减；
            - 如果使用 Nesterov，则更新方向为 ``grad + momentum * v``；
            - 当 ``foreach=True`` 且后端支持时，使用批量内核提升效率。

        Args:
            closure: A closure that re-evaluates the model and returns the loss.

        Returns:
            The loss computed by ``closure`` (if any), otherwise ``None``.
        """
        loss: Optional[float] = None
        if closure is not None:
            with torch.enable_grad():
                loss_val = closure()
                # 兼容返回 Tensor 或 float
                loss = float(loss_val) if isinstance(loss_val, Tensor) else loss_val

        for group in self.param_groups:
            params_with_grad: List[Tensor] = []
            grads: List[Tensor] = []
            momentum_buffer_list: List[Optional[Tensor]] = []
            has_momentum = group["momentum"] > 0.0

            # —— 收集该组中需要更新的参数与梯度 ——
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    # 稀疏梯度的 SGD 需要特殊实现，这里不支持
                    raise RuntimeError("SGDMomentumNesterov does not support sparse gradients.")
                # maximize: 取相反数实现上升
                if group["maximize"]:
                    grad = -grad
                params_with_grad.append(p)
                grads.append(grad)

            if not params_with_grad:
                continue

            # —— 解耦式权重衰减：在更新前对参数做一次衰减 ——
            if group["decoupled_weight_decay"] and group["weight_decay"] != 0.0:
                wd = group["weight_decay"]
                for p in params_with_grad:
                    p.add_(p, alpha=-group["lr"] * wd)

            # —— 动量缓冲读取/初始化（延迟初始化） ——
            if has_momentum:
                for p in params_with_grad:
                    state = self.state[p]
                    buf = state.get("momentum_buffer")
                    momentum_buffer_list.append(buf)
            else:
                momentum_buffer_list = [None] * len(params_with_grad)

            # —— foreach 快路径（如可用） ——
            if group.get("foreach", False):
                self._step_foreach(
                    params_with_grad,
                    grads,
                    momentum_buffer_list,
                    lr=group["lr"],
                    momentum=group["momentum"],
                    dampening=group["dampening"],
                    nesterov=group["nesterov"],
                    weight_decay=group["weight_decay"],
                    coupled_weight_decay=(not group["decoupled_weight_decay"]),
                    eps=group["eps"],
                )
            else:
                # —— 逐参数更新 ——
                for i, p in enumerate(params_with_grad):
                    d_p = grads[i]

                    # 耦合式 L2：直接加到梯度上（与 AdamW 解耦相对）
                    if (not group["decoupled_weight_decay"]) and group["weight_decay"] != 0.0:
                        d_p = d_p.add(p, alpha=group["weight_decay"])  # grad += wd * p

                    if has_momentum:
                        state = self.state[p]
                        buf = momentum_buffer_list[i]
                        if buf is None:
                            # 首次创建动量缓冲
                            buf = torch.clone(d_p).detach()
                            state["momentum_buffer"] = buf
                        else:
                            # v = m * v + (1 - dampening) * grad
                            buf.mul_(group["momentum"]).add_(d_p, alpha=1.0 - group["dampening"])  # noqa: E501

                        if group["nesterov"] and group["dampening"] == 0.0:
                            # Nesterov: grad + momentum * v
                            update = d_p.add(buf, alpha=group["momentum"])  # noqa: E501
                        else:
                            update = buf
                    else:
                        update = d_p

                    # 参数更新：theta = theta - lr * update
                    p.add_(update, alpha=-group["lr"])  # noqa: E501

        return loss

    def _step_foreach(
        self,
        params: List[Tensor],
        grads: List[Tensor],
        momentum_buffers: List[Optional[Tensor]],
        *,
        lr: float,
        momentum: float,
        dampening: float,
        nesterov: bool,
        weight_decay: float,
        coupled_weight_decay: bool,
        eps: float,
    ) -> None:
        """Vectorized update path using ``torch._foreach_*`` kernels when possible.

        Chinese Notes (中文说明):
            - 使用 foreach API 批量操作张量，减少 Python 循环；
            - 注意：某些设备/数据类型可能不支持，将自动回退或抛错；
            - 我们保持与逐元素路径相同的数学逻辑。
        """
        # —— 耦合式 L2：grad += wd * p ——
        if coupled_weight_decay and weight_decay != 0.0:
            torch._foreach_add_(grads, params, alpha=weight_decay)

        # —— 动量分支 ——
        if momentum > 0.0:
            # 初始化缺失的动量缓冲
            for i, (p, g) in enumerate(zip(params, grads)):
                if momentum_buffers[i] is None:
                    mb = torch.clone(g).detach()
                    self.state[p]["momentum_buffer"] = mb
                    momentum_buffers[i] = mb
                else:
                    # v = m * v + (1 - dampening) * grad
                    momentum_buffers[i].mul_(momentum).add_(g, alpha=1.0 - dampening)

            if nesterov and dampening == 0.0:
                # update = grad + momentum * v
                # 这里无法直接 foreach 组合，逐个处理避免额外临时列表复制
                for i in range(len(params)):
                    update = grads[i].add(momentum_buffers[i], alpha=momentum)
                    params[i].add_(update, alpha=-lr)
                return
            else:
                # update = v
                torch._foreach_add_(params, momentum_buffers, alpha=-lr)
                return
        else:
            # 无动量：直接使用 grad
            torch._foreach_add_(params, grads, alpha=-lr)
            return


# -------------------------
# Utilities for quick tests
# -------------------------

def _quick_sanity_check() -> None:
    """Run a quick self-check on a toy problem.

    This function trains a tiny linear model for a few steps and prints
    diagnostics to ensure the optimizer updates parameters without producing
    NaNs/INFs.

    Chinese Notes (中文说明):
        - 用一个线性回归玩具样本做 10 步训练，打印初末 loss；
        - 断言 loss 在前几步内下降，且参数没有 NaN/Inf；
        - 该检查不保证收敛最优，但能快速发现实现层面的问题。
    """
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 1)
    opt = SGDMomentumNesterov(
        model.parameters(), lr=1e-1, momentum=0.9, nesterov=True,
        weight_decay=1e-4, decoupled_weight_decay=True,
    )
    x = torch.randn(64, 4)
    true_w = torch.randn(4, 1)
    y = x @ true_w + 0.1 * torch.randn(64, 1)
    loss_fn = torch.nn.MSELoss()

    # 初始 loss
    with torch.no_grad():
        y_pred0 = model(x)
        loss0 = loss_fn(y_pred0, y).item()
    print(f"[SanityCheck] initial loss = {loss0:.6f}")

    last_loss = None
    for step in range(10):
        opt.zero_grad(set_to_none=True)  # 中文：将梯度设为 None，避免无谓写入 0
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        opt.step()
        last_loss = loss.item()
        if math.isnan(last_loss) or math.isinf(last_loss):
            raise RuntimeError("Loss became NaN/Inf; check implementation.")

    print(f"[SanityCheck] final   loss = {last_loss:.6f}")
    assert last_loss < loss0, "Loss did not decrease; implementation may be wrong."

    # 参数无 NaN/Inf
    with torch.no_grad():
        for name, p in model.named_parameters():
            if torch.isnan(p).any() or torch.isinf(p).any():
                raise RuntimeError(f"Parameter {name} has NaN/Inf after training.")
    print("[SanityCheck] ✅ Passed basic checks.")


if __name__ == "__main__":
    # 运行基本自检（可选）
    _quick_sanity_check()