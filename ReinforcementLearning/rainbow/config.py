# -*- coding: utf-8 -*-
"""Experiment configuration dataclasses for Rainbow DQN."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AgentExtensions:
    """Toggle switches for Rainbow extensions.

    Attributes:
        use_double_q: Enable Double Q-learning.
        use_per: Enable Prioritized Experience Replay.
        use_dueling: Enable Dueling network.
        use_n_step: Enable n-step returns.
        use_distributional: Enable distributional RL (C51).
        use_noisy_nets: Enable Noisy linear layers.
        per_priority_type: 'kl' or 'abs_td' for PER priority.
    """

    use_double_q: bool = True
    use_per: bool = True
    use_dueling: bool = True
    use_n_step: bool = True
    use_distributional: bool = True
    use_noisy_nets: bool = True
    per_priority_type: str = "kl"  # or "abs_td"

    def validate(self) -> None:
        """Validate combination constraints and values."""
        # 中文：当禁用 distributional 时，kl 作为优先级依据不成立，自动切换到 abs_td
        if not self.use_distributional and self.per_priority_type == "kl":
            self.per_priority_type = "abs_td"
        if self.per_priority_type not in {"kl", "abs_td"}:
            raise ValueError("per_priority_type must be 'kl' or 'abs_td'.")


@dataclass
class TrainConfig:
    """Training configuration for Rainbow DQN.

    Attributes:
        env_id: Gymnasium Atari environment ID.
        seed: Random seed.
        total_frames: Total environment frames to train.
        learning_starts: Number of frames before training starts.
        train_freq: Number of environment steps per optimization step.
        target_update_interval: Frequency (in steps) to update target net.
        eval_interval: Interval (in steps) to run evaluation episodes.
        eval_episodes: Number of episodes to evaluate.
        frame_stack: Number of stacked frames.
        gamma: Discount factor.
        n_step: N-step return length (if use_n_step is True).
        lr: Learning rate.
        adam_eps: Adam epsilon.
        batch_size: Batch size.
        buffer_size: Replay buffer capacity.
        per_alpha: PER alpha.
        per_beta0: Initial PER beta.
        per_beta_steps: Steps to anneal beta to 1.0.
        vmin: Min value support (C51).
        vmax: Max value support (C51).
        atoms: Number of atoms (C51).
        gradient_clip_norm: Max global grad norm (None to disable).
        max_noop: Max no-op actions at reset.
        clip_reward: Whether to clip reward to {-1,0,1}.
        log_dir: Directory for tensorboard/event logs and checkpoints.
        run_name: Human-readable run name.
        device: 'cuda' or 'cpu'.
        epsilon_start: If not using NoisyNets, starting epsilon.
        epsilon_final: Final epsilon.
        epsilon_decay_frames: Linear decay frames for epsilon.
        save_interval: Save checkpoint interval in steps.
    """

    env_id: str = "ALE/Breakout-v5"
    seed: int = 42
    total_frames: int = 5_000_000
    learning_starts: int = 80_000
    train_freq: int = 4
    target_update_interval: int = 10_000
    eval_interval: int = 100_000
    eval_episodes: int = 5
    frame_stack: int = 4

    gamma: float = 0.99
    n_step: int = 3

    lr: float = 1e-4
    adam_eps: float = 1.5e-4
    batch_size: int = 32
    buffer_size: int = 1_000_000

    per_alpha: float = 0.5
    per_beta0: float = 0.4
    per_beta_steps: int = 5_000_000

    vmin: float = -10.0
    vmax: float = 10.0
    atoms: int = 51

    gradient_clip_norm: Optional[float] = 10.0
    max_noop: int = 30
    clip_reward: bool = True

    log_dir: str = "runs"
    run_name: str = "rainbow_default"
    device: str = "cuda"

    epsilon_start: float = 1.0
    epsilon_final: float = 0.01
    epsilon_decay_frames: int = 1_000_000

    save_interval: int = 250_000

    def validate(self, ext: AgentExtensions) -> None:
        """Validate numeric ranges and relationships."""
        if self.total_frames <= 0:
            raise ValueError("total_frames must be > 0.")
        if self.learning_starts < 1_000:
            raise ValueError("learning_starts is too small; Atari commonly ≥ 80k.")
        if self.batch_size <= 0 or self.batch_size > 1024:
            raise ValueError("batch_size out of reasonable bounds.")
        if self.buffer_size < 100_000:
            raise ValueError("buffer_size too small for Atari scale.")
        if self.atoms <= 1 and ext.use_distributional:
            raise ValueError("atoms must be > 1 for distributional RL.")
        if self.vmax <= self.vmin and ext.use_distributional:
            raise ValueError("vmax must be > vmin for distributional RL.")
        if self.n_step <= 0:
            raise ValueError("n_step must be >= 1.")
        if self.per_alpha < 0 or self.per_alpha > 1:
            raise ValueError("per_alpha must be in [0,1].")
        if self.per_beta0 <= 0 or self.per_beta0 > 1:
            raise ValueError("per_beta0 must be in (0,1].")
        if self.per_beta_steps < 1:
            raise ValueError("per_beta_steps must be positive.")
        if self.epsilon_start < self.epsilon_final and not ext.use_noisy_nets:
            raise ValueError("epsilon_start must be >= epsilon_final.")