"""transformer_rope.py
====================================
A scalable Transformer Encoder implementation with Rotary Positional Embedding (RoPE).

This module is designed for industrial usage. 重点特性：
* 完整的 Transformer Encoder，支持多层堆叠与残差规范化。
* Rotary Positional Embeddings (RoPE) 集成于多头注意力，支持超长序列 extrapolation。
* 代码符合 Google Python Style Guide，Docstring 遵循 PEP 257，类型注解完整。
* 关键步骤附带中文注释，便于国内团队维护。
* 对输入形状、维度等边界情况进行健壮性检查。
* 高度模块化，便于二次开发与扩展。

Author: ChatGPT (o3)
Date: 2025‑08‑08
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Rotary Positional Embedding (RoPE)
# -----------------------------------------------------------------------------


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding implementation (可无限外推).

    Rotary Embedding 把位置编码转换为向量旋转，从而注入相对位置信息。
    """

    def __init__(self, dim: int, base: int = 10000) -> None:
        """Initialize rotary embedding.

        Args:
            dim: The embedding dimension for each head (必须是偶数).
            base: The base for frequency calculation. 10000 与论文一致。
        """
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("`dim` 必须为偶数，以便拆分为 [cos, sin] 对.")
        self.dim: int = dim
        self.base: int = base

        # 预计算角频率 (inv_freq) 以提高效率
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)  # 不随模型保存权重变化

    # --------------------------- 私有方法 ----------------------------------
    def _get_angles(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute sin/cos tables for given sequence length.

        Args:
            seq_len: 当前序列长度.
            device: 设备.
        Returns:
            A tuple (cos, sin) of shape (seq_len, dim).
        """
        # [seq_len, dim/2]
        positions = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
        angles = positions * self.inv_freq.to(device)  # 广播乘法
        cos: torch.Tensor = torch.cos(angles)
        sin: torch.Tensor = torch.sin(angles)
        # 重排为 [seq_len, dim]
        cos = torch.stack((cos, cos), dim=-1).reshape(seq_len, self.dim)
        sin = torch.stack((sin, sin), dim=-1).reshape(seq_len, self.dim)
        return cos, sin

    @staticmethod
    def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotary transformation to tensor x.

        Args:
            x: 待旋转张量 [..., seq_len, dim].
            cos: 余弦表 [seq_len, dim].
            sin: 正弦表 [seq_len, dim].
        Returns:
            旋转后的张量.
        """
        # 拆分为偶数索引与奇数索引两部分
        x1, x2 = x[..., ::2], x[..., 1::2]
        x_rotated_even = x1 * cos - x2 * sin
        x_rotated_odd = x1 * sin + x2 * cos
        # 将两部分重新交错合并
        x_out = torch.stack((x_rotated_even, x_rotated_odd), dim=-1).flatten(-2)
        return x_out

    # ---------------------- 公有接口 ---------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RoPE to input tensor.

        Args:
            x: Tensor with shape [batch, seq_len, dim].
        Returns:
            Tensor with rotary positional embedding applied.
        """
        batch_size, seq_len, dim = x.shape
        if dim != self.dim:
            raise ValueError(f"输入维度 {dim} 不匹配 RoPE 维度 {self.dim}.")
        cos, sin = self._get_angles(seq_len, x.device)
        cos = cos.unsqueeze(0).expand(batch_size, -1, -1)
        sin = sin.unsqueeze(0).expand(batch_size, -1, -1)
        return self._apply_rotary(x, cos, sin)

# -----------------------------------------------------------------------------
# Multi‑Head Attention with RoPE support
# -----------------------------------------------------------------------------


class MultiHeadSelfAttention(nn.Module):
    """Multi‑Head Self‑Attention layer with integrated RoPE.

    Attributes:
        num_heads: Number of attention heads.
        head_dim: Dimension of each head.
        dropout_p: Dropout probability for attention weights.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("`d_model` 必须可以整除 `num_heads`.")
        self.d_model: int = d_model
        self.num_heads: int = num_heads
        self.head_dim: int = d_model // num_heads
        self.scale: float = 1.0 / math.sqrt(self.head_dim)

        # Q, K, V projection
        self.qkv_proj: nn.Linear = nn.Linear(d_model, 3 * d_model)
        # Output projection
        self.out_proj: nn.Linear = nn.Linear(d_model, d_model)
        self.dropout: nn.Dropout = nn.Dropout(dropout_p)
        # Rotary embedding unit per head_dim
        self.rope = RotaryEmbedding(dim=self.head_dim)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split last dimension into (num_heads, head_dim)."""
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        # 转换为 [batch, num_heads, seq_len, head_dim]
        return x.permute(0, 2, 1, 3)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Merge (num_heads, head_dim) back to last dimension."""
        batch_size, num_heads, seq_len, head_dim = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch_size, seq_len, num_heads * head_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute multi‑head self‑attention with RoPE.

        Args:
            x: Input tensor of shape [batch, seq_len, d_model].
            mask: Optional tensor for attention mask. Shape [batch, seq_len], bool.
        Returns:
            Tensor after self‑attention, shape [batch, seq_len, d_model].
        """
        batch_size, seq_len, _ = x.shape
        # 线性投影 QKV
        qkv: torch.Tensor = self.qkv_proj(x)  # [batch, seq_len, 3*d_model]
        q, k, v = qkv.chunk(3, dim=-1)

        # 拆 heads
        q = self._split_heads(q)  # [batch, heads, seq_len, head_dim]
        k = self._split_heads(k)
        v = self._split_heads(v)

        # --- 应用 RoPE (仅作用于 Q, K) ---
        q = q.reshape(batch_size * self.num_heads, seq_len, self.head_dim)
        k = k.reshape(batch_size * self.num_heads, seq_len, self.head_dim)
        q = self.rope(q)
        k = self.rope(k)
        # [batch*heads, seq_len, head_dim] -> [batch, heads, seq_len, head_dim]
        q = q.view(batch_size, self.num_heads, seq_len, self.head_dim)
        k = k.view(batch_size, self.num_heads, seq_len, self.head_dim)

        # Attention score
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [batch, heads, seq_len, seq_len]

        # Mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            attn_scores = attn_scores.masked_fill(~mask, float("-inf"))

        attn_probs = F.softmax(attn_scores, dim=-1)  # [batch, heads, seq_len, seq_len]
        attn_probs = self.dropout(attn_probs)

        attn_output = torch.matmul(attn_probs, v)  # [batch, heads, seq_len, head_dim]
        attn_output = self._merge_heads(attn_output)  # [batch, seq_len, d_model]
        attn_output = self.out_proj(attn_output)  # [batch, seq_len, d_model]
        return attn_output

# -----------------------------------------------------------------------------
# Transformer Encoder Layer
# -----------------------------------------------------------------------------


class TransformerEncoderLayer(nn.Module):
    """A single Transformer encoder layer with RoPE self‑attention."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        dropout_p: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout_p)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_p)
        self.dropout2 = nn.Dropout(dropout_p)

        # 前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation_fn = getattr(F, activation)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for encoder layer."""
        # 自注意力子层
        attn_output = self.self_attn(src, mask=src_mask)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        # 前馈子层
        ff_output = self.linear2(self.activation_fn(self.linear1(src)))
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)
        return src

# -----------------------------------------------------------------------------
# Transformer Encoder (stack of layers)
# -----------------------------------------------------------------------------


class TransformerEncoder(nn.Module):
    """Transformer Encoder with RoPE, suitable for industrial scenarios."""

    def __init__(
        self,
        num_tokens: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        dropout_p: float = 0.1,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(num_tokens, d_model, padding_idx=pad_token_id)
        self.dropout = nn.Dropout(dropout_p)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout_p=dropout_p,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode input ids with RoPE‑enhanced Transformer.

        Args:
            input_ids: LongTensor [batch, seq_len].
            attention_mask: BoolTensor [batch, seq_len]. (True for keep, False for pad)
        Returns:
            Encoded representation [batch, seq_len, d_model].
        """
        embeddings = self.token_embedding(input_ids)  # [batch, seq_len, d_model]
        embeddings = self.dropout(embeddings)

        out = embeddings
        for i, layer in enumerate(self.layers):
            # 逐层调试信息打印
            print(f"Layer {i}: input shape {out.shape}")
            out = layer(out, src_mask=attention_mask)
        out = self.norm(out)
        return out

# -----------------------------------------------------------------------------
# Quick Sanity Check (run only when executing this file directly)
# -----------------------------------------------------------------------------


def _sanity_check() -> None:
    """Run a quick forward pass to verify model logic and shapes."""
    batch_size: int = 2
    seq_len: int = 16
    vocab_size: int = 100
    d_model: int = 64
    num_layers: int = 2
    num_heads: int = 8

    device = torch.device("cpu")

    # 随机生成假输入
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    attention_mask = (input_ids != 0)  # 假设 0 为 pad

    model = TransformerEncoder(
        num_tokens=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
    ).to(device)

    # 前向传播
    outputs = model(input_ids, attention_mask)
    print("=== Sanity Check ===")
    print("Output shape:", outputs.shape)  # 应为 [batch, seq_len, d_model]
    assert outputs.shape == (batch_size, seq_len, d_model)
    print("Sanity check passed! ✅")


if __name__ == "__main__":
    _sanity_check()
