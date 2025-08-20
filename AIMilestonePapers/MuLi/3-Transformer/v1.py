#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Transformer Ablation Study on WMT14 with PyTorch.

This module provides an industrial-grade, end-to-end pipeline to run
ablation studies (as in "Attention is All You Need", Table 3) on the
Transformer architecture, focusing on:
  - Number of attention heads (h)
  - Model hidden dimension (d_model) and feed-forward dimension (d_ff)
  - Dropout rate
  - Positional encoding type (sinusoidal vs. learned)

Key features:
  * Robust engineering: boundary checks, clear class abstractions,
    Google style docstrings (PEP 257), PEP 8 compliance, and typing.
  * Training tricks: Adam + Noam warmup, label smoothing, tied embeddings.
  * Logging & viz: loguru for logs and tensorboardX for metrics.
  * Evaluation: SacreBLEU-based translation quality evaluation.
  * Data: HuggingFace datasets "wmt14" (if available) or local TSV files,
    SentencePiece BPE training or reuse for tokenization.
  * Sanity checks to validate shapes, masks and forward/backward logic.

Author: GPT-5 Thinking
Date: 2025-08-20
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Dataset

# Optional dependencies:
# datasets: for WMT14 loading. sentencepiece: for BPE tokenizer.
try:
    from datasets import load_dataset, DatasetDict  # type: ignore
    HF_DATASETS_AVAILABLE = True
except Exception:
    HF_DATASETS_AVAILABLE = False

try:
    import sentencepiece as spm  # type: ignore
    SENTENCEPIECE_AVAILABLE = True
except Exception:
    SENTENCEPIECE_AVAILABLE = False

try:
    import sacrebleu  # type: ignore
    SACREBLEU_AVAILABLE = True
except Exception:
    SACREBLEU_AVAILABLE = False

# -----------------------------
# Utilities and Configuration
# -----------------------------


def set_global_seed(seed: int) -> None:
    """Set global random seed for reproducibility.

    Args:
        seed: Random seed.
    """
    # 设置全局随机种子，保证实验可复现
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path, clear_if_exists: bool = False) -> Path:
    """Ensure a directory exists.

    Args:
        path: Directory path.
        clear_if_exists: Whether to clear the directory if exists.

    Returns:
        The directory path as Path.
    """
    # 确保输出目录存在；必要时清空
    p = Path(path)
    if p.exists() and clear_if_exists:
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass
class ExperimentConfig:
    """Experiment configuration.

    Attributes:
        run_name: Unique run name (used for logging/checkpoints).
        log_dir: TensorBoard & log directory.
        save_dir: Checkpoint directory.
        data_dir: Local data directory fallback (if HF datasets unavailable).
        spm_dir: SentencePiece model dir.
        src_lang: Source language code (e.g., 'en').
        tgt_lang: Target language code (e.g., 'de').
        vocab_size: Subword vocabulary size.
        max_len: Maximum sequence length (tokens).
        batch_size: Tokens per batch (approx; dynamic padding).
        num_epochs: Number of epochs.
        lr: Base learning rate (Noam scaled).
        warmup_steps: Warmup steps for learning rate.
        label_smoothing: Label smoothing epsilon [0, 1).
        dropout: Dropout rate [0, 1).
        d_model: Hidden size (must be divisible by n_heads).
        n_heads: Number of attention heads.
        n_layers: Number of encoder/decoder layers.
        d_ff: Feed-forward hidden size.
        pos_encoding: 'sinusoidal' or 'learned'.
        weight_tying: Weight tying between embedding and generator.
        grad_clip: Gradient clipping norm (0 to disable).
        fp16: Use automatic mixed precision if True.
        save_every: Save checkpoint every N steps.
        log_every: Log every N steps.
        eval_every: Evaluate every N steps.
        num_workers: DataLoader workers.
        seed: Random seed.
        device: 'cuda' or 'cpu'.
        max_train_samples: Optional cap on training samples (debug).
        max_eval_samples: Optional cap on eval samples (debug).
        group: Ablation group tag for aggregation.
    """
    run_name: str = "baseline_en-de"
    log_dir: str = "runs"
    save_dir: str = "checkpoints"
    data_dir: str = "data_wmt14"
    spm_dir: str = "spm_models"
    src_lang: str = "en"
    tgt_lang: str = "de"
    vocab_size: int = 37000
    max_len: int = 256
    batch_size: int = 8192  # 动态padding, 以 tokens 数近似
    num_epochs: int = 10
    lr: float = 2.0  # Noam lr factor
    warmup_steps: int = 4000
    label_smoothing: float = 0.1
    dropout: float = 0.1
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    pos_encoding: Literal["sinusoidal", "learned"] = "sinusoidal"
    weight_tying: bool = True
    grad_clip: float = 1.0
    fp16: bool = True
    save_every: int = 5000
    log_every: int = 100
    eval_every: int = 5000
    num_workers: int = 4
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    group: str = "ablation"


# -----------------------------
# Tokenization with SentencePiece
# -----------------------------


class SentencePieceTokenizer:
    """SentencePiece tokenizer wrapper supporting training and inference.

    This wrapper provides encode/decode methods and special token handling.

    Note:
        Requires 'sentencepiece' package installed.

    Attributes:
        model_prefix: Prefix (path) for the SentencePiece model files.
        vocab_size: Vocabulary size.
        character_coverage: Character coverage for training.
        model_type: BPE model type.
        pad_id: Padding token id.
        bos_id: Begin-of-sentence id.
        eos_id: End-of-sentence id.
        unk_id: Unknown token id.
    """

    def __init__(
        self,
        model_prefix: str,
        vocab_size: int = 37000,
        character_coverage: float = 1.0,
        model_type: str = "bpe",
    ) -> None:
        if not SENTENCEPIECE_AVAILABLE:
            raise RuntimeError("sentencepiece is required but not installed.")
        self.model_prefix = model_prefix
        self.vocab_size = vocab_size
        self.character_coverage = character_coverage
        self.model_type = model_type
        self._sp = spm.SentencePieceProcessor()  # type: ignore
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 3

    def train(
        self,
        input_files: List[str],
        user_defined_symbols: Optional[List[str]] = None,
    ) -> None:
        """Train a SentencePiece model.

        Args:
            input_files: List of input text files for training.
            user_defined_symbols: Optional extra symbols.
        """
        # 训练 SentencePiece 模型，生成 .model/.vocab
        args = {
            "input": ",".join(input_files),
            "model_prefix": self.model_prefix,
            "vocab_size": self.vocab_size,
            "character_coverage": self.character_coverage,
            "model_type": self.model_type,
            "pad_id": self.pad_id,
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
            "unk_id": self.unk_id,
        }
        if user_defined_symbols:
            args["user_defined_symbols"] = ",".join(user_defined_symbols)
        spm.SentencePieceTrainer.Train(**args)  # type: ignore

    def load(self) -> None:
        """Load an existing SentencePiece model."""
        # 加载已训练的 SPM 模型
        model_file = f"{self.model_prefix}.model"
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"SentencePiece model not found: {model_file}")
        self._sp.Load(model_file)  # type: ignore

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        """Encode text to a list of token ids.

        Args:
            text: Input string.
            add_bos: Prepend BOS token if True.
            add_eos: Append EOS token if True.

        Returns:
            List of token ids.
        """
        # 编码文本为 token id 列表
        ids = self._sp.EncodeAsIds(text)  # type: ignore
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode token ids to text.

        Args:
            ids: Token id list.

        Returns:
            Decoded string.
        """
        # 将 token id 列表解码为字符串
        return self._sp.DecodeIds(ids)  # type: ignore

    @property
    def vocab_size_(self) -> int:
        """Return vocabulary size including special tokens."""
        return self._sp.GetPieceSize()  # type: ignore


# -----------------------------
# Dataset and DataModule
# -----------------------------


class TranslationExample(Dataset):
    """PyTorch dataset for tokenized parallel examples."""

    def __init__(
        self,
        src: List[List[int]],
        tgt: List[List[int]],
        max_len: int,
    ) -> None:
        # 存储已分词的源/目标序列
        self.src = src
        self.tgt = tgt
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.src)

    def __getitem__(self, index: int) -> Tuple[List[int], List[int]]:
        # 返回单个样本的 (src_ids, tgt_ids)
        s = self.src[index][: self.max_len]
        t = self.tgt[index][: self.max_len]
        return s, t


class DynamicBatchCollator:
    """Dynamic padding and mask creation for Transformer training."""

    def __init__(self, pad_id: int, device: torch.device) -> None:
        # collate 函数：根据 batch 序列最长长度进行动态 padding
        self.pad_id = pad_id
        self.device = device

    def _pad(self, sequences: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        # 对不等长序列进行 padding，并返回 mask
        max_len = max(len(s) for s in sequences)
        batch = len(sequences)
        tensor = torch.full((batch, max_len), fill_value=self.pad_id, dtype=torch.long)
        for i, seq in enumerate(sequences):
            tensor[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        # padding mask: True indicates padding positions
        pad_mask = (tensor == self.pad_id)
        return tensor, pad_mask

    def _subsequent_mask(self, size: int) -> torch.Tensor:
        # 下三角 mask，防止 decoder 看到未来位置
        attn_shape = (1, size, size)
        subsequent_mask = torch.triu(torch.ones(attn_shape, dtype=torch.bool), diagonal=1)
        return subsequent_mask  # shape: (1, T, T)

    def __call__(
        self, batch: List[Tuple[List[int], List[int]]]
    ) -> Dict[str, torch.Tensor]:
        # 组装 batch，并生成模型所需的 mask
        src_list, tgt_list = zip(*batch)
        src, src_pad_mask = self._pad(list(src_list))
        tgt, tgt_pad_mask = self._pad(list(tgt_list))

        # 训练时将 tgt 分成输入 (tgt_in) 与 label (tgt_out)
        # 例如: [BOS, y1, y2, ..., y_{n-1}] -> 输入
        #       [y1, y2, ..., y_{n-1}, EOS] -> 标签
        tgt_in = tgt[:, :-1].contiguous()
        tgt_out = tgt[:, 1:].contiguous()

        # subsequent mask 与 pad mask 组合
        seq_len = tgt_in.size(1)
        subsequent = self._subsequent_mask(seq_len).to(self.device)
        # 注意：Transformer 里的 key_padding_mask 语义为 True 表示 padding
        # 对 decoder self-attention 的 mask 需要同时考虑 pad 和 subsequent
        src_key_padding_mask = src_pad_mask.to(self.device)  # (B, S)
        tgt_key_padding_mask = tgt_pad_mask[:, :-1].to(self.device)  # (B, T)
        tgt_mask = subsequent  # (1, T, T)

        return {
            "src": src.to(self.device),
            "tgt_in": tgt_in.to(self.device),
            "tgt_out": tgt_out.to(self.device),
            "src_key_padding_mask": src_key_padding_mask,
            "tgt_key_padding_mask": tgt_key_padding_mask,
            "tgt_mask": tgt_mask,
        }


class WMT14DataModule:
    """Data module to orchestrate loading, tokenization, and dataloaders.

    This supports HF datasets (if available) or local TSV files with columns:
    src \t tgt

    Attributes:
        cfg: Experiment configuration.
        sp_src: SentencePiece tokenizer for source language.
        sp_tgt: SentencePiece tokenizer for target language.
    """

    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg
        self.sp_src: Optional[SentencePieceTokenizer] = None
        self.sp_tgt: Optional[SentencePieceTokenizer] = None
        self.ds_train: Optional[TranslationExample] = None
        self.ds_valid: Optional[TranslationExample] = None
        self.ds_test: Optional[TranslationExample] = None

    def _maybe_train_or_load_spm(
        self,
        src_corpus_files: List[str],
        tgt_corpus_files: List[str],
    ) -> None:
        # 训练或加载 SentencePiece 模型（若不存在）
        ensure_dir(self.cfg.spm_dir)
        src_prefix = str(Path(self.cfg.spm_dir) / f"spm_{self.cfg.src_lang}")
        tgt_prefix = str(Path(self.cfg.spm_dir) / f"spm_{self.cfg.tgt_lang}")

        self.sp_src = SentencePieceTokenizer(
            src_prefix, vocab_size=self.cfg.vocab_size
        )
        self.sp_tgt = SentencePieceTokenizer(
            tgt_prefix, vocab_size=self.cfg.vocab_size
        )

        # 若模型已存在则直接加载，否则训练
        if not os.path.exists(f"{src_prefix}.model") or not os.path.exists(
            f"{tgt_prefix}.model"
        ):
            logger.info("Training SentencePiece models for src/tgt ...")
            self.sp_src.train(src_corpus_files)
            self.sp_tgt.train(tgt_corpus_files)
        else:
            logger.info("Loading existing SentencePiece models for src/tgt ...")

        self.sp_src.load()
        self.sp_tgt.load()

    def _load_hf_wmt14(self) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
        """Load WMT14 via HuggingFace datasets if available.

        Returns:
            (train_src, train_tgt, valid_src, valid_tgt, test_src, test_tgt)
        """
        # 使用 HF datasets 加载 WMT14（若可用）
        # 为了兼容性，这里尝试几种配置名称
        configs_try = [
            (f"{self.cfg.src_lang}-{self.cfg.tgt_lang}", self.cfg.src_lang, self.cfg.tgt_lang),
            (f"{self.cfg.tgt_lang}-{self.cfg.src_lang}", self.cfg.tgt_lang, self.cfg.src_lang),
        ]
        last_exc = None
        for config_name, src, tgt in configs_try:
            try:
                dsd: DatasetDict = load_dataset("wmt14", config_name)  # type: ignore
                # datasets 字段可能为 'translation' 或 'translation' 键下字典
                def _extract_pairs(split_name: str) -> Tuple[List[str], List[str]]:
                    split = dsd[split_name]
                    xs, ys = [], []
                    for ex in split:
                        # HF wmt14 通常提供 ex['translation'] = {src_lang: str, tgt_lang: str}
                        titem = ex["translation"]
                        xs.append(titem[src])
                        ys.append(titem[tgt])
                    return xs, ys

                train_x, train_y = _extract_pairs("train")
                valid_x, valid_y = _extract_pairs("validation") if "validation" in dsd else _extract_pairs("test")
                test_x, test_y = _extract_pairs("test")

                # 若配置顺序与期望相反，则交换
                if src != self.cfg.src_lang:
                    # 交换回我们的定义
                    train_x, train_y = train_y, train_x
                    valid_x, valid_y = valid_y, valid_x
                    test_x, test_y = test_y, test_x

                return train_x, train_y, valid_x, valid_y, test_x, test_y
            except Exception as e:  # noqa: BLE001
                last_exc = e
                continue
        raise RuntimeError(
            f"Failed to load WMT14 via datasets. Last error: {last_exc}"
        )

    def _load_local_parallel(
        self,
        split: str,
    ) -> Tuple[List[str], List[str]]:
        """Load local TSV parallel data for a split.

        Args:
            split: One of 'train', 'valid', 'test'.

        Returns:
            (src_texts, tgt_texts)
        """
        # 从本地 TSV 读取平行语料，格式: src \t tgt
        file_path = Path(self.cfg.data_dir) / f"{split}.{self.cfg.src_lang}-{self.cfg.tgt_lang}.tsv"
        if not file_path.exists():
            raise FileNotFoundError(f"Local data file not found: {file_path}")
        xs, ys = [], []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 2:
                    # 跳过非法行
                    continue
                xs.append(parts[0])
                ys.append(parts[1])
        return xs, ys

    def prepare(self) -> None:
        """Prepare tokenizers and datasets."""
        # 准备数据：加载文本、（训练或加载）SPM、编码、构造Dataset
        logger.info("Preparing data module ...")

        if HF_DATASETS_AVAILABLE:
            try:
                train_x, train_y, valid_x, valid_y, test_x, test_y = self._load_hf_wmt14()
                logger.info("Loaded WMT14 from HuggingFace datasets.")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"HF datasets loading failed: {e}. Falling back to local files.")
                train_x, train_y = self._load_local_parallel("train")
                valid_x, valid_y = self._load_local_parallel("valid")
                test_x, test_y = self._load_local_parallel("test")
        else:
            logger.warning("datasets not available. Using local files.")
            train_x, train_y = self._load_local_parallel("train")
            valid_x, valid_y = self._load_local_parallel("valid")
            test_x, test_y = self._load_local_parallel("test")

        # 限制样本数（调试/快速运行）
        if self.cfg.max_train_samples:
            train_x, train_y = train_x[: self.cfg.max_train_samples], train_y[: self.cfg.max_train_samples]
        if self.cfg.max_eval_samples:
            valid_x, valid_y = valid_x[: self.cfg.max_eval_samples], valid_y[: self.cfg.max_eval_samples]
            test_x, test_y = test_x[: self.cfg.max_eval_samples], test_y[: self.cfg.max_eval_samples]

        # 训练/加载 SentencePiece
        if not SENTENCEPIECE_AVAILABLE:
            raise RuntimeError("sentencepiece is required for BPE tokenization.")
        ensure_dir(self.cfg.spm_dir)
        # 为了更稳妥，SPM 用完整训练集训练
        src_tmp = str(Path(self.cfg.spm_dir) / f"tmp_src_{self.cfg.src_lang}.txt")
        tgt_tmp = str(Path(self.cfg.spm_dir) / f"tmp_tgt_{self.cfg.tgt_lang}.txt")
        with open(src_tmp, "w", encoding="utf-8") as f:
            for s in train_x:
                f.write(s.strip() + "\n")
        with open(tgt_tmp, "w", encoding="utf-8") as f:
            for s in train_y:
                f.write(s.strip() + "\n")
        self._maybe_train_or_load_spm([src_tmp], [tgt_tmp])
        # 清理临时文件
        try:
            os.remove(src_tmp)
            os.remove(tgt_tmp)
        except Exception:
            pass

        assert self.sp_src is not None and self.sp_tgt is not None

        # 编码文本为 token 序列
        def _encode_corpus(xs: List[str], ys: List[str]) -> Tuple[List[List[int]], List[List[int]]]:
            src_tok = [self.sp_src.encode(x) for x in xs]
            tgt_tok = [self.sp_tgt.encode(y) for y in ys]
            return src_tok, tgt_tok

        train_src_tok, train_tgt_tok = _encode_corpus(train_x, train_y)
        valid_src_tok, valid_tgt_tok = _encode_corpus(valid_x, valid_y)
        test_src_tok, test_tgt_tok = _encode_corpus(test_x, test_y)

        # 构造 PyTorch Dataset
        self.ds_train = TranslationExample(train_src_tok, train_tgt_tok, self.cfg.max_len)
        self.ds_valid = TranslationExample(valid_src_tok, valid_tgt_tok, self.cfg.max_len)
        self.ds_test = TranslationExample(test_src_tok, test_tgt_tok, self.cfg.max_len)

        logger.info(
            f"Prepared datasets: train={len(self.ds_train)} valid={len(self.ds_valid)} test={len(self.ds_test)}"
        )

    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create dataloaders for train/valid/test.

        Returns:
            Tuple of train_loader, valid_loader, test_loader.
        """
        assert self.ds_train and self.ds_valid and self.ds_test
        device = torch.device(self.cfg.device)
        collate = DynamicBatchCollator(pad_id=0, device=device)

        # 注意：这里 batch_size 是句子数；由于我们使用动态 padding，
        # 若想更近似“按token数”控制，可改造为动态采样器（此处采用固定batch简化）。
        train_loader = DataLoader(
            self.ds_train,
            batch_size=max(1, self.cfg.batch_size // 128),  # 经验估计，避免OOM
            shuffle=True,
            num_workers=self.cfg.num_workers,
            collate_fn=collate,
            pin_memory=True,
            drop_last=True,
        )
        valid_loader = DataLoader(
            self.ds_valid,
            batch_size=max(1, self.cfg.batch_size // 256),
            shuffle=False,
            num_workers=self.cfg.num_workers,
            collate_fn=collate,
            pin_memory=True,
            drop_last=False,
        )
        test_loader = DataLoader(
            self.ds_test,
            batch_size=max(1, self.cfg.batch_size // 256),
            shuffle=False,
            num_workers=self.cfg.num_workers,
            collate_fn=collate,
            pin_memory=True,
            drop_last=False,
        )
        return train_loader, valid_loader, test_loader


# -----------------------------
# Model Components
# -----------------------------


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding as in Vaswani et al. 2017."""

    def __init__(self, d_model: int, max_len: int = 10000) -> None:
        super().__init__()
        # 预计算正弦位置编码矩阵，避免重复计算
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 注册为 buffer，不参与训练
        self.register_buffer("pe", pe)  # shape: (max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encodings to input.

        Args:
            x: Input tensor with shape (B, T, D)

        Returns:
            Tensor with positional encodings added (B, T, D).
        """
        # 将预计算的位置编码加到输入，保持 shape 不变
        bsz, seq_len, d_model = x.size()
        x = x + self.pe[:seq_len].unsqueeze(0)
        return x


class LearnedPositionalEncoding(nn.Embedding):
    """Learned positional embedding."""

    def __init__(self, max_len: int, d_model: int) -> None:
        super().__init__(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add learned positional embeddings.

        Args:
            x: Input tensor with shape (B, T, D)

        Returns:
            Tensor with positional embeddings added (B, T, D).
        """
        bsz, seq_len, d_model = x.size()
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).repeat(bsz, 1)
        pos_embed = super().forward(positions)  # (B, T, D)
        return x + pos_embed


class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss for sequence-to-sequence tasks."""

    def __init__(self, vocab_size: int, smoothing: float, ignore_index: int = 0) -> None:
        super().__init__()
        if not (0.0 <= smoothing < 1.0):
            raise ValueError("label_smoothing must be in [0,1).")
        # Label Smoothing: 将 one-hot 标签分配少量概率给其他类
        self.ignore_index = ignore_index
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.vocab_size = vocab_size

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute smoothed cross-entropy.

        Args:
            pred: Logits of shape (B*T, V).
            target: Target token ids of shape (B*T,).

        Returns:
            Loss scalar tensor.
        """
        # 忽略 padding 的位置
        mask = (target != self.ignore_index)
        target = target[mask]
        pred = pred[mask]
        if pred.numel() == 0:
            return pred.new_tensor(0.0)

        # 构造平滑分布
        with torch.no_grad():
            true_dist = torch.full_like(pred, fill_value=self.smoothing / (self.vocab_size - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        loss = F.kl_div(F.log_softmax(pred, dim=-1), true_dist, reduction="batchmean")
        return loss


class NoamScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Noam learning rate scheduler from Vaswani et al. (2017)."""

    def __init__(self, optimizer: torch.optim.Optimizer, d_model: int, warmup_steps: int, last_epoch: int = -1) -> None:
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        # 根据 step 动态计算学习率，warmup 后按 1/sqrt(step) 衰减
        step = max(1, self.last_epoch + 1)
        scale = (self.d_model ** -0.5) * min(step ** -0.5, step * (self.warmup_steps ** -1.5))
        return [base_lr * scale for base_lr in self.base_lrs]


class TransformerSeq2Seq(nn.Module):
    """Transformer Seq2Seq model wrapping nn.Transformer with custom embeddings/PE.

    Supports:
      - Sinusoidal or learned positional encodings
      - Weight tying (encoder/decoder emb <-> generator)
    """

    def __init__(
        self,
        vocab_src: int,
        vocab_tgt: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
        max_len: int,
        pos_encoding: Literal["sinusoidal", "learned"] = "sinusoidal",
        weight_tying: bool = True,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0,1).")

        # Embeddings
        self.src_embed = nn.Embedding(vocab_src, d_model, padding_idx=0)
        self.tgt_embed = nn.Embedding(vocab_tgt, d_model, padding_idx=0)

        # Positional Encoding
        if pos_encoding == "sinusoidal":
            self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        elif pos_encoding == "learned":
            self.pos_enc = LearnedPositionalEncoding(max_len=max_len, d_model=d_model)
        else:
            raise ValueError("pos_encoding must be 'sinusoidal' or 'learned'.")

        # Core Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=n_heads,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,  # 使用 (B, T, D) 接口
            norm_first=False,  # 与原论文后归一化一致
        )

        # Generator (pre-softmax linear)
        self.generator = nn.Linear(d_model, vocab_tgt, bias=False)

        # Weight tying
        if weight_tying:
            if self.tgt_embed.weight.size() != self.generator.weight.size():
                raise ValueError("Weight tying requires tgt_embed and generator to have same shape.")
            self.generator.weight = self.tgt_embed.weight  # weight tying

        # Initialization
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # 参数初始化（Xavier）
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src_ids: torch.Tensor, src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        """Encode source sequence.

        Args:
            src_ids: Source token ids (B, S).
            src_key_padding_mask: Padding mask (B, S), True for PAD positions.

        Returns:
            Encoder memory tensor (B, S, D).
        """
        src_emb = self.src_embed(src_ids) * math.sqrt(self.transformer.d_model)
        src_emb = self.pos_enc(src_emb)
        memory = self.transformer.encoder(
            src_emb,
            src_key_padding_mask=src_key_padding_mask,  # True 表示 padding
        )
        return memory

    def decode(
        self,
        tgt_in: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Decode target sequence given encoder memory.

        Args:
            tgt_in: Target input ids (B, T).
            memory: Encoder memory (B, S, D).
            tgt_mask: Causal mask for decoder self-attention (T, T) or (1, T, T).
            tgt_key_padding_mask: Padding mask for target (B, T).
            memory_key_padding_mask: Padding mask for source (B, S).

        Returns:
            Decoder output logits (B, T, V).
        """
        tgt_emb = self.tgt_embed(tgt_in) * math.sqrt(self.transformer.d_model)
        tgt_emb = self.pos_enc(tgt_emb)
        out = self.transformer.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask[0],  # (T, T)
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        logits = self.generator(out)  # (B, T, V)
        return logits

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_in: torch.Tensor,
        tgt_mask: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for training.

        Args:
            src_ids: Source token ids (B, S).
            tgt_in: Target input ids (B, T).
            tgt_mask: Causal mask (1, T, T).
            src_key_padding_mask: Source padding mask (B, S).
            tgt_key_padding_mask: Target padding mask (B, T).

        Returns:
            Logits (B, T, V).
        """
        memory = self.encode(src_ids, src_key_padding_mask)
        logits = self.decode(
            tgt_in=tgt_in,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return logits

    @torch.no_grad()
    def greedy_decode(
        self,
        src_ids: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
        max_len: int,
        bos_id: int,
        eos_id: int,
    ) -> torch.Tensor:
        """Greedy decoding for inference.

        Args:
            src_ids: Source ids (B, S).
            src_key_padding_mask: Source padding mask (B, S).
            max_len: Max generation length.
            bos_id: BOS token id.
            eos_id: EOS token id.

        Returns:
            Generated ids (B, T_out).
        """
        # 贪心解码，逐步生成，直到 EOS 或达到最大长度
        device = src_ids.device
        memory = self.encode(src_ids, src_key_padding_mask)
        B = src_ids.size(0)
        ys = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            T = ys.size(1)
            subsequent_mask = torch.triu(
                torch.ones((T, T), device=device, dtype=torch.bool), diagonal=1
            ).unsqueeze(0)  # (1, T, T)
            logits = self.decode(
                tgt_in=ys,
                memory=memory,
                tgt_mask=subsequent_mask,
                tgt_key_padding_mask=torch.zeros((B, T), dtype=torch.bool, device=device),
                memory_key_padding_mask=src_key_padding_mask,
            )
            next_token = torch.argmax(logits[:, -1, :], dim=-1)  # (B,)
            ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
            finished = finished | (next_token == eos_id)
            if torch.all(finished):
                break
        return ys


# -----------------------------
# Trainer
# -----------------------------


class Trainer:
    """Trainer handling training loop, evaluation, and logging."""

    def __init__(self, cfg: ExperimentConfig, data: WMT14DataModule) -> None:
        self.cfg = cfg
        self.data = data
        self.device = torch.device(cfg.device)
        self.writer: Optional[SummaryWriter] = None
        self.global_step = 0

        # 初始化日志与目录
        ensure_dir(cfg.log_dir)
        ensure_dir(cfg.save_dir)
        self.run_dir = ensure_dir(Path(cfg.log_dir) / cfg.group / cfg.run_name, clear_if_exists=False)
        self.ckpt_dir = ensure_dir(Path(cfg.save_dir) / cfg.group / cfg.run_name, clear_if_exists=False)

    def _build_model_and_optim(self) -> Tuple[TransformerSeq2Seq, torch.optim.Optimizer, NoamScheduler]:
        assert self.data.sp_src is not None and self.data.sp_tgt is not None
        vocab_src = self.data.sp_src.vocab_size_
        vocab_tgt = self.data.sp_tgt.vocab_size_
        logger.info(f"Vocab sizes -> src: {vocab_src} tgt: {vocab_tgt}")

        model = TransformerSeq2Seq(
            vocab_src=vocab_src,
            vocab_tgt=vocab_tgt,
            d_model=self.cfg.d_model,
            n_heads=self.cfg.n_heads,
            n_layers=self.cfg.n_layers,
            d_ff=self.cfg.d_ff,
            dropout=self.cfg.dropout,
            max_len=self.cfg.max_len,
            pos_encoding=self.cfg.pos_encoding,
            weight_tying=self.cfg.weight_tying,
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.lr, betas=(0.9, 0.98), eps=1e-9)
        scheduler = NoamScheduler(optimizer, d_model=self.cfg.d_model, warmup_steps=self.cfg.warmup_steps)

        return model, optimizer, scheduler

    def _evaluate_bleu(self, model: TransformerSeq2Seq, loader: DataLoader) -> float:
        # 使用 SacreBLEU 对验证集/测试集进行评估
        if not SACREBLEU_AVAILABLE:
            logger.warning("sacrebleu not installed; returning 0.0 as BLEU.")
            return 0.0
        assert self.data.sp_tgt is not None and self.data.sp_src is not None
        refs: List[str] = []
        hyps: List[str] = []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                src = batch["src"]
                src_kpm = batch["src_key_padding_mask"]
                # 贪心解码（可扩展 beam search）
                gen = model.greedy_decode(
                    src_ids=src,
                    src_key_padding_mask=src_kpm,
                    max_len=self.cfg.max_len,
                    bos_id=self.data.sp_tgt.bos_id,
                    eos_id=self.data.sp_tgt.eos_id,
                )
                # 取 batch 中的参考/假设句子（去掉BOS）
                tgt_out = batch["tgt_out"].cpu().tolist()
                gen_out = gen.cpu().tolist()
                for ref_ids, hyp_ids in zip(tgt_out, gen_out):
                    # 去除 padding 与 超出 eos 的部分
                    ref_ids_clean = [tid for tid in ref_ids if tid not in (0,)]
                    hyp_ids_clean = [tid for tid in hyp_ids if tid not in (0,)]
                    # 去掉起始 BOS
                    if len(hyp_ids_clean) > 0 and hyp_ids_clean[0] == self.data.sp_tgt.bos_id:
                        hyp_ids_clean = hyp_ids_clean[1:]
                    # 截断至 EOS
                    if self.data.sp_tgt.eos_id in hyp_ids_clean:
                        idx = hyp_ids_clean.index(self.data.sp_tgt.eos_id)
                        hyp_ids_clean = hyp_ids_clean[: idx + 1]
                    ref_text = self.data.sp_tgt.decode(ref_ids_clean)
                    hyp_text = self.data.sp_tgt.decode(hyp_ids_clean)
                    refs.append(ref_text)
                    hyps.append(hyp_text)

        bleu = sacrebleu.corpus_bleu(hyps, [refs]).score  # type: ignore
        return float(bleu)

    def train(self) -> Dict[str, Any]:
        """Run full training with periodic evaluation and checkpointing."""
        train_loader, valid_loader, test_loader = self.data.create_dataloaders()
        model, optimizer, scheduler = self._build_model_and_optim()
        criterion = LabelSmoothingLoss(
            vocab_size=self.data.sp_tgt.vocab_size_,
            smoothing=self.cfg.label_smoothing,
            ignore_index=0,
        )

        self.writer = SummaryWriter(logdir=str(self.run_dir))

        # 混合精度
        scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.fp16)

        logger.info(f"Start training: epochs={self.cfg.num_epochs}, device={self.cfg.device}")
        best_valid_bleu = -1.0
        best_ckpt_path = None
        start_time = time.time()

        for epoch in range(1, self.cfg.num_epochs + 1):
            model.train()
            epoch_loss = 0.0
            step_in_epoch = 0

            for batch in train_loader:
                self.global_step += 1
                step_in_epoch += 1

                src = batch["src"]
                tgt_in = batch["tgt_in"]
                tgt_out = batch["tgt_out"]
                src_kpm = batch["src_key_padding_mask"]
                tgt_kpm = batch["tgt_key_padding_mask"]
                tgt_mask = batch["tgt_mask"]

                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=self.cfg.fp16):
                    logits = model(
                        src_ids=src,
                        tgt_in=tgt_in,
                        tgt_mask=tgt_mask,
                        src_key_padding_mask=src_kpm,
                        tgt_key_padding_mask=tgt_kpm,
                    )
                    # 展平计算交叉熵/标签平滑
                    B, T, V = logits.shape
                    loss = criterion(
                        logits.reshape(B * T, V),
                        tgt_out.reshape(B * T),
                    )

                scaler.scale(loss).backward()
                if self.cfg.grad_clip and self.cfg.grad_clip > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                epoch_loss += loss.item()

                if self.global_step % self.cfg.log_every == 0:
                    cur_lr = scheduler.get_last_lr()[0]
                    avg_loss = epoch_loss / step_in_epoch
                    logger.info(
                        f"step={self.global_step} epoch={epoch} step_in_epoch={step_in_epoch} "
                        f"loss={loss.item():.4f} avg_loss={avg_loss:.4f} lr={cur_lr:.6e}"
                    )
                    assert self.writer is not None
                    self.writer.add_scalar("train/loss", loss.item(), self.global_step)
                    self.writer.add_scalar("train/lr", cur_lr, self.global_step)

                if self.global_step % self.cfg.eval_every == 0:
                    valid_bleu = self._evaluate_bleu(model, valid_loader)
                    logger.info(f"[Eval] step={self.global_step} valid_BLEU={valid_bleu:.2f}")
                    assert self.writer is not None
                    self.writer.add_scalar("valid/BLEU", valid_bleu, self.global_step)

                    # Save best
                    if valid_bleu > best_valid_bleu:
                        best_valid_bleu = valid_bleu
                        best_ckpt_path = Path(self.ckpt_dir) / f"best_step{self.global_step}.pt"
                        torch.save(
                            {
                                "model": model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "scheduler": scheduler.state_dict(),
                                "config": dataclasses.asdict(self.cfg),
                                "global_step": self.global_step,
                            },
                            str(best_ckpt_path),
                        )
                        logger.info(f"Saved best checkpoint to {best_ckpt_path}")

                if self.global_step % self.cfg.save_every == 0:
                    ckpt_path = Path(self.ckpt_dir) / f"ckpt_step{self.global_step}.pt"
                    torch.save(
                        {"model": model.state_dict(), "global_step": self.global_step},
                        str(ckpt_path),
                    )
                    logger.info(f"Saved checkpoint to {ckpt_path}")

            # 每个 epoch 结束，做一次评估
            valid_bleu = self._evaluate_bleu(model, valid_loader)
            logger.info(f"[EpochEnd] epoch={epoch} valid_BLEU={valid_bleu:.2f}")
            assert self.writer is not None
            self.writer.add_scalar("valid/BLEU_epoch", valid_bleu, epoch)

            if valid_bleu > best_valid_bleu:
                best_valid_bleu = valid_bleu
                best_ckpt_path = Path(self.ckpt_dir) / f"best_epoch{epoch}.pt"
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "config": dataclasses.asdict(self.cfg),
                        "global_step": self.global_step,
                    },
                    str(best_ckpt_path),
                )
                logger.info(f"Saved best checkpoint to {best_ckpt_path}")

        total_time = time.time() - start_time
        logger.info(f"Training finished in {total_time/3600:.2f}h. Best valid BLEU={best_valid_bleu:.2f}")

        # 测试集评估（使用最佳模型）
        test_bleu = 0.0
        if best_ckpt_path and os.path.exists(best_ckpt_path):
            logger.info(f"Loading best checkpoint {best_ckpt_path} for test evaluation ...")
            ckpt = torch.load(best_ckpt_path, map_location=self.device)
            model.load_state_dict(ckpt["model"])
            test_bleu = self._evaluate_bleu(model, test_loader)
            logger.info(f"[Test] BLEU={test_bleu:.2f}")
            assert self.writer is not None
            self.writer.add_scalar("test/BLEU", test_bleu, self.global_step)

        # 关闭 writer
        if self.writer is not None:
            self.writer.close()

        return {"best_valid_bleu": best_valid_bleu, "test_bleu": test_bleu}


# -----------------------------
# Ablation Orchestrator
# -----------------------------


class AblationRunner:
    """Run a set of ablation experiments and collate results."""

    def __init__(self, base_cfg: ExperimentConfig) -> None:
        self.base_cfg = base_cfg
        self.results: List[Dict[str, Any]] = []

    def _run_single(self, cfg: ExperimentConfig) -> Dict[str, Any]:
        # 运行单次实验（包含训练与评估），并记录结果
        set_global_seed(cfg.seed)
        logger.info(f"Running experiment: {cfg.run_name} | group={cfg.group}")
        logger.info(json.dumps(dataclasses.asdict(cfg), indent=2, ensure_ascii=False))

        data_mod = WMT14DataModule(cfg)
        data_mod.prepare()

        trainer = Trainer(cfg, data_mod)
        result = trainer.train()
        result_row = {
            "run_name": cfg.run_name,
            "group": cfg.group,
            "d_model": cfg.d_model,
            "d_ff": cfg.d_ff,
            "n_heads": cfg.n_heads,
            "dropout": cfg.dropout,
            "pos_encoding": cfg.pos_encoding,
            "best_valid_bleu": result["best_valid_bleu"],
            "test_bleu": result["test_bleu"],
        }
        self.results.append(result_row)
        # 将结果写入 group 目录下 JSON
        out_dir = ensure_dir(Path(cfg.log_dir) / cfg.group)
        with open(out_dir / f"summary_{int(time.time())}.json", "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        return result_row

    def run_group(self, group: Literal["heads", "dimensions", "dropout", "posenc", "baseline"]) -> List[Dict[str, Any]]:
        """Run an ablation group similar to Table 3 in the paper.

        Args:
            group: One of "heads", "dimensions", "dropout", "posenc", "baseline".

        Returns:
            List of results dicts.
        """
        cfgs: List[ExperimentConfig] = []
        base = dataclasses.replace(self.base_cfg)

        if group == "baseline":
            cfgs = [dataclasses.replace(base, run_name=f"baseline_{base.src_lang}-{base.tgt_lang}")]
        elif group == "heads":
            # Vary number of heads keeping computation roughly similar by adjusting per-head dim
            for h in [1, 4, 8, 16]:
                # 保持 d_model 不变（实际论文里保持 per-head dim 变动）
                cfgs.append(
                    dataclasses.replace(
                        base,
                        n_heads=h,
                        run_name=f"heads_h{h}_{base.src_lang}-{base.tgt_lang}",
                    )
                )
        elif group == "dimensions":
            # Vary d_model and d_ff
            for (dm, df) in [(256, 1024), (512, 2048), (1024, 4096)]:
                cfgs.append(
                    dataclasses.replace(
                        base,
                        d_model=dm,
                        d_ff=df,
                        n_heads=8 if dm % 8 == 0 else 4,  # 保证可整除
                        run_name=f"dim_dm{dm}_df{df}_{base.src_lang}-{base.tgt_lang}",
                    )
                )
        elif group == "dropout":
            for p in [0.0, 0.1, 0.2, 0.3]:
                cfgs.append(
                    dataclasses.replace(
                        base,
                        dropout=p,
                        run_name=f"dropout_p{str(p).replace('.','_')}_{base.src_lang}-{base.tgt_lang}",
                    )
                )
        elif group == "posenc":
            for t in ["sinusoidal", "learned"]:
                cfgs.append(
                    dataclasses.replace(
                        base,
                        pos_encoding=t,  # type: ignore[arg-type]
                        run_name=f"posenc_{t}_{base.src_lang}-{base.tgt_lang}",
                    )
                )
        else:
            raise ValueError(f"Unknown group: {group}")

        all_results = []
        for cfg in cfgs:
            all_results.append(self._run_single(cfg))
        return all_results


# -----------------------------
# Sanity Checks
# -----------------------------


def run_sanity_checks() -> None:
    """Run minimal sanity checks for shapes and masks.

    This does not train; it only validates forward pass shapes and decoding.
    """
    logger.info("Running sanity checks for model shapes and logic ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_model, n_heads, d_ff = 256, 4, 1024
    vocab = 100
    max_len = 64
    model = TransformerSeq2Seq(
        vocab_src=vocab,
        vocab_tgt=vocab,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=2,
        d_ff=d_ff,
        dropout=0.1,
        max_len=max_len,
        pos_encoding="sinusoidal",
        weight_tying=False,
    ).to(device)

    B, S, T = 2, 10, 12
    src = torch.randint(4, vocab, (B, S), device=device)
    tgt_in = torch.randint(4, vocab, (B, T), device=device)

    src_kpm = torch.zeros((B, S), dtype=torch.bool, device=device)
    tgt_kpm = torch.zeros((B, T), dtype=torch.bool, device=device)
    tgt_mask = torch.triu(torch.ones((1, T, T), dtype=torch.bool, device=device), diagonal=1)

    logits = model(
        src_ids=src,
        tgt_in=tgt_in,
        tgt_mask=tgt_mask,
        src_key_padding_mask=src_kpm,
        tgt_key_padding_mask=tgt_kpm,
    )
    assert logits.shape == (B, T, vocab), f"Unexpected logits shape: {logits.shape}"

    gen = model.greedy_decode(
        src_ids=src,
        src_key_padding_mask=src_kpm,
        max_len=20,
        bos_id=1,
        eos_id=2,
    )
    assert gen.shape[0] == B and gen.shape[1] >= 1, "Greedy decode shape mismatch."
    logger.info("Sanity checks passed ✅")


# -----------------------------
# Main and CLI
# -----------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Optional argument sequence.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Transformer Ablation Study on WMT14 (PyTorch + tensorboardX + loguru)"
    )
    parser.add_argument("--run_name", type=str, default="baseline_en-de", help="Run name.")
    parser.add_argument("--group", type=str, default="baseline", choices=["baseline", "heads", "dimensions", "dropout", "posenc"], help="Ablation group.")
    parser.add_argument("--src_lang", type=str, default="en", help="Source language.")
    parser.add_argument("--tgt_lang", type=str, default="de", help="Target language.")
    parser.add_argument("--vocab_size", type=int, default=37000, help="SentencePiece vocab size.")
    parser.add_argument("--max_len", type=int, default=256, help="Max token length.")
    parser.add_argument("--batch_size", type=int, default=8192, help="Approx tokens per batch (heuristic).")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=2.0, help="Base LR factor (Noam).")
    parser.add_argument("--warmup", type=int, default=4000, help="Warmup steps.")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing epsilon.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout.")
    parser.add_argument("--d_model", type=int, default=512, help="Model hidden size.")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--n_layers", type=int, default=6, help="Number of layers.")
    parser.add_argument("--d_ff", type=int, default=2048, help="FFN hidden size.")
    parser.add_argument("--pos_enc", type=str, default="sinusoidal", choices=["sinusoidal", "learned"], help="Positional encoding type.")
    parser.add_argument("--no_tying", action="store_true", help="Disable weight tying.")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Grad clipping norm.")
    parser.add_argument("--no_fp16", action="store_true", help="Disable mixed precision.")
    parser.add_argument("--save_every", type=int, default=5000, help="Save checkpoint every N steps.")
    parser.add_argument("--log_every", type=int, default=100, help="Log every N steps.")
    parser.add_argument("--eval_every", type=int, default=5000, help="Eval every N steps.")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device.")
    parser.add_argument("--log_dir", type=str, default="runs", help="TensorBoard log dir.")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Checkpoint dir.")
    parser.add_argument("--data_dir", type=str, default="data_wmt14", help="Local data dir fallback.")
    parser.add_argument("--spm_dir", type=str, default="spm_models", help="SentencePiece model dir.")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Limit train samples (debug).")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Limit eval samples (debug).")
    parser.add_argument("--sanity_check", action="store_true", help="Run sanity checks and exit.")
    return parser.parse_args(argv)


def build_config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    """Build ExperimentConfig from argparse args.

    Args:
        args: Parsed arguments.

    Returns:
        ExperimentConfig instance.
    """
    cfg = ExperimentConfig(
        run_name=args.run_name,
        log_dir=args.log_dir,
        save_dir=args.save_dir,
        data_dir=args.data_dir,
        spm_dir=args.spm_dir,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        vocab_size=args.vocab_size,
        max_len=args.max_len,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        warmup_steps=args.warmup,
        label_smoothing=args.label_smoothing,
        dropout=args.dropout,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        pos_encoding=args.pos_enc,
        weight_tying=not args.no_tying,
        grad_clip=args.grad_clip,
        fp16=not args.no_fp16,
        save_every=args.save_every,
        log_every=args.log_every,
        eval_every=args.eval_every,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        group=args.group,
    )
    return cfg


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Main entry point for ablation experiments."""
    args = parse_args(argv)
    # 设置日志格式
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{message}</cyan>")

    if args.sanity_check:
        run_sanity_checks()
        return

    cfg = build_config_from_args(args)
    set_global_seed(cfg.seed)

    # 单组/多组消融实验入口
    ablation = AblationRunner(cfg)
    results = ablation.run_group(cfg.group)

    # 打印总览表
    logger.info("==== Ablation Summary ====")
    for row in results:
        logger.info(
            f"[{row['run_name']}] h={row['n_heads']} d_model={row['d_model']} d_ff={row['d_ff']} "
            f"dropout={row['dropout']} pos={row['pos_encoding']} | "
            f"best_valid_BLEU={row['best_valid_bleu']:.2f} test_BLEU={row['test_bleu']:.2f}"
        )


if __name__ == "__main__":
    main()