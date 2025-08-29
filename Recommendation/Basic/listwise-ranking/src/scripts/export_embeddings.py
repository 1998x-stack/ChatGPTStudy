from __future__ import annotations
import argparse
import os
import torch
from loguru import logger

from ..models.two_tower import TwoTowerScorer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--project_root', type=str, required=True)
    p.add_argument('--ckpt', type=str, required=False, help='Path to torch saved state_dict .pt/.ckpt')
    p.add_argument('--num_users', type=int, required=True)
    p.add_argument('--num_items', type=int, required=True)
    p.add_argument('--embedding_dim', type=int, default=64)
    p.add_argument('--mlp_hidden', type=int, default=128)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--out', type=str, default='embeddings.pt')
    return p.parse_args()


def main():
    args = parse_args()
    model = TwoTowerScorer(args.num_users, args.num_items, args.embedding_dim, args.mlp_hidden, args.dropout)
    if args.ckpt and os.path.exists(args.ckpt):
        state = torch.load(args.ckpt, map_location='cpu')
        model.load_state_dict(state)
        logger.info("Loaded checkpoint {}", args.ckpt)
    torch.save({
        'user_emb': model.user_emb.weight.detach().cpu(),
        'item_emb': model.item_emb.weight.detach().cpu(),
    }, os.path.join(args.project_root, args.out))
    logger.info("Exported embeddings to {}", os.path.join(args.project_root, args.out))


if __name__ == '__main__':
    main()