from __future__ import annotations
import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

from itemcf.cli_pipeline import run_pipeline

if __name__ == "__main__":
    # 示例运行：
    # python scripts/run_pipeline.py --top_k 50 --last_n 200 --final_topn 100
    run_pipeline()