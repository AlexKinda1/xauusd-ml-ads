"""Phase-1 entry point: load OHLCV, fetch macro + sentiment, build aligned dataset.

Usage:
    poetry run python scripts/01_collect_all_data.py
    poetry run python scripts/01_collect_all_data.py --skip-external

The ``--skip-external`` flag is for offline / sandboxed environments: only the
local OHLCV CSV is processed; macro & sentiment are skipped. The aligned
dataset is then OHLCV-only and macro features will be missing downstream.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data import align, collect_macro, collect_ohlcv, collect_sentiment
from src.utils.config import load_data_config
from src.utils.logging import get_logger
from src.utils.seed import set_global_seed

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 — data collection")
    parser.add_argument(
        "--skip-external",
        action="store_true",
        help="Skip macro & sentiment downloads (offline mode).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Abort if OHLCV validation reports ERROR-severity issues.",
    )
    args = parser.parse_args()

    set_global_seed(42)
    cfg = load_data_config()

    # 1. OHLCV
    logger.info("=== Step 1/4: load + validate XAU/USD H1 OHLCV ===")
    ohlcv = collect_ohlcv.run(strict=args.strict)

    # 2 & 3. Macro & sentiment
    if not args.skip_external:
        logger.info("=== Step 2/4: fetch macro series ===")
        collect_macro.run(skip_on_error=True)
        logger.info("=== Step 3/4: fetch sentiment series ===")
        collect_sentiment.run()
    else:
        logger.warning("--skip-external set — macro & sentiment NOT fetched")

    # 4. Align
    logger.info("=== Step 4/4: align external sources onto H1 grid ===")
    externals = align.load_external_parquets("data/external")
    aligned = align.build_aligned_dataset(ohlcv, externals)
    out_path = align.save_aligned(aligned, "data/processed/dataset_aligned.parquet")

    # Quick summary
    summary = {
        "ohlcv": collect_ohlcv.summary_stats(ohlcv),
        "external_sources": list(externals.keys()),
        "aligned_shape": list(aligned.shape),
        "aligned_path": str(out_path),
    }
    Path("reports/tables").mkdir(parents=True, exist_ok=True)
    Path("reports/tables/phase1_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Phase 1 complete. Summary written to reports/tables/phase1_summary.json")


if __name__ == "__main__":
    main()
