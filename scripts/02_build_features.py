"""Phase-2 entry point: build features and targets from the aligned dataset.

Usage:
    poetry run python scripts/02_build_features.py
"""

from __future__ import annotations

import json
from pathlib import Path

from src.features import pipeline
from src.utils.config import PROJECT_ROOT
from src.utils.logging import get_logger
from src.utils.seed import set_global_seed

logger = get_logger(__name__)


def main() -> None:
    set_global_seed(42)
    feats = pipeline.run()
    summary = pipeline.summarise(feats, horizon=24)
    out = PROJECT_ROOT / "reports/tables/phase2_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    logger.info("Phase 2 complete. Summary written to %s", out)


if __name__ == "__main__":
    main()
