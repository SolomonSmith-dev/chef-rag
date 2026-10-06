"""Spend ledger enforcing the hard USD cap on live LLM and embedding calls."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class BudgetExceeded(RuntimeError):
    """Raised before a call that would exceed the cap."""


class SpendLedger:
    """Append-only JSON ledger. Every live call records tokens and estimated cost."""

    def __init__(self, path: Path, cap_usd: float = 5.0) -> None:
        self.path = path
        self.cap_usd = cap_usd
        self.entries: list[dict[str, Any]] = (
            json.loads(path.read_text())["entries"] if path.exists() else []
        )

    @property
    def total_usd(self) -> float:
        return float(sum(e["usd"] for e in self.entries))

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.cap_usd - self.total_usd)

    def check(self) -> None:
        if self.total_usd >= self.cap_usd:
            raise BudgetExceeded(f"spend cap reached: ${self.total_usd:.4f} of ${self.cap_usd:.2f}")

    def ensure_room(self, projected_usd: float) -> None:
        if self.total_usd + projected_usd > self.cap_usd:
            raise BudgetExceeded(
                f"projected ${projected_usd:.4f} + spent ${self.total_usd:.4f} "
                f"exceeds cap ${self.cap_usd:.2f}"
            )

    def record(
        self, label: str, model: str, prompt_tokens: int, completion_tokens: int, usd: float
    ) -> None:
        self.entries.append(
            {
                "ts": datetime.now(UTC).isoformat(),
                "label": label,
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "usd": usd,
            }
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"cap_usd": self.cap_usd, "entries": self.entries}, indent=1)
        )
