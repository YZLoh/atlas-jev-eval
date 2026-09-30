"""Ground truth, kept strictly independent of any evaluator.

The trap this module exists to avoid
------------------------------------
"Did the attack succeed?" is a property of *the response the target actually
produced*, not of the test case. A case cannot therefore carry a pre-authored
label for it: run the same case against two models, or the same model twice at
temperature > 0, and the correct answer changes.

If ground truth were stored on the case, every "Jev accuracy" figure in this
project would silently be measuring something else - mostly target-model
variance - and the central research question would be unanswerable.

So there are two test beds:

**Bed A - fixtures (evaluator-only).** A fixed ``(attack_prompt, response)``
pair with an immutable hand-authored label. No target model runs. Evaluator
accuracy here is measured against real ground truth and is exactly reproducible.
This is the bed in which Jev can be *shown* to be wrong.

**Bed B - live target.** The target model runs and a human labels what it
actually produced. Labels live in ``test_cases/labels.jsonl``, keyed by a hash
of the response text, so a label stays bound to the exact text it was assigned
to. If the target is re-run and says something different, the label is treated
as **missing**, not reused.

Unlabelled responses are reported as unlabelled. They are never defaulted to a
value, because defaulting would fabricate ground truth.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from atlas_jev.config import TEST_CASES_DIR

LABELS_FILENAME = "labels.jsonl"


@dataclass
class Label:
    """A human judgement about one specific target response."""

    case_id: str
    response_hash: str
    # Question name -> ground-truth value. Keys match QuestionSpec names.
    values: dict[str, Any] = field(default_factory=dict)
    labeller: str = "unknown"
    note: str = ""
    # Kept for human readability when auditing the file by eye. Not used for
    # matching - the hash is authoritative.
    response_excerpt: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "response_hash": self.response_hash,
            "values": self.values,
            "labeller": self.labeller,
            "note": self.note,
            "response_excerpt": self.response_excerpt,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Label:
        return cls(
            case_id=data["case_id"],
            response_hash=data["response_hash"],
            values=data.get("values", {}),
            labeller=data.get("labeller", "unknown"),
            note=data.get("note", ""),
            response_excerpt=data.get("response_excerpt", ""),
        )


class LabelStore:
    """Append-only store of human labels, keyed by ``(case_id, response_hash)``."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or TEST_CASES_DIR / LABELS_FILENAME
        self._labels: dict[tuple[str, str], Label] = {}
        self.load()

    def load(self) -> None:
        self._labels.clear()
        if not self.path.exists():
            return
        for line_no, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                label = Label.from_json(json.loads(line))
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"{self.path}:{line_no} is not a valid label record: {exc}")
            # A later line for the same key supersedes an earlier one, so a
            # mislabelling can be corrected by appending rather than editing.
            self._labels[(label.case_id, label.response_hash)] = label

    def get(self, case_id: str, response_hash: str) -> Label | None:
        return self._labels.get((case_id, response_hash))

    def add(self, label: Label) -> None:
        self._labels[(label.case_id, label.response_hash)] = label
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(label.to_json(), ensure_ascii=False) + "\n")

    def __len__(self) -> int:
        return len(self._labels)

    def drifted(self, records: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
        """Cases labelled before whose response text has since changed.

        Hash-keying correctly refuses to reuse a label written for different
        text. But the *consequence* is that a re-run silently drops that label
        and the case falls out of scoring - and a dropped negative label can
        make an evaluator's precision improve, which looks like good news.

        Observed in practice: llama3.2:3b at temperature=0 with a fixed seed is
        not bit-reproducible across runs, so this is not a hypothetical.

        Returns ``(case_id, old_hash, new_hash)`` for each affected case.
        """
        labelled_by_case: dict[str, set[str]] = {}
        for case_id, response_hash in self._labels:
            labelled_by_case.setdefault(case_id, set()).add(response_hash)

        drift: list[tuple[str, str, str]] = []
        for record in records:
            case_id = record.get("case_id")
            new_hash = record.get("response_hash")
            if record.get("error") or record.get("fixture_labels"):
                continue
            known = labelled_by_case.get(case_id)
            if known and new_hash not in known:
                drift.append((case_id, min(known), new_hash))
        return drift
