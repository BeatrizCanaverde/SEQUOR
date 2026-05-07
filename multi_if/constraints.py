from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict


class ConstraintData(TypedDict):
    id: str
    text: str


@dataclass(frozen=True)
class Constraint:
    id: str
    text: str
    slug: str


def load_constraints(path: Path) -> list[Constraint]:
    """Load constraints from a JSONL file with a `constraint` field."""
    constraints: list[Constraint] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            raw_text = record.get("constraint") or record.get("instruction") or record.get("text")
            if not raw_text:
                continue
            cid = str(record.get("id") or record.get("name") or f"c{idx:04d}")
            text = str(raw_text).strip()
            slug = _slugify(text) or cid
            constraints.append(Constraint(id=cid, text=text, slug=slug))
    if not constraints:
        raise ValueError(f"No constraints found in {path}")
    return constraints


def load_constraint_groups(constraints_path: Path) -> list[list[Constraint]]:
    groups: list[list[Constraint]] = []

    with constraints_path.open("r", encoding="utf-8") as f:
        for group_idx, line in enumerate(f, start=1):
            if not line.strip():
                continue

            record = json.loads(line)
            raw_constraints = record.get("constraints")
            raw_ids = record.get("constraint_ids")
            if not raw_constraints:
                continue

            group: list[Constraint] = []
            group_id = str(record.get("id", "")).strip()
            base_id = group_id or None
            for constraint_idx, text in enumerate(raw_constraints, start=1):
                text_str = str(text).strip()
                if not text_str:
                    continue

                cid = None
                if raw_ids and len(raw_ids) >= constraint_idx:
                    cid = str(raw_ids[constraint_idx - 1]).strip() or None

                if not cid:
                    cid = f"{base_id}_c{constraint_idx:02d}" if base_id else f"g{group_idx:04d}_c{constraint_idx:02d}"

                slug = cid
                group.append(Constraint(id=cid, text=text_str, slug=slug))

            if group:
                groups.append(group)

    if not groups:
        raise ValueError(f"No constraint groups found in {constraints_path}")

    return groups


def constraint_to_data(constraint: Constraint) -> ConstraintData:
    return ConstraintData(id=constraint.id, text=constraint.text)


def _slugify(text: str, max_length: int = 60) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    if len(text) > max_length:
        text = text[:max_length].rstrip("-")
    return text
