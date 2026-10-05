"""The GICS structure: 11 sectors, 25 industry groups, 74 industries, 163 sub-industries.

Codes nest by prefix (``10`` > ``1010`` > ``101010`` > ``10101010``), which is all the
hierarchy a lookup needs. The bundled JSON carries codes and names only; definitions
are S&P/MSCI text and are loaded from a file you produce with
:mod:`gics_classifier.workbook` from your own copy of the GICS workbook.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

LEVELS = ("Sector", "Industry Group", "Industry", "Sub-Industry")
PATH_SEPARATOR = " > "


@dataclass(frozen=True)
class Node:
    code: str
    name: str
    level: int  # 1..4
    parent: str | None
    definition: str | None = None


@dataclass
class Structure:
    effective_date: str
    nodes: dict[str, Node] = field(default_factory=dict)

    @property
    def sub_industries(self) -> list[Node]:
        return [n for n in self.nodes.values() if n.level == 4]

    def ancestors(self, code: str) -> list[Node]:
        """Sector, industry group, industry and sub-industry nodes for a code of any level."""
        return [self.nodes[code[:k]] for k in (2, 4, 6, 8) if k <= len(code) and code[:k] in self.nodes]

    def path(self, code: str) -> str:
        return PATH_SEPARATOR.join(n.name for n in self.ancestors(code))

    def name(self, code: str) -> str:
        return self.nodes[code].name

    def definition_text(self, code: str) -> str:
        """Path plus definition, the text the model compares companies against."""
        node = self.nodes[code]
        return f"{self.path(code)}. {node.definition or ''}".strip()

    def has_definitions(self) -> bool:
        return any(n.definition for n in self.sub_industries)

    def with_definitions(self, definitions: dict[str, str]) -> "Structure":
        nodes = {c: Node(n.code, n.name, n.level, n.parent, definitions.get(c, n.definition)) for c, n in self.nodes.items()}
        return Structure(self.effective_date, nodes)


def load_structure(path: str | Path | None = None, definitions: str | Path | None = None) -> Structure:
    """Load the bundled structure, or a JSON file in the same shape (``nodes`` with
    ``code``, ``name``, ``level``, ``parent`` and optionally ``definition``).

    ``definitions`` is an optional JSON file mapping sub-industry code to definition
    text, as written by ``gics-classify parse-workbook``.
    """
    if path is None:
        raw = json.loads(resources.files("gics_classifier").joinpath("data/gics_2023_structure.json").read_text(encoding="utf-8"))
    else:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    nodes = {n["code"]: Node(n["code"], n["name"], int(n["level"]), n.get("parent"), n.get("definition")) for n in raw["nodes"]}
    structure = Structure(raw.get("effective_date", ""), nodes)
    if definitions is not None:
        defs = json.loads(Path(definitions).read_text(encoding="utf-8"))
        if "definitions" in defs:
            defs = defs["definitions"]
        structure = structure.with_definitions({str(k): str(v) for k, v in defs.items()})
    return structure
