"""Extract the sub-industry definitions from your own copy of the GICS workbook.

S&P Dow Jones Indices and MSCI publish the structure with a definition per
sub-industry as an Excel workbook ("GICS Structure" / "Sector Map"; the sheet for
the 2023 structure is titled "Effective close of Mar 17 2023"). The definitions are
their copyrighted text, so this package does not ship them; run::

    gics-classify parse-workbook "Sector Map 2025.xlsx" --out gics_definitions.json

and pass ``--definitions gics_definitions.json`` to the model. Rows with
"(Discontinued)" in the name are dropped.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


def _clean(value) -> str:
    return re.sub(r"\s+", " ", str(value).replace("\xa0", " ")).strip()


def parse_workbook(path: str | Path, sheet: str | None = None) -> dict:
    """Return ``{"effective_date", "nodes": [...with definition...], "definitions": {code: text}}``."""
    import openpyxl  # optional dependency

    wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    ws = wb[sheet] if sheet else next((w for w in wb.worksheets if "effective" in w.title.lower()), wb.worksheets[0])
    rows = list(ws.iter_rows(values_only=True))
    header_idx = next(i for i, r in enumerate(rows) if r and str(r[0] or "").strip().lower() == "sector")
    nodes: list[dict] = []
    cur: dict[int, dict] = {}
    last_sub: dict | None = None
    for row in rows[header_idx + 1:]:
        a, b, c, d, e, f, g, h = (list(row) + [None] * 8)[:8]
        if isinstance(a, int):
            cur[1] = {"code": str(a), "name": _clean(b), "level": 1, "parent": None}
            nodes.append(cur[1])
        if isinstance(c, int):
            cur[2] = {"code": str(c), "name": _clean(d), "level": 2, "parent": cur[1]["code"]}
            nodes.append(cur[2])
        if isinstance(e, int):
            cur[3] = {"code": str(e), "name": _clean(f), "level": 3, "parent": cur[2]["code"]}
            nodes.append(cur[3])
        if isinstance(g, int):
            last_sub = {"code": str(g), "name": _clean(h), "level": 4, "parent": cur[3]["code"], "definition": None}
            nodes.append(last_sub)
        elif g is None and h and last_sub is not None and not any(isinstance(x, int) for x in (a, c, e)):
            text = str(h).replace("\xa0", " ").strip()
            last_sub["definition"] = (last_sub["definition"] + "\n" + text) if last_sub["definition"] else text
    live = [n for n in nodes if "discontinued" not in n["name"].lower()]
    codes = {n["code"] for n in live}
    live = [n for n in live if n["parent"] is None or n["parent"] in codes]
    for n in live:
        n["sheet_name"] = n["name"]
        n["name"] = re.sub(r"\s*\([^)]*\)\s*$", "", n["name"]).replace("*", "").strip()
    effective = ""
    m = re.search(r"(\w+ \d{1,2}, \d{4})", str(rows[1][0] if len(rows) > 1 and rows[1] else ""))
    if m:
        effective = m.group(1)
    return {
        "effective_date": effective,
        "sheet": ws.title,
        "nodes": live,
        "definitions": {n["code"]: n["definition"] for n in live if n["level"] == 4 and n.get("definition")},
    }


def write_definitions(parsed: dict, out: str | Path) -> None:
    Path(out).write_text(json.dumps({"effective_date": parsed["effective_date"], "definitions": parsed["definitions"]},
                                    indent=1, ensure_ascii=False), encoding="utf-8")
