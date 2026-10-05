"""Command line: ``gics-classify crosswalk`` and ``gics-classify model`` over CSV files.

Input CSV columns (header names, any order; missing ones are treated as empty):
``id``, ``ticker``, ``name``, ``description``, ``sic_code``, ``sic_description``,
``yahoo_industry``, ``market_cap``. The model's training CSV needs a ``label`` column
holding the GICS sub-industry code; the default training set is the input rows whose
crosswalk placement has confidence >= 0.9.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from .crosswalk import Evidence, Placement, classify
from .structure import load_structure

COLUMNS = ("id", "ticker", "name", "description", "sic_code", "sic_description", "yahoo_industry", "market_cap")


def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return [{k: (v or "") for k, v in row.items()} for row in csv.DictReader(fh)]


def _evidence(row: dict, i: int) -> Evidence:
    return Evidence(row.get("id") or i, row.get("ticker") or "", row.get("name") or None, row.get("description") or None,
                    row.get("sic_code") or None, row.get("sic_description") or None, row.get("yahoo_industry") or None)


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def cmd_crosswalk(args: argparse.Namespace) -> int:
    structure = load_structure(args.structure)
    rows = _read(Path(args.input))
    out = []
    for i, row in enumerate(rows):
        p = classify(_evidence(row, i))
        out.append({**{k: row.get(k, "") for k in COLUMNS}, "gics_code": p.code or "",
                    "gics_path": structure.path(p.code) if p.code else "", "confidence": p.confidence if p.confidence is not None else "",
                    "method": p.method, "excluded_reason": p.excluded_reason or "", "rationale": p.rationale})
    _write(Path(args.output), out)
    placed = sum(1 for o in out if o["gics_code"])
    print(f"{len(out)} rows: {placed} placed, {len(out) - placed} excluded -> {args.output}", file=sys.stderr)
    return 0


def cmd_model(args: argparse.Namespace) -> int:
    from .model import HybridModel, decide
    from .text import company_text

    structure = load_structure(args.structure, args.definitions)
    if not structure.has_definitions():
        print("note: no definitions loaded; similarity features are built from node names only "
              "(run parse-workbook on your copy of the GICS workbook for the full model)", file=sys.stderr)
    rows = _read(Path(args.input))
    evidence = [_evidence(r, i) for i, r in enumerate(rows)]
    priors = [classify(ev) for ev in evidence]
    mcap = [float(r["market_cap"]) if r.get("market_cap") else None for r in rows]
    texts = [company_text(ev, m) for ev, m in zip(evidence, mcap)]

    model = HybridModel(structure, C=args.C)
    model.fit_features(texts)
    X = model.features(texts)

    if args.train:
        train_rows = _read(Path(args.train))
        train_ev = [_evidence(r, i) for i, r in enumerate(train_rows)]
        train_texts = [company_text(ev, float(r["market_cap"]) if r.get("market_cap") else None) for ev, r in zip(train_ev, train_rows)]
        y = np.array([r["label"].strip() for r in train_rows])
        Xt = model.features(train_texts)
    else:
        idx = [i for i, p in enumerate(priors) if p.code and p.confidence is not None and p.confidence >= args.label_min_confidence]
        y = np.array([priors[i].code for i in idx])
        Xt = X[idx]
    print(f"training on {len(y)} labels over {len(set(y))} sub-industries", file=sys.stderr)
    if not args.no_cv and len(y) >= 50:
        cv = model.cross_validate(Xt, y, args.folds)
        print(f"cross-validation ({args.folds} folds, {cv['n']} rows, {cv['classes']} classes): sub-industry {cv['sub_industry']:.3f} | "
              f"industry {cv['industry']:.3f} | industry group {cv['industry_group']:.3f} | sector {cv['sector']:.3f}", file=sys.stderr)
    model.fit(Xt, y)
    proba, classes = model.predict_proba(X)
    sims = model.similarities(texts)

    out = []
    for i, (row, prior) in enumerate(zip(rows, priors)):
        if prior.excluded_reason:
            d = None
        else:
            nearest = (model.codes[int(np.argmax(sims[i]))], float(np.max(sims[i])))
            d = decide(model, proba[i], classes, prior if prior.code else None, nearest_definition=nearest,
                       override_threshold=args.override_threshold, max_prior_prob=args.max_prior_prob,
                       min_class_labels=args.min_class_labels, min_confidence=args.min_confidence)
        code = d.code if d else None
        out.append({**{k: row.get(k, "") for k in COLUMNS},
                    "gics_code": code or "", "gics_path": structure.path(code) if code else "",
                    "confidence": round(d.confidence, 3) if d and d.confidence is not None else "",
                    "method": d.method if d else prior.method, "outcome": d.outcome if d else "excluded",
                    "excluded_reason": prior.excluded_reason or "", "rationale": d.rationale if d else prior.rationale})
    _write(Path(args.output), out)
    from collections import Counter
    print(json.dumps(dict(Counter(o["outcome"] for o in out)), indent=1), file=sys.stderr)
    return 0


def cmd_parse_workbook(args: argparse.Namespace) -> int:
    from .workbook import parse_workbook, write_definitions

    parsed = parse_workbook(args.workbook, args.sheet)
    write_definitions(parsed, args.out)
    print(f"{len(parsed['definitions'])} sub-industry definitions -> {args.out} (sheet {parsed['sheet']!r})", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="gics-classify", description="Place companies in the GICS structure")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("crosswalk", help="rule-based placement from Yahoo industry / SIC / name words")
    c.add_argument("input"); c.add_argument("output")
    c.add_argument("--structure", default=None, help="structure JSON (default: bundled GICS 2023)")
    c.set_defaults(func=cmd_crosswalk)

    m = sub.add_parser("model", help="hybrid text model: train, then score every row against the crosswalk prior")
    m.add_argument("input"); m.add_argument("output")
    m.add_argument("--structure", default=None)
    m.add_argument("--definitions", default=None, help="JSON from parse-workbook")
    m.add_argument("--train", default=None, help="CSV with a `label` column; default: crosswalk rows with confidence >= --label-min-confidence")
    m.add_argument("--label-min-confidence", type=float, default=0.9)
    m.add_argument("--C", type=float, default=8.0)
    m.add_argument("--folds", type=int, default=5)
    m.add_argument("--no-cv", action="store_true")
    m.add_argument("--override-threshold", type=float, default=0.7)
    m.add_argument("--max-prior-prob", type=float, default=0.2)
    m.add_argument("--min-class-labels", type=int, default=5)
    m.add_argument("--min-confidence", type=float, default=0.5)
    m.set_defaults(func=cmd_model)

    w = sub.add_parser("parse-workbook", help="extract sub-industry definitions from your copy of the GICS workbook")
    w.add_argument("workbook"); w.add_argument("--sheet", default=None); w.add_argument("--out", default="gics_definitions.json")
    w.set_defaults(func=cmd_parse_workbook)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
