# gics-classifier

Place companies in the GICS structure (11 sectors, 25 industry groups, 74 industries,
163 sub-industries) from the evidence you already have: a Yahoo Finance industry, a SEC
SIC code, the company's name and description. Two classifiers, one vocabulary:

| | What it is | Needs | Good for |
|---|---|---|---|
| **Crosswalk** | Rules: every Yahoo industry and SIC code mapped to a sub-industry, with keyword refinements where a source category fans out | nothing | a first placement for any company with a Yahoo industry or a SIC code |
| **Hybrid model** | TF-IDF over the company's text **plus its cosine similarity to each sub-industry definition**, into a class-balanced logistic regression; a decision rule that confirms, overrides or defers to the crosswalk | labelled examples (the crosswalk's two-source agreements are a free starting set); optionally the definitions | reading the company itself, catching stale SIC codes and coarse Yahoo buckets |

On a universe of 6,237 US common stocks (October 2026) the model scored **96.4%** at
sub-industry and **99.2%** at sector on 5-fold cross-validation over 2,713 two-source
labels, and overruled the crosswalk on 5 companies, all correctly. Your numbers will
depend on your data; the model prints its own evaluation every run.

> **GICS is a registered trademark of MSCI and S&P Global.** This package ships the
> published structure (codes and names) and makes its own placements by rules and a
> model. They are not GICS Direct data and not the owners' assignments. The
> sub-industry *definitions* are the owners' copyrighted text and are not included;
> `gics-classify parse-workbook` extracts them from your own copy of the official
> workbook into a local file. See [NOTICE.md](NOTICE.md).

## Install

```bash
pip install git+https://github.com/simulationcoder/gics-classifier
# or, for the workbook parser too:
pip install "gics-classifier[workbook] @ git+https://github.com/simulationcoder/gics-classifier"
```

Python 3.10+, scikit-learn, numpy, scipy. `openpyxl` only for `parse-workbook`.

## Use

A CSV with any of these columns (missing ones are fine): `id, ticker, name,
description, sic_code, sic_description, yahoo_industry, market_cap`.

```bash
# 1. Rules only. Seconds. Writes gics_code, gics_path, confidence, method, rationale.
gics-classify crosswalk companies.csv placed.csv

# 2. Your copy of the owners' workbook -> local definitions file (git-ignored).
gics-classify parse-workbook "Sector Map 2025.xlsx" --out gics_definitions.json

# 3. The model: trains on the crosswalk rows with confidence >= 0.9 (or --train labels.csv
#    with a `label` column of sub-industry codes), scores every row against the crosswalk
#    prior, prints its cross-validation, writes outcome (agree / override / prior_kept /
#    placed_new / unplaced) and a rationale per row.
gics-classify model companies.csv placed.csv --definitions gics_definitions.json
```

As a library:

```python
from gics_classifier import Evidence, classify, load_structure

p = classify(Evidence(1, "AAPL", "Apple Inc.", "Apple's iPhone accounts for the majority of the firm's sales...",
                      "3571", "ELECTRONIC COMPUTERS", "Consumer Electronics"))
p.code, p.confidence, p.method   # ('45202030', 0.9, 'yahoo+sic')
load_structure().path(p.code)    # 'Information Technology > Technology Hardware & Equipment > ... > Technology Hardware, Storage & Peripherals'
```

## How the crosswalk decides

Yahoo leads (it is current), SIC confirms or fills (it is authoritative but stale: a
company that came public through a SPAC keeps code 6770 for years), name words are the
last resort. Confidence follows the evidence:

| Evidence | Confidence |
|---|---|
| Yahoo and SIC map to the same sub-industry | 0.90 |
| same industry group / same sector; Yahoo wins | 0.75 / 0.70 |
| Yahoo only, or Yahoo over a disagreeing SIC | 0.65 / 0.60 |
| SIC only | 0.55 |
| name words only | 0.35 |

Three outcomes are deliberately not placements and come back with an `excluded_reason`:
`shell_company` (SPACs and blank checks: the GICS methodology assigns them nothing),
`not_common_equity` (names that are notes, preferreds, depositary shares or funds),
`no_evidence`.

The mapping is a table in [`gics_classifier/crosswalk.py`](gics_classifier/crosswalk.py).
Refinements are anchored on what a company *does* ("industrial gas supplier", "sells
smartphones"), not on words that merely appear: a refrigerant reclaimer that mentions
"industrial gas sales" stays where it is. A test parses the module and fails on any
duplicate key, because a duplicate silently replaces the earlier entry and its rules.

## How the model decides

1. **Text.** Name, description, SIC description, plus tokens for the SIC code, the Yahoo
   industry and a market-cap size bucket (some GICS lines, such as Diversified versus
   Regional Banks, are about scale, which no description states).
2. **TF-IDF** over word uni- and bigrams, fitted on the company texts *and* the
   definitions, so both live in one vocabulary.
3. **Definition similarity.** Cosine between the company and each of the 163 definition
   texts: 163 dense features that let the definitions shape the decision. Without
   definitions the similarities are to the node names only, which still helps a little.
4. **Logistic regression**, multinomial, class-balanced (Regional Banks has hundreds of
   labels, Diversified Banks a dozen; unweighted, every bank becomes regional). The top
   probability is the confidence.
5. **Against the prior.** Agreement raises confidence. An override needs probability
   >= 0.7, more than the prior's confidence, the prior's class given <= 0.2, and at
   least 5 training labels for the prior's class: a class the model barely saw it
   cannot reject (without that rule every reinsurer became Property & Casualty).
   Otherwise the prior stands and the model's dissent is kept in the rationale.

All thresholds are CLI flags. A full retrain over 6,000 companies takes about a minute.

## A full run, as data

[`examples/us_common_stocks_2026-10-04.csv`](examples/us_common_stocks_2026-10-04.csv)
is the output of both passes over every active US common stock on 4 October 2026:
6,240 rows, one per ticker, with the four GICS levels (codes and names), the
confidence, how the row was decided (`method`) and, for the 589 rows with no
placement, why (`excluded_reason`). The inputs were company descriptions from a market
data vendor, SEC SIC codes and Yahoo Finance industries; only the SIC code is reproduced
here, the rest is not ours to redistribute. The placements are ours, not S&P's or
MSCI's, and a few will be wrong: the `confidence` column says how much to trust each.

| method | rows | meaning |
|---|---|---|
| `model+crosswalk` | 4,626 | the model agreed with the crosswalk; confidence is the higher of the two |
| `yahoo+sic`, `yahoo`, `sic`, `keywords` | 1,019 | the crosswalk's placement stood; the model dissented or had no view strong enough to overrule |
| `model` | 5 | the model overruled the crosswalk |
| excluded | 589 | 332 shell companies, 118 notes / preferreds / funds filed as common stock, 139 with no usable evidence |

## What it cannot do

- No meaning, only vocabulary: synonyms are strangers to TF-IDF, and marketing-language
  descriptions give it nothing. A sentence-embedding similarity is the natural third
  feature block if you need that.
- Labels shape the classes: sub-industries with no training examples can never be
  proposed by the model (the crosswalk can still place them).
- Scale is invisible beyond the size token.

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

MIT licensed. Built for [Nurral](https://nurral.com); the structure parser, crosswalk and
model were developed against a live US equity universe and are released as-is.
