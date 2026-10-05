"""Pins for the crosswalk: placements that have gone wrong once, and the shape of the
mapping tables (no duplicate keys in any dict literal: a duplicate silently replaces
the earlier entry and its refinement rules, which once put Apple in Consumer
Discretionary)."""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

import pytest

import gics_classifier.crosswalk as x
from gics_classifier.structure import load_structure

CROSSWALK_SOURCE = Path(x.__file__)


def test_no_dict_literal_has_duplicate_keys():
    tree = ast.parse(CROSSWALK_SOURCE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            keys = [k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)]
            dups = sorted(k for k, n in Counter(keys).items() if n > 1)
            assert not dups, f"duplicate keys at line {node.lineno}: {dups}"


def test_no_key_carries_whitespace():
    for table in (x.YAHOO_TO_GICS, x.SIC_TO_GICS, x.SIC_PREFIX_TO_GICS):
        assert all(k == k.strip() for k in table)


def ev(ticker, name, yahoo=None, sic=None, sicd=None, desc=None):
    return x.Evidence(1, ticker, name, desc, sic, sicd, yahoo)


APPLE = ("Apple is among the largest companies in the world, with a broad portfolio of hardware and "
         "software products. Apple's iPhone accounts for the majority of the firm's sales, and Apple's "
         "other products, such as the Mac, iPad, and Watch, are designed around the iPhone.")
LINDE = ("Linde is the largest industrial gas supplier in the world, with operations in over 100 countries. "
         "The firm's main products are atmospheric gases (including oxygen, nitrogen, and argon).")
AIR_PRODUCTS = ("Since its founding in 1940, Air Products has become one of the leading industrial gas "
                "suppliers globally. The company is the world's largest supplier of hydrogen and helium.")
HUDSON = ("Hudson Technologies Inc is a refrigerant services company providing solutions to recurring "
          "challenges within the refrigeration industry. Its offerings include refrigerant and industrial "
          "gas sales, refrigerant management services and reclamation.")
WEARABLE = ("Wearable Devices Ltd develops and sells human-machine interface solutions for the smart "
            "wearables industry. These digital devices include consumer electronics, smart watches, "
            "smartphones, AR glasses, VR headsets, televisions, PCs, laptop computers, drones, robotics.")


@pytest.mark.parametrize("evidence, code, method", [
    # The row that exposed the duplicate-key bug: Yahoo says Consumer Electronics, the
    # description says the iPhone is the business, SIC 3571 agrees with hardware.
    (ev("AAPL", "Apple Inc.", "Consumer Electronics", "3571", "ELECTRONIC COMPUTERS", APPLE), "45202030", "yahoo+sic"),
    # A compatibility list is not a product line.
    (ev("WLDS", "Wearable Devices Ltd.", "Consumer Electronics", "3576", "COMPUTER COMMUNICATIONS EQUIPMENT", WEARABLE), "25201010", "yahoo"),
    # Industrial-gas producers move; a reclaimer that mentions gas sales does not.
    (ev("LIN", "Linde plc", "Specialty Chemicals", "2810", "INDUSTRIAL INORGANIC CHEMICALS", LINDE), "15101040", "yahoo+sic"),
    (ev("APD", "Air Products & Chemicals", "Specialty Chemicals", "2810", "INDUSTRIAL INORGANIC CHEMICALS", AIR_PRODUCTS), "15101040", "yahoo+sic"),
    (ev("HDSN", "Hudson Technologies Inc", "Specialty Chemicals", "5080", "WHOLESALE-MACHINERY, EQUIPMENT & SUPPLIES", HUDSON), "15101050", "yahoo"),
    # Two sources agreeing is the strongest signal we hold.
    (ev("CTBI", "Community Trust Bancorp", "Banks - Regional", "6022", "STATE COMMERCIAL BANKS", "A bank holding company."), "40101015", "yahoo+sic"),
])
def test_placements(evidence, code, method):
    p = x.classify(evidence)
    assert (p.code, p.method) == (code, method), p.rationale


@pytest.mark.parametrize("evidence, reason", [
    (ev("XYZA", "XYZ Acquisition Corp. Class A", "Shell Companies", "6770", "BLANK CHECKS", "A blank check company."), "shell_company"),
    (ev("OFSSH", "OFS Capital Corporation 4.95% Notes due 2028", "Asset Management", "6282", None, None), "not_common_equity"),
    (ev("ZZZZ", "Unknown Holdings", None, None, None, None), "no_evidence"),
])
def test_exclusions(evidence, reason):
    p = x.classify(evidence)
    assert p.code is None and p.excluded_reason == reason, p.rationale


def test_every_target_code_is_a_gics_sub_industry():
    subs = {n.code for n in load_structure().sub_industries}
    assert x.all_target_codes() <= subs
