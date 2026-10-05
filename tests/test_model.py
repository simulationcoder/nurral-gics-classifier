import numpy as np

from gics_classifier.crosswalk import Evidence, Placement, classify
from gics_classifier.model import HybridModel, decide, level_of_difference
from gics_classifier.structure import load_structure
from gics_classifier.text import company_text

BANK = "a bank holding company whose subsidiary bank accepts deposits and makes commercial and consumer loans in its region"
BIOTECH = "a clinical-stage biotechnology company developing monoclonal antibody candidates for oncology, with a pipeline in phase 2"
SOFTWARE = "provides cloud-based application software for sales teams and customer relationship management on a subscription basis"
DRILLER = "an offshore drilling contractor that owns and operates drilling rigs under contract to oil and gas producers"


def _rows():
    rows = []
    for i in range(12):
        rows.append((f"Bank {i}", BANK, "6022", "STATE COMMERCIAL BANKS", "Banks - Regional", "40101015"))
        rows.append((f"Bio {i}", BIOTECH, "2836", "BIOLOGICAL PRODUCTS", "Biotechnology", "35201010"))
        rows.append((f"Soft {i}", SOFTWARE, "7372", "SERVICES-PREPACKAGED SOFTWARE", "Software - Application", "45103010"))
        rows.append((f"Rig {i}", DRILLER, "1381", "DRILLING OIL & GAS WELLS", "Oil & Gas Drilling", "10101010"))
    return rows


def test_levels():
    assert level_of_difference("10101010", "10101010") == "same"
    assert level_of_difference("10101010", "10101020") == "sub_industry"
    assert level_of_difference("10101010", "10102010") == "industry"
    assert level_of_difference("10101010", "15101010") == "sector"


def test_train_and_score_roundtrip_without_definitions():
    structure = load_structure()  # bundled: names only, no definitions
    rows = _rows()
    texts = [company_text(Evidence(i, f"T{i}", n, d, s, sd, y)) for i, (n, d, s, sd, y, _) in enumerate(rows)]
    y = np.array([r[5] for r in rows])
    model = HybridModel(structure)
    model.fit_features(texts)
    X = model.features(texts)
    assert X.shape[1] > 163  # words plus one similarity column per sub-industry
    model.fit(X, y)
    proba, classes = model.predict_proba(model.features([company_text(Evidence(99, "NEW", "New Regional Bancorp", BANK, "6022", "STATE COMMERCIAL BANKS", "Banks - Regional"))]))
    assert classes[int(np.argmax(proba[0]))] == "40101015"
    cv = model.cross_validate(X, y, folds=4)
    assert cv["sub_industry"] == 1.0 and cv["n"] == len(rows)


def test_decide_policy():
    structure = load_structure()
    model = HybridModel(structure)
    model.label_counts.update({"40101015": 10, "40101010": 1})
    classes = ["40101010", "40101015", "35201010"]
    sure = np.array([0.05, 0.9, 0.05])
    prior = Placement("40101015", 0.9, "yahoo+sic", "prior")
    d = decide(model, sure, classes, prior)
    assert d.outcome == "agree" and d.confidence == 0.9 and d.method == "model+crosswalk"

    # Model prefers Diversified Banks but saw only one label for Regional: cannot override.
    other = np.array([0.85, 0.1, 0.05])
    d = decide(model, other, classes, Placement("40101015", 0.6, "yahoo", "prior"), min_class_labels=5)
    assert d.outcome == "override:sub_industry" and d.code == "40101010"
    # The reverse: the model prefers Regional over a Diversified prior, but it saw a
    # single Diversified label, so it has no standing to reject that class.
    d = decide(model, np.array([0.1, 0.85, 0.05]), classes, Placement("40101010", 0.6, "yahoo", "prior"), min_class_labels=5)
    assert d.outcome == "prior_kept" and d.code == "40101010"

    d = decide(model, np.array([0.3, 0.3, 0.4]), classes, None, min_confidence=0.5)
    assert d.outcome == "unplaced" and d.code is None
    d = decide(model, np.array([0.1, 0.1, 0.8]), classes, None)
    assert d.outcome == "placed_new" and d.code == "35201010"


def test_crosswalk_prior_feeds_decision():
    p = classify(Evidence(1, "X", "Example Bancorp", BANK, "6022", "STATE COMMERCIAL BANKS", "Banks - Regional"))
    assert p.code == "40101015" and p.confidence == 0.9
