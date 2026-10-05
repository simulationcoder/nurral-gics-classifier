"""Hybrid classifier: a supervised text model whose features include the company's
similarity to every sub-industry definition.

Two feature blocks over one TF-IDF vocabulary fitted on the company texts *and* the
definitions (when you have them):

1. the company's own TF-IDF vector (word uni/bigrams, plus SIC code, Yahoo industry and
   size tokens from :func:`gics_classifier.text.company_text`);
2. the cosine similarity of that vector to each sub-industry's definition text, 163
   dense features that let the definitions shape the decision directly.

A class-balanced multinomial logistic regression on top. Train it on placements you
trust (we used the crosswalk rows where Yahoo and SIC independently agreed); the top
probability is the confidence.

:func:`decide` is the policy against a prior placement (typically the crosswalk's):
agreement raises confidence; an override needs the model to be sure, surer than the
prior, to give the prior's class little probability, and to have seen enough training
examples of the prior's class (a class it barely saw it cannot reject); otherwise the
prior stands and the dissent is kept in the rationale.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from .crosswalk import Placement
from .structure import Structure

MODEL_VERSION = "gics_classifier.model:v1"


def level_of_difference(a: str, b: str) -> str:
    if a == b:
        return "same"
    if a[:2] != b[:2]:
        return "sector"
    if a[:4] != b[:4]:
        return "industry_group"
    if a[:6] != b[:6]:
        return "industry"
    return "sub_industry"


class HybridModel:
    """TF-IDF over company text, plus cosine similarity to every definition, into a
    multinomial logistic regression. One object so it can be pickled whole."""

    def __init__(self, structure: Structure, C: float = 8.0, similarity_weight: float = 2.0):
        self.structure = structure
        self.codes = sorted(n.code for n in structure.sub_industries)
        self.C = C
        self.similarity_weight = similarity_weight
        self.vectoriser = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000,
                                          sublinear_tf=True, strip_accents="unicode")
        self.definition_matrix: sparse.csr_matrix | None = None
        self.model = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
        self.label_counts: Counter = Counter()

    # -- features ---------------------------------------------------------------
    def fit_features(self, company_texts: list[str]) -> None:
        definition_texts = [self.structure.definition_text(c) for c in self.codes]
        self.vectoriser.fit(list(company_texts) + definition_texts)
        self.definition_matrix = self.vectoriser.transform(definition_texts)

    def features(self, texts: list[str]) -> sparse.csr_matrix:
        x = self.vectoriser.transform(texts)
        sims = x @ self.definition_matrix.T  # cosine: both sides are L2-normalised
        return sparse.hstack([x, sims * self.similarity_weight]).tocsr()

    def similarities(self, texts: list[str]) -> np.ndarray:
        return (self.vectoriser.transform(texts) @ self.definition_matrix.T).toarray()

    # -- training ---------------------------------------------------------------
    def fit(self, X: sparse.csr_matrix, y: np.ndarray) -> None:
        self.label_counts = Counter(y)
        self.model.fit(X, y)

    def cross_validate(self, X: sparse.csr_matrix, y: np.ndarray, folds: int = 5) -> dict[str, float]:
        """Stratified k-fold accuracy at every level, over classes with >= folds labels."""
        counts = Counter(y)
        eligible = np.array([counts[c] >= folds for c in y])
        Xe, ye = X[eligible], y[eligible]
        hits: Counter = Counter()
        n = 0
        for train_idx, test_idx in StratifiedKFold(n_splits=folds, shuffle=True, random_state=7).split(Xe, ye):
            clf = LogisticRegression(C=self.C, max_iter=3000, class_weight="balanced").fit(Xe[train_idx], ye[train_idx])
            for a, b in zip(clf.predict(Xe[test_idx]), ye[test_idx]):
                n += 1
                lvl = level_of_difference(a, b)
                if lvl == "same":
                    hits["sub_industry"] += 1
                if lvl in ("same", "sub_industry"):
                    hits["industry"] += 1
                if lvl in ("same", "sub_industry", "industry"):
                    hits["industry_group"] += 1
                if lvl != "sector":
                    hits["sector"] += 1
        out = {k: hits[k] / max(1, n) for k in ("sub_industry", "industry", "industry_group", "sector")}
        out["n"] = n
        out["classes"] = len(set(ye))
        return out

    # -- scoring ----------------------------------------------------------------
    def predict_proba(self, X: sparse.csr_matrix) -> tuple[np.ndarray, list[str]]:
        return self.model.predict_proba(X), list(self.model.classes_)


@dataclass
class Decision:
    code: str | None
    confidence: float | None
    method: str          # model+crosswalk | model | <prior method> | none
    outcome: str         # agree | override:<level> | prior_kept | placed_new | unplaced
    rationale: str


def decide(
    model: HybridModel,
    proba: np.ndarray,
    classes: list[str],
    prior: Placement | None,
    *,
    nearest_definition: tuple[str, float] | None = None,
    override_threshold: float = 0.7,
    max_prior_prob: float = 0.2,
    min_class_labels: int = 5,
    min_confidence: float = 0.5,
) -> Decision:
    """The policy for one company: its probability row, the class list, and the prior."""
    order = np.argsort(-proba)
    top = [(classes[j], float(proba[j])) for j in order[:3]]
    m_code, m_prob = top[0]
    name = model.structure.name
    line = f"{MODEL_VERSION}: " + "; ".join(f"{name(c)} {p:.2f}" for c, p in top)
    if nearest_definition:
        line += f" | nearest definition: {name(nearest_definition[0])} cos={nearest_definition[1]:.2f}"
    class_index = {c: j for j, c in enumerate(classes)}

    if prior is not None and prior.code:
        if m_code == prior.code:
            return Decision(m_code, max(prior.confidence or 0.0, m_prob), "model+crosswalk", "agree",
                            f"{line} | prior: {prior.code} {name(prior.code)} ({prior.method}, {prior.confidence})")
        prior_prob = float(proba[class_index[prior.code]]) if prior.code in class_index else 0.0
        if (m_prob >= override_threshold and m_prob > (prior.confidence or 0.0)
                and model.label_counts.get(prior.code, 0) >= min_class_labels and prior_prob <= max_prior_prob):
            return Decision(m_code, m_prob, "model", f"override:{level_of_difference(m_code, prior.code)}",
                            f"{line} | prior: {prior.code} {name(prior.code)} ({prior.method}, {prior.confidence})")
        return Decision(prior.code, prior.confidence, prior.method, "prior_kept",
                        f"{prior.rationale} | model dissent kept: {line}")
    if m_prob >= min_confidence:
        return Decision(m_code, m_prob, "model", "placed_new", f"{line} | no prior")
    return Decision(None, None, "none", "unplaced", f"model top {name(m_code)} {m_prob:.2f} below {min_confidence}")
