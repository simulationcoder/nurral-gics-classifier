"""Place companies in the GICS structure from the evidence you already have.

Two classifiers that share one vocabulary (the GICS 2023 structure):

* :mod:`gics_classifier.crosswalk` -- rule-based: a Yahoo Finance industry and/or a
  SEC SIC code, refined by words in the company's name and description, to a GICS
  sub-industry. No training data needed.
* :mod:`gics_classifier.model` -- a supervised text model (TF-IDF over the company's
  text, plus its cosine similarity to each sub-industry definition when you have
  them, into a class-balanced logistic regression) trained on placements you trust,
  with a decision rule that confirms, overrides or defers to the crosswalk's prior.

GICS is a registered trademark of MSCI Inc. and S&P Global Market Intelligence. This
package ships the published structure (codes and names) and places companies in it
by its own means; it is not GICS Direct data and its placements are not S&P/MSCI's.
"""

from .crosswalk import Evidence, Placement, classify
from .structure import Structure, load_structure

__all__ = ["Evidence", "Placement", "classify", "Structure", "load_structure"]
__version__ = "0.1.0"
