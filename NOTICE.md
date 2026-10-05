# Notice

**GICS** (Global Industry Classification Standard) is a registered trademark of
MSCI Inc. and S&P Global Market Intelligence, who own and license the standard.

This repository:

- ships the **structure** of GICS as published by its owners (sector, industry group,
  industry and sub-industry codes and names, effective after close of business on
  17 March 2023), which is the vocabulary the classifiers place companies into;
- does **not** ship the sub-industry **definitions**, which are the owners'
  copyrighted text. `gics-classify parse-workbook` extracts them from a copy of the
  official workbook you obtain yourself, into a local file this repository ignores;
- produces **its own** placements by rules and a statistical model. They are not
  GICS Direct data, not S&P's or MSCI's assignments, and may differ from them. Do not
  present them as official GICS classifications.

Yahoo Finance industry names and SEC SIC codes are used as input evidence; their
respective owners' terms apply to obtaining that data.
