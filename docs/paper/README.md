# docs/paper — ESWA submission files

| File | Purpose |
|---|---|
| `eswa_paper.tex` | **The submission manuscript** — Elsevier `elsarticle` class, ESWA structure: Introduction / Literature review / Methodology / Results and discussion / Conclusion, with a TikZ architecture figure. |
| `plots/width_vs_budget.png` | Certification cost frontier figure (generated from `main_results/paper_wine_width_vs_budget.csv` + probe CSVs). |
| `highlights.txt` | ESWA Highlights (submitted separately, ≤85 chars/bullet). |
| `response_letter_skeleton.md` | Pre-emptive answers to the 3 predictable reviewer attacks. |
| `main.tex` | Earlier working draft (two-column `article` class) — kept for reference. |
| `paper_outline.md` | Paper outline. |

## How to compile `eswa_paper.tex`

`elsarticle.cls` cannot be downloaded inside this sandbox (CTAN is unreachable
here), but it is the standard Elsevier class:

- **Easiest:** Overleaf → New Project → *Elsevier Journals* template (it ships
  `elsarticle.cls`) → replace the body with this file → compile.
- **Locally:** `tlmgr install elsarticle` (TeX Live) or fetch
  `elsarticle.cls` from CTAN, then:
  ```bash
  cd docs/paper
  pdflatex eswa_paper.tex && pdflatex eswa_paper.tex
  ```

The TikZ figure (`fig:arch`) needs no external files — it is drawn inline.
`plots/width_vs_budget.png` is referenced by `fig:frontier` and is committed
in this folder.

## Before submitting to ESWA (author-only items)

1. Fill the `\affiliation[aff1]{organization=..., city=..., country=...}` field.
2. Optionally add co-authors (multiple `\author` + `\affiliation` entries).
3. Prepare the ESWA **graphical abstract** (Elsevier requires one at submission).
4. Check the journal's reference style (numbered; `elsarticle-num.bst` if you
   switch from the manual `thebibliography`).
