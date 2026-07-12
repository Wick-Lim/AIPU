# Paper — arXiv preprint (draft)

`aipu.tex` is a self-contained, dependency-light LaTeX preprint describing the
AIPU accelerator: a verification-first RTL design for local GGUF *k*-quant LLM
inference (bit-exact datapath, memory-bandwidth-bound residency design, formal
verification of the control plane).

**Honest scope:** the paper is deliberately *pre-silicon*. It reports design,
bit-exact/formal verification, an FPGA place-and-route, and cycle-accurate
emulation; every token rate is a roofline projection tagged `[EST]`. There is no
fabricated chip and no measured end-to-end token rate. The verification ledger
(Table 1) is reproducible from this repository.

## Build

No exotic packages are used (only `geometry`, `amsmath`, `booktabs`, `graphicx`,
`xcolor`, `enumitem`, `url`, `hyperref`). Any of:

```sh
pdflatex aipu.tex && pdflatex aipu.tex     # twice, for refs + hyperlinks
# or
latexmk -pdf aipu.tex
```

Or upload `aipu.tex` directly to Overleaf or to arXiv (single-file submission).

## Before posting to arXiv — checklist

- **Verify every reference's metadata** (authors, venue, year, arXiv IDs) against
  the primary sources. The bibliography lists well-known works; confirm the exact
  details before publication.
- **Confirm the GLM-5.2 model citation** (`\bibitem{glmweights}`) points to the
  exact weights/report you intend to cite.
- Re-read the *Limitations and Honest Scope* section against the current state of
  the repository so no claim drifts ahead of the evidence.
- Author/affiliation on the title page: `Wick-Lim, Independent Researcher`.

## Status

Draft. Content is synchronized with the repository's verification ledger
(`README.md`), the design-space docs (`docs/R3_APPLIANCE_SPEC.md`,
`docs/H_MEASUREMENT.md`, `docs/CYCLE_EMULATION.md`), the GGUF cross-check
(`docs/GGUF_CROSSCHECK.md`), and the board study (`hw/board_study/`).
