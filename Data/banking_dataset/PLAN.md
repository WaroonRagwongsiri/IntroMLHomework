# Plan — Round 1 Presentation Deck (Canva)

Durable, repo-tracked copy of the plan executed to build the Round 1 Canva deck for
01076641 (ML Project for Case Analysis). Round 1 (Aug 31, 2026) covers problem setting,
feature selection/insight, and case-analysis/data-prep — no models yet (that's Round 2,
Oct 12, 2026).

## Context

Project state at time of writing: EDA only, split across `classify.ipynb` (target `y`)
and `regression.ipynb` (target `balance`), documented in `CLASSIFY_NOTES.md`,
`REGRESSION_NOTES.md`, `MATRIX_METHODS.md`, dataset provenance in `README.md`.

Decisions confirmed with the user:
- Deck scope: Round 1 only (data insights / problem-setting), not Round 2 model results.
- Build method: AI-generated Canva presentation via `generate-design`, not the blank
  linked "IntroML Banking Dataset" Canva design (`DAHTNCR56EE`), which is left untouched.
- Language: English.
- Title slide credit: two placeholder lines, `6801XXXX - Full Name` x2, filled in later.

## Plan

0. Persist this plan to `banking_dataset/PLAN.md` (this file).
1. Compose a `generate-design` query (`presentation`, `balanced` length, ~8-12 slides)
   covering: title, problem setting, dataset overview, feature selection/insight for
   classification (`y`) and regression (`balance`) — including justification for the
   `RELEVANCE_THRESHOLD = 0.1` cutoff (Cohen's 1988 "small effect" convention for
   correlation-scale statistics) — key EDA findings (leakage, seasonality, balance/y
   link), case analysis / data prep (drop `test.csv`, own seeded split, `unknown`
   handling, signed-log `balance` transform), and a brief Round 2 next-steps slide.
2. Generate the design, inspect candidates, `create-design-from-candidate` to make it
   editable.
3. Chart images: notebook matplotlib PNGs are local, not public URLs, so
   `upload-asset-from-url` can't embed them (and won't be used as an insecure
   public-hosting workaround). Deck represents numbers natively (tables/bullets/stat
   call-outs); user gets a list of chart cells to manually screenshot into Canva if
   wanted.
4. Review pass: check generated slide text against `CLASSIFY_NOTES.md` /
   `REGRESSION_NOTES.md` numbers; fix any drift via an editing transaction.

## Verification

- `create-design-from-candidate` returns a valid design ID, page count in 8-12 range.
- Slide numbers spot-checked against notes files.
- Final Canva edit URL reported to the user, plus the chart-cell list for manual embed.
