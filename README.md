# Developer Workflow Fragmentation — ML Theory Assessment

## Project status
Starter implementation and proposal checkpoint created on 25 October 2026. **No real-data findings are claimed yet.** Run the pipeline and replace all report placeholders with measured results before submission.

## Research focus
**Working title:** Beyond Commit Counts: Detecting Unusual Developer Workflow Fragmentation from Public GitHub Activity

This project studies observable public GitHub event sequences, not private work, cognitive load, exhaustion, or burnout. Public events are incomplete proxies for developers' actual work.

### Research question
Do temporal repository-switching features provide useful information beyond basic contribution counts and repository entropy when identifying unusual activity patterns in public GitHub event data?

### Proposed contributions to test
1. Define a Short-Horizon Re-entry Rate (SHRR): the fraction of repository transitions that return to a repository observed recently within a chosen time window.
2. Combine repository entropy, transition intensity, and SHRR into an exploratory Workflow Fragmentation Score (WFS).
3. Compare a population-level anomaly detector with a per-developer historical baseline when enough history is available.

These are **candidate contributions**, not claims of global novelty. The literature review must be expanded before claiming originality.

## Data
The pipeline downloads hourly JSON archives from the OpenDigger GHArchive-compatible archive (`https://gharchive.open-digger.cn`). Its hourly archive is reported as available from 2026-09-06 onward. The original GH Archive documents the public-event dataset at `https://www.gharchive.org/`. Public-event coverage and completeness can vary; verify downloaded files and manifests and record missing hours. The archive does not record all developer actions, private repositories, local branch changes, or all task switches. The analysis therefore calls the measured behaviour *observed public-repository activity fragmentation*, not burnout or cognitive exhaustion.

## Setup
Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

## Run a small pilot
Download a short date range. One day (24 archives) is a pilot, not enough for reliable per-developer weekly baselines.

```bash
python src/gharchive_pipeline.py --start 2026-10-18 --end 2026-10-18 --out data/raw
```

Then extract features:

```bash
python src/gharchive_pipeline.py --features data/raw --out data/processed
```

Run a controlled synthetic sanity check (synthetic labels are not real-world ground truth):

```bash
python src/synthetic_benchmark.py --out reports/synthetic_benchmark.csv
```

Read `literature_review.md` for the initial focused literature review and its limitations.

Run anomaly detection on extracted actor-week rows:

```bash
python src/train_anomaly_model.py --input data/processed/actor_week_features.csv --out reports
```

The anomaly model is exploratory and unsupervised. It does not classify burnout. If there are too few actor-weeks, gather more dates or report that the sample is insufficient.

## Reproducibility checklist
- Record the exact date range and downloaded file count.
- Record missing/unavailable archives and parsing errors.
- Preserve the random seed and package versions.
- Do not mix the same actor-week across train/test splits if later building a supervised model.
- Avoid interpreting UTC timestamps as the developer's local time.
- Do not infer health status, work hours, or productivity from public events alone.
- Add actual model outputs, plots, and limitations to the IEEE paper.

## Outputs
- `actor_week_features.csv`: feature table for observed public events.
- `anomaly_scores.csv`: model scores and flags.
- `proposal_checkpoint.md`: initial proposal record.
- `progress_log.md`: dated development log; update it honestly.
- `IEEE_paper_draft.docx`: structured paper draft with no fabricated results.

## Network note
The assistant runtime may not have direct access to download archives. Run the download command from your own computer if it fails in this environment. Record failed or missing hours; do not silently treat incomplete archives as complete data.

## Current status (25 October 2026)
- Unit tests: 2 passed.
- Synthetic sanity check: completed; see `reports/synthetic_benchmark.csv` and `reports/RESULTS_TEMPLATE.md`.
- Real GHArchive download and real-data evaluation: not completed in the assistant runtime because outbound network access is unavailable. Run the download locally and record actual coverage.
