# Proposal Checkpoint — 25 October 2026

## Working title
Beyond Commit Counts: Detecting Unusual Developer Workflow Fragmentation from Public GitHub Activity

## Problem
Contribution counts summarize activity volume, but do not describe the temporal sequence of activity across repositories. Public event records may show how an actor's observed activity is distributed across repositories and whether repository transitions recur within short windows.

## Target users
Researchers studying software engineering activity and maintainers interested in aggregate, privacy-conscious workflow analytics. This prototype is not intended for employee surveillance, performance ranking, or mental-health assessment.

## Motivation
A developer with contributions across multiple repositories may have a different activity pattern from a developer who repeatedly alternates between a small set of repositories. A simple count or entropy measure may not capture this temporal distinction. This project tests that possibility.

## Research question
Do temporal repository-switching features provide useful information beyond basic contribution counts and repository entropy when identifying unusual activity patterns in public GitHub event data?

## Candidate innovation
1. Short-Horizon Re-entry Rate (SHRR): proportion of repository transitions that return to a repository seen within a chosen recent time window.
2. Exploratory Workflow Fragmentation Score (WFS): a normalized combination of repository entropy, transition intensity, and SHRR.
3. Personal-baseline comparison: compare an actor's current observed pattern with their own earlier activity, only where sufficient history exists.

## Hypotheses to test
- H1: SHRR captures temporal information not represented by repository entropy alone.
- H2: Personal baselines identify deviations that a single global threshold misses.
- H3: Findings vary with the time-window definition and event coverage.

These are hypotheses, not conclusions.

## Dataset plan
Use the OpenDigger GHArchive-compatible hourly public event archives for the selected date range (available from 2026-09-06 onward), and document GH Archive as the underlying public-event source. Filter to contribution-related event types. Keep actor, repository, timestamp, and event type. Do not collect or infer health information. Record source, date range, archive availability, filtering, and exclusions.

## ML approach
Isolation Forest as an exploratory unsupervised anomaly detector on actor-week feature vectors. Compare its flagged observations with simple baselines (e.g., high switch-intensity percentile and personal historical deviation) and inspect stability across settings. Because real anomaly labels are unavailable, do not report supervised classification accuracy on real events.

## Evaluation plan
- Data quality: archive coverage, parse failures, event counts, actors, and actor-weeks.
- Metric robustness: vary the re-entry window (15, 30, 60, 120 minutes).
- Synthetic sanity check: test whether SHRR responds as expected to controlled event sequences; report this separately from real-data results.
- Anomaly detection: score distributions, top flagged actor-weeks, stability across seeds/parameters, and comparison with simple baselines.
- Limitations: event stream incompleteness, bots, UTC timestamps, public-only activity, sparse observations, no ground-truth context-switch or burnout labels.

## Originality status
Novelty is **not established yet**. Related research already studies developer task switches and work rhythms. The final claim must be narrowed after a literature review and based on measured results.

## AI-use / integrity note
AI assistance may have been used to organize the project plan and draft code. The student must follow the course's AI policy, verify all claims and citations, run and understand the code, and disclose assistance if required. Do not backdate this checkpoint.
