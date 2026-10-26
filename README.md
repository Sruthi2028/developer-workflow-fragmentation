Beyond Commit Counts: Detecting Unusual Developer Workflow Fragmentation from Public GitHub Activity

<p align="center">
  <strong>An Exploratory Machine Learning Study of Repository-Switching Patterns</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/ML-Isolation%20Forest-orange" alt="Machine Learning">
  <img src="https://img.shields.io/badge/Data-GH%20Archive-green" alt="GH Archive">
  <img src="https://img.shields.io/badge/Status-Research%20Prototype-purple" alt="Research Prototype">
</p>

Overview

Developer activity is often summarized using event counts, commits, or the number of repositories involved. These measures do not fully describe the order in which activity occurs across repositories.

This project investigates whether repository-switching patterns provide additional information for identifying unusual public GitHub activity. It extracts activity-based and sequence-derived features from public GitHub events and compares them using the unsupervised Isolation Forest algorithm.

This is an exploratory software-repository-mining study. It examines how feature representations affect anomaly flags; it does not infer developers' productivity, mental state, or working conditions.

Research Question

Do sequence-derived repository-switching features change which public GitHub activity records are flagged as unusual compared with a conventional activity-based baseline?

Objectives

Examine public activity distributed across multiple repositories.

Measure repository transitions and short-horizon repository re-entry.

Compare activity-based, temporal-only, and combined feature configurations.

Investigate sensitivity to model parameters and feature definitions.

Examine overlap between records flagged by different configurations.

Methodology

The pipeline follows these stages:

Data collection: Use selected public GitHub event archives from GH Archive.

Preprocessing: Parse relevant event types, clean records, and deduplicate events according to the project’s preprocessing procedure.

Actor-week aggregation: Group observed events by actor and week.

Feature engineering: Calculate activity volume, repository diversity, repository transitions, transition intensity, and short-horizon re-entry.

Unsupervised modelling: Apply Isolation Forest to three feature configurations.

Comparison and sensitivity analysis: Compare flag counts, overlap between flagged sets, feature relationships, and parameter sensitivity.

Interpretation: Discuss results in light of sparse sampling and the absence of independently verified anomaly labels.

Feature Engineering

Feature

Description

Event count

Number of observed events associated with an actor-week.

Distinct repositories

Number of different repositories observed.

Normalized repository entropy

Measures how activity is distributed across repositories.

Repository transitions

Number of changes between consecutive observed repositories.

Transition intensity

Repository transitions divided by the observed time span, using a minimum one-hour denominator.

Short-horizon re-entry rate (SHRR)

Proportion of repository transitions that return to a previously observed repository within a specified time window.

These features describe observed public event sequences. They do not reveal why a repository switch occurred or whether it interrupted a developer’s work.

Dataset and Data Availability

Source and sampling window

The experiment used four selected GH Archive snapshots from October 1, 2026, corresponding to 00:00, 06:00, 12:00, and 18:00 UTC.

Source: GH Archive

Dataset summary

Dataset characteristic

Count

Raw event lines

329,310

Relevant events before deduplication and cleanup

312,125

Clean unique events

311,034

Distinct actors

108,666

Distinct repositories

158,960

Actor-week records before filtering

108,666

Actor-week records retained for the main comparison

31,500

The main model comparison retained actor-week records containing at least two observed events.

Event types

The preprocessing pipeline includes these GitHub event types:

PushEvent

PullRequestEvent

PullRequestReviewEvent

IssuesEvent

IssueCommentEvent

CreateEvent

Data files and access

The raw event data can be obtained from GH Archive. The scripts for processing and analysis are in the repository’s src/ directory. Consult data/README.md for any project-specific data instructions.

Availability note: The raw GH Archive data and complete processed real-world dataset are not currently included in this repository. The dataset counts above describe the data used in the reported experiment; they do not mean that all source or processed data files can be downloaded directly from this repository. Some local raw and processed data files may be excluded from version control because of their size.

Sampling caveats

Four selected snapshots from one date do not represent continuous activity across the full day or week.

The actor-week grouping is based on sparse sampled events, not a complete weekly activity record.

Automated accounts were not comprehensively filtered.

Results may depend on the sampled files, deduplication rules, feature definitions, and model configuration.

Experimental Results

Model comparison

The three Isolation Forest configurations were evaluated on the same 31,500 filtered actor-week records, using a nominal contamination setting of 0.05.

Configuration

Features

Flagged records

Flag rate

Activity baseline

Event count, distinct repositories, normalized repository entropy

1,555

4.9365%

Temporal-only

Repository transitions, transition intensity, SHRR

1,550

4.9206%

Combined

All six features

1,570

4.9841%

The flagged sets overlapped only partially. Their Jaccard similarities were:

Pair of configurations

Jaccard similarity

Activity baseline vs. temporal-only

33.55%

Activity baseline vs. combined

54.55%

Temporal-only vs. combined

66.49%

The similar flag rates but partial overlap indicate that feature selection changes which records are identified as unusual. These results do not establish that one configuration is more accurate, because independently verified labels are not available for the real-world records.

Contamination sensitivity

For the temporal-only Isolation Forest configuration, the reported sensitivity analysis was:

Contamination setting

Flagged records

Flag rate

Median anomaly score

95th-percentile score

1%

315

1.0000%

-0.3553

-0.1185

2%

601

1.9079%

-0.3250

-0.0882

5%

1,550

4.9206%

-0.2368

0.0000

10%

3,146

9.9873%

-0.1705

0.0663

Increasing the contamination setting increased the number of flagged records. The contamination setting controls the expected proportion of outliers and should not be interpreted as a measured real-world anomaly prevalence.

SHRR window sensitivity

The reported short-horizon re-entry rate sensitivity analysis was:

Re-entry window

Records with nonzero SHRR

Percentage of 108,666 records

Mean SHRR

15 minutes

569

0.5236%

0.001766

30 minutes

759

0.6985%

0.002739

60 minutes

910

0.8374%

0.003410

120 minutes

910

0.8374%

0.003410

The median and 95th-percentile SHRR were zero for each reported window. The 60-minute and 120-minute windows produced the same count of records with nonzero SHRR in this sample.

Synthetic benchmark

A separate synthetic benchmark was used to check model behaviour on constructed activity patterns. The benchmark contained 1,200 synthetic sequences, with a reported test set of 300 sequences and random seed 42.

Model

Accuracy

Precision

Recall

F1 score

ROC-AUC

Activity baseline

0.6767

0.6828

0.6600

0.6712

0.7664

Extended temporal model

1.0000

1.0000

1.0000

1.0000

1.0000

The synthetic labels were constructed from patterns the features were designed to measure. Perfect scores on this benchmark are therefore an implementation/control check, not evidence of perfect real-world anomaly detection. The saved confusion-matrix totals should be reconciled with the stated 300-sequence test set before those counts are reported.

Interpretation

The current results support a limited conclusion: activity-based and repository-sequence-based features can flag different public GitHub activity records as unusual, even when the overall flag rates are similar.

The experiment does not establish that flagged records represent workflow fragmentation, and it does not measure developer performance or well-being. Real-world precision, recall, F1, and ROC-AUC cannot be reported without independently verified labels.

Technologies Used

Python — data processing and feature engineering

pandas — event cleaning, aggregation, and analysis

NumPy — numerical computation

scikit-learn — Isolation Forest and evaluation

Matplotlib — experimental visualizations

GH Archive — public GitHub event data

LaTeX / IEEE format — research paper preparation

Git and GitHub — version control and project hosting

Repository Structure

The repository includes source code, reports, dataset instructions, and experiment outputs. The following is a high-level guide; consult the actual directory contents for the current files.

developer-workflow-fragmentation/
├── data/
│   ├── README.md
│   └── validation/
├── src/
│   ├── gharchive_pipeline.py
│   ├── train_real_data.py
│   ├── compare_real_data_models.py
│   ├── synthetic_benchmark.py
│   └── ...
├── reports/
│   ├── RESULTS_TEMPLATE.md
│   ├── synthetic_benchmark.csv
│   ├── synthetic_benchmark_run.txt
│   ├── test_run.txt
│   └── ...
├── tests/
├── progress_log.md
├── proposal_checkpoint.md
├── requirements.txt
├── .gitignore
└── README.md

Raw archives and large processed datasets may exist only in the local working environment and may be excluded from Git.

Reproducibility

To reproduce the analysis:

Clone the repository.

Create and activate a Python environment.

Install the dependencies listed in requirements.txt.

Obtain the required GH Archive snapshots for the documented timestamps.

Follow data/README.md and the relevant source-code instructions to prepare the data.

Run the preprocessing and feature-engineering pipeline.

Execute the model comparison, sensitivity analysis, and synthetic benchmark scripts as appropriate.

Compare generated outputs with the reported results.

Exact reproduction requires the same source files, preprocessing and deduplication rules, feature definitions, random seeds where applicable, and model settings. Do not assume that every locally generated data file is included in the public repository.

Limitations

The real-data analysis uses four selected snapshots from one date.

Public GitHub events provide only a partial view of development activity.

Automated accounts were not comprehensively filtered.

Some activity and transition features are correlated and may be redundant.

Real-data anomaly flags have no independently verified ground-truth labels.

Synthetic benchmark labels are constructed and do not validate real-world performance.

Repository switching alone cannot establish distraction, productivity, stress, burnout, or cognitive load.

The findings should therefore be interpreted as exploratory evidence about model behaviour on the observed sample.

Future Work

Collect a longer and more continuous observation window.

Establish and document a bot-filtering strategy.

Test feature ablations to measure the contribution of individual features.

Compare additional unsupervised anomaly-detection methods.

Evaluate flagged records using independent labels or a transparent human-review protocol.

Record exact dependencies, parameters, data provenance, and reproducibility instructions.

Reconcile synthetic benchmark test-set size and confusion-matrix outputs before reporting the matrix counts.

Research Paper

The IEEE-format paper and its figures can be included in a dedicated paper/ directory if they are prepared for release. If LaTeX source files are available, compile them using Overleaf or a local LaTeX distribution.

References

GH Archive, The public GitHub timeline.

F. T. Liu, K. M. Ting, and Z.-H. Zhou, “Isolation Forest,” Proceedings of the 8th IEEE International Conference on Data Mining, 2008, pp. 413–422. https://doi.org/10.1109/ICDM.2008.17.

E. Kalliamvakou et al., “The promises and perils of mining GitHub,” Proceedings of the 11th Working Conference on Mining Software Repositories, 2014, pp. 92–101. https://doi.org/10.1145/2597073.2597074.

M. M. Breunig, H.-P. Kriegel, R. T. Ng, and J. Sander, “LOF: Identifying density-based local outliers,” Proceedings of the ACM SIGMOD International Conference on Management of Data, 2000, pp. 93–104. https://doi.org/10.1145/342009.335388.

Author: K. Sruthi
Institution: VIT-AP University, Amaravati, India

This repository is an academic research project. Its findings describe exploratory anomaly detection on sampled public GitHub activity and should not be interpreted as validated measurements of individual developer behaviour.
