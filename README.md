# Beyond Commit Counts: Detecting Unusual Developer Workflow Fragmentation from Public GitHub Activity

<p align="center">
  <strong>An Exploratory Machine Learning Study of Repository-Switching Patterns</strong>
</p>

## Table of Contents

* [Overview](#overview)
* [Research Question](#research-question)
* [Objectives](#objectives)
* [Dataset and Data Sources](#dataset-and-data-sources)
* [Data Collection and Preprocessing](#data-collection-and-preprocessing)
* [Methodology](#methodology)
* [Feature Engineering](#feature-engineering)
* [Machine Learning Models](#machine-learning-models)
* [Experimental Results](#experimental-results)
* [Sensitivity Analysis](#sensitivity-analysis)
* [Synthetic Benchmark](#synthetic-benchmark)
* [Technologies Used](#technologies-used)
* [Repository Structure](#repository-structure)
* [Installation and Reproducibility](#installation-and-reproducibility)
* [Limitations](#limitations)
* [Future Work](#future-work)
* [References](#references)
* [Author](#author)

---

## Overview

Developer activity is often summarized using simple measures such as commit counts, event counts, or the number of repositories involved. However, these measures do not fully describe the sequence of activities across repositories.

This project investigates whether **repository-switching patterns provide additional information for identifying unusual public GitHub activity**.

Using public GitHub event data, the study extracts activity-based and sequence-derived features and compares their behaviour using the unsupervised **Isolation Forest** algorithm.

The research focuses on how different feature representations influence anomaly-detection results. It examines whether incorporating temporal and repository-switching information changes which activity records are flagged as unusual.

This is an exploratory software repository mining study. The detected anomalies are not validated measurements of developer productivity, distraction, stress, burnout, or cognitive load.

## Research Question

**Do sequence-derived repository-switching features change which public GitHub activity records are flagged as unusual compared with a conventional activity-based baseline?**

## Objectives

* Analyze public GitHub activity across multiple repositories.
* Measure repository transitions and short-horizon repository re-entry.
* Develop activity-based and temporal feature representations.
* Compare activity-only, temporal-only, and combined feature configurations.
* Investigate the sensitivity of anomaly flags to model parameters.
* Examine overlap between records flagged by different configurations.
* Evaluate model behaviour using a separate synthetic benchmark.

---

## Dataset and Data Sources

### Primary Data Source

The project uses public GitHub event data from **[GH Archive](https://www.gharchive.org/)**.

GH Archive collects public GitHub activity events and makes archived event data available for analysis.

### Sampling Window

The experiment used four selected GH Archive snapshots from **October 1, 2026**, corresponding to:

| Snapshot | Time (UTC) |
| -------- | ---------- |
| 1        | 00:00      |
| 2        | 06:00      |
| 3        | 12:00      |
| 4        | 18:00      |

These snapshots provide a sparse sample of public GitHub activity from one date.

### Dataset Statistics

The following statistics were recorded during data collection and preprocessing.

| Dataset characteristic                                    |   Count |
| --------------------------------------------------------- | ------: |
| Raw event lines                                           | 329,310 |
| Relevant events before deduplication and cleanup          | 312,125 |
| Clean unique events                                       | 311,034 |
| Distinct actors                                           | 108,666 |
| Distinct repositories                                     | 158,960 |
| Actor-week records before filtering                       | 108,666 |
| Actor-week records retained for the main model comparison |  31,500 |

The main model comparison retained actor-week records with at least two observed events.

### Event Types

The preprocessing pipeline includes the following GitHub event types:

* `PushEvent`
* `PullRequestEvent`
* `PullRequestReviewEvent`
* `IssuesEvent`
* `IssueCommentEvent`
* `CreateEvent`

### Dataset Components

| Component               | Description                                                                        |
| ----------------------- | ---------------------------------------------------------------------------------- |
| Raw event data          | Public GitHub events obtained from GH Archive                                      |
| Cleaned event data      | Relevant events after cleaning and deduplication                                   |
| Actor-week records      | Events grouped by actor and week                                                   |
| Engineered features     | Activity volume, repository diversity, transitions, transition intensity, and SHRR |
| Real-data model outputs | Anomaly scores, flags, and model-comparison results                                |
| Synthetic benchmark     | Constructed activity sequences used for controlled model evaluation                |

### Data Availability

The raw event archives can be obtained from [GH Archive](https://www.gharchive.org/).

The repository contains the processing scripts, experiment reports, and synthetic benchmark outputs. However, **the raw GH Archive files and complete processed real-world dataset are not currently included in the repository**.

The reported dataset statistics describe the data used in the experiment; they do not imply that every raw or processed dataset file can be downloaded directly from this repository.

Refer to [`data/README.md`](data/README.md) for any additional dataset instructions and available validation files.

---

## Data Collection and Preprocessing

The research pipeline processes public GitHub events through the following stages:

1. **Data collection:** Obtain the selected GH Archive snapshots.
2. **Event extraction:** Parse the relevant GitHub event types.
3. **Data cleaning:** Remove records that do not satisfy the preprocessing requirements.
4. **Deduplication:** Remove duplicate events according to the project's deduplication procedure.
5. **Actor-week aggregation:** Group the observed events by actor and week.
6. **Filtering:** Retain actor-week records containing at least two observed events for the main model comparison.
7. **Feature engineering:** Calculate activity-based and repository-switching features.
8. **Model evaluation:** Compare anomaly flags generated by the different feature configurations.

The dataset is based on sparse observations rather than a complete record of every developer action throughout an entire week.

---

## Methodology

The project compares three feature configurations using Isolation Forest.

### 1. Activity-Based Baseline

The baseline uses conventional activity and repository-diversity features:

* Event count
* Distinct repositories
* Normalized repository entropy

This configuration establishes a reference for comparing the additional information represented by repository-switching features.

### 2. Temporal-Only Model

The temporal configuration uses:

* Repository transitions
* Transition intensity
* Short-horizon re-entry rate

It focuses on the observed order and timing of repository activity.

### 3. Combined Model

The combined configuration uses all six features from the activity-based and temporal representations.

This allows the experiment to examine whether combining activity volume, repository diversity, and sequence-derived features changes the resulting anomaly flags.

---

## Feature Engineering

Six features were used in the main real-data model comparison.

| Feature                            | Description                                                                                                         |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| Event count                        | Number of observed events associated with an actor-week                                                             |
| Distinct repositories              | Number of different repositories observed                                                                           |
| Normalized repository entropy      | Measures how activity is distributed across repositories                                                            |
| Repository transitions             | Number of changes between consecutive observed repositories                                                         |
| Transition intensity               | Repository transitions divided by the observed time span, using a minimum one-hour denominator                      |
| Short-horizon re-entry rate (SHRR) | Proportion of repository transitions that return to a previously observed repository within a specified time window |

### Short-Horizon Re-Entry Rate (SHRR)

SHRR measures the extent to which an observed repository sequence returns to a previously visited repository within a specified time window.

The default window used in the main feature configuration is **60 minutes**.

A zero SHRR value indicates that no qualifying re-entry was recorded under the feature definition for that sequence. It does not necessarily mean that the developer did not return to a repository outside the observed sample.

---

## Machine Learning Models

### Isolation Forest

The project uses **Isolation Forest**, an unsupervised anomaly-detection algorithm.

Isolation Forest identifies observations that can be isolated relatively easily in the feature space. It produces anomaly scores and labels observations as inliers or outliers according to the selected decision threshold.

The algorithm does not require ground-truth anomaly labels during training.

### Model Configurations

| Configuration     | Number of features | Feature groups                                        |
| ----------------- | -----------------: | ----------------------------------------------------- |
| Activity baseline |                  3 | Activity and repository diversity                     |
| Temporal-only     |                  3 | Repository transitions and re-entry                   |
| Combined          |                  6 | Activity, repository diversity, and temporal features |

For the reported main comparison, all three configurations used a nominal contamination setting of `0.05`.

The contamination parameter controls the expected proportion of outliers identified by the model. It is a model setting, not a measured estimate of the true prevalence of unusual developer workflows.

---

## Experimental Results

### Main Model Comparison

All three configurations were evaluated on the same **31,500 filtered actor-week records**.

| Configuration     | Features                                                          | Flagged records | Flag rate |
| ----------------- | ----------------------------------------------------------------- | --------------: | --------: |
| Activity baseline | Event count, distinct repositories, normalized repository entropy |           1,555 |   4.9365% |
| Temporal-only     | Repository transitions, transition intensity, SHRR                |           1,550 |   4.9206% |
| Combined          | All six features                                                  |           1,570 |   4.9841% |

The overall flag rates were similar, but the models did not identify exactly the same records.

### Overlap Between Flagged Records

Jaccard similarity was used to measure the overlap between the flagged sets.

| Model pair                          | Jaccard similarity |
| ----------------------------------- | -----------------: |
| Activity baseline vs. temporal-only |             33.55% |
| Activity baseline vs. combined      |             54.55% |
| Temporal-only vs. combined          |             66.49% |

The results indicate that **feature selection changes which records Isolation Forest identifies as unusual**.

The lower overlap between the activity baseline and temporal-only model suggests that the two configurations capture different aspects of the observed event data.

However, the overlap results do not establish that one model is more accurate than another.

Because the real-world records do not have independently verified anomaly labels, real-data precision, recall, F1 score, and ROC-AUC cannot be established from these experiments.

---

## Sensitivity Analysis

The experiment examined the effect of different contamination settings on the temporal-only Isolation Forest configuration.

### Contamination Sensitivity

| Contamination | Flagged records | Flag rate | Median anomaly score | 95th-percentile score |
| ------------: | --------------: | --------: | -------------------: | --------------------: |
|            1% |             315 |   1.0000% |              -0.3553 |               -0.1185 |
|            2% |             601 |   1.9079% |              -0.3250 |               -0.0882 |
|            5% |           1,550 |   4.9206% |              -0.2368 |                0.0000 |
|           10% |           3,146 |   9.9873% |              -0.1705 |                0.0663 |

As the contamination setting increased, the number of flagged records increased.

This illustrates the influence of the model's decision setting on the number of records classified as outliers.

### SHRR Window Sensitivity

SHRR was also examined using four different re-entry windows.

| Re-entry window | Records with nonzero SHRR | Percentage of 108,666 records | Mean SHRR |
| --------------- | ------------------------: | ----------------------------: | --------: |
| 15 minutes      |                       569 |                       0.5236% |  0.001766 |
| 30 minutes      |                       759 |                       0.6985% |  0.002739 |
| 60 minutes      |                       910 |                       0.8374% |  0.003410 |
| 120 minutes     |                       910 |                       0.8374% |  0.003410 |

The median and 95th-percentile SHRR were zero for all four windows.

The 60-minute and 120-minute windows produced the same number of records with nonzero SHRR in the reported sample.

These results describe the sampled event data and should not be interpreted as general estimates of developer behaviour.

---

## Synthetic Benchmark

A separate synthetic benchmark was used to test model behaviour on constructed activity sequences.

### Benchmark Setup

| Parameter                 | Value |
| ------------------------- | ----- |
| Total synthetic sequences | 1,200 |
| Reported test sequences   | 300   |
| Random seed               | 42    |

The synthetic labels were constructed from activity patterns that the features were designed to represent.

### Reported Results

| Model                   | Accuracy | Precision | Recall | F1 score | ROC-AUC |
| ----------------------- | -------: | --------: | -----: | -------: | ------: |
| Activity baseline       |   0.6767 |    0.6828 | 0.6600 |   0.6712 |  0.7664 |
| Extended temporal model |   1.0000 |    1.0000 | 1.0000 |   1.0000 |  1.0000 |

The extended model achieved perfect reported scores on the synthetic benchmark.

These results should be interpreted cautiously because the synthetic labels were constructed from patterns the features were designed to measure. The benchmark therefore serves as an implementation and controlled-behaviour check, not proof of perfect real-world anomaly detection.

The saved confusion-matrix totals should also be reconciled with the stated 300-sequence test set before the exact matrix counts are reported.

---

## Technologies Used

| Technology          | Purpose                                  |
| ------------------- | ---------------------------------------- |
| Python              | Data processing and feature engineering  |
| pandas              | Data cleaning, aggregation, and analysis |
| NumPy               | Numerical computation                    |
| scikit-learn        | Isolation Forest and evaluation          |
| Matplotlib          | Experimental visualizations              |
| GH Archive          | Public GitHub event data                 |
| Git                 | Version control                          |
| GitHub              | Source-code hosting and collaboration    |
| LaTeX / IEEE format | Research paper preparation               |

---

## Repository Structure

The repository includes source code, experiment reports, dataset instructions, tests, and benchmark outputs.

```text
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
```

This is a high-level overview. The exact files and subdirectories should be checked against the current repository contents.

Raw event archives and large processed datasets may be stored locally and excluded from version control.

---

## Installation and Reproducibility

### Prerequisites

* Python 3.x
* Git
* The dependencies listed in `requirements.txt`
* The required GH Archive snapshots for the documented sampling window

### Setup

Clone the repository:

```bash
git clone https://github.com/Sruthi2028/developer-workflow-fragmentation.git
cd developer-workflow-fragmentation
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

### Reproducing the Analysis

1. Obtain the required public event archives from [GH Archive](https://www.gharchive.org/).
2. Follow the instructions in `data/README.md`.
3. Prepare the event data using the repository's preprocessing pipeline.
4. Generate the actor-week feature dataset.
5. Run the relevant real-data model comparison and sensitivity scripts.
6. Run the synthetic benchmark separately.
7. Compare the generated outputs with the reported results.

Relevant scripts include:

* `src/gharchive_pipeline.py`
* `src/train_real_data.py`
* `src/compare_real_data_models.py`
* `src/synthetic_benchmark.py`

Check each script's implementation and argument requirements before executing it. The commands and file paths may depend on the local dataset layout.

Exact reproduction requires the same source data, preprocessing rules, feature definitions, and model settings.

---

## Limitations

The study has several important limitations:

1. **Sparse sampling:** The real-data analysis uses four selected snapshots from a single date rather than continuous activity throughout an entire week.
2. **Partial observability:** Public GitHub events do not capture all development work or activity on private repositories.
3. **Automated accounts:** Bots were not comprehensively filtered from the dataset.
4. **Feature redundancy:** Some activity and transition features are correlated.
5. **No real-data ground truth:** The real-world anomaly flags do not have independently verified labels.
6. **Synthetic evaluation:** Synthetic benchmark results do not establish real-world predictive performance.
7. **Interpretation limits:** Repository switching alone cannot establish distraction, productivity, stress, burnout, or cognitive load.

The results should therefore be interpreted as exploratory evidence about the behaviour of anomaly-detection models on the sampled public event data.

---

## Future Work

Potential extensions include:

* Collecting a longer and more continuous observation window.
* Developing and documenting a reproducible bot-filtering strategy.
* Testing feature ablations to quantify the contribution of individual features.
* Comparing additional unsupervised anomaly-detection algorithms.
* Evaluating flagged records using independent labels or a transparent human-review protocol.
* Improving dataset documentation and data provenance.
* Recording exact dependencies, parameters, and execution commands.
* Reconciling synthetic benchmark test-set and confusion-matrix outputs before reporting exact matrix counts.

---

## References

1. GH Archive. *The public GitHub timeline*. https://www.gharchive.org/

2. F. T. Liu, K. M. Ting, and Z.-H. Zhou. “Isolation Forest.” *Proceedings of the 8th IEEE International Conference on Data Mining*, 2008, pp. 413–422. https://doi.org/10.1109/ICDM.2008.17

3. E. Kalliamvakou et al. “The promises and perils of mining GitHub.” *Proceedings of the 11th Working Conference on Mining Software Repositories*, 2014, pp. 92–101. https://doi.org/10.1145/2597073.2597074

4. M. M. Breunig, H.-P. Kriegel, R. T. Ng, and J. Sander. “LOF: Identifying density-based local outliers.” *Proceedings of the ACM SIGMOD International Conference on Management of Data*, 2000, pp. 93–104. https://doi.org/10.1145/342009.335388

---

## Author

**K. Sruthi**
VIT-AP University
Amaravati, India

---

*This repository is an academic research project. Its findings describe exploratory anomaly detection on sampled public GitHub activity and should not be interpreted as validated measurements of individual developer behaviour.*
