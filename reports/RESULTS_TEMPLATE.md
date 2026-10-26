# Results Record

## A. Controlled Synthetic Sanity Check — Completed

* **Synthetic actor-week sequences:** 1,200
* **Held-out test sequences:** 300
* **Random seed:** 42
* **Labels:** Constructed from synthetic low-/high-fragmentation patterns; not real developer labels.

| Feature set                                                         | Accuracy | Precision | Recall |     F1 | ROC-AUC |
| ------------------------------------------------------------------- | -------: | --------: | -----: | -----: | ------: |
| Event count + distinct repositories + normalized repository entropy |   0.6767 |    0.6828 | 0.6600 | 0.6712 |  0.7664 |
| Baseline + repository transitions + transition intensity + SHRR     |   1.0000 |    1.0000 | 1.0000 | 1.0000 |  1.0000 |

**Interpretation:** This is a controlled implementation check. Since the synthetic labels were constructed to reflect differences in temporal switching patterns, perfect performance is not evidence that the method detects real-world fragmentation. These results must not be used to claim burnout detection or real-world generalization.

## B. Real Public-Event Archive — Completed

* **Data source:** Official GitHub Archive, `https://www.gharchive.org/`
* **Date sampled:** October 1, 2026 (UTC).
* **Hourly archives expected:** 4.
* **Hourly archives downloaded and validated:** 4.
* **Hourly archives parsed:** 4.
* **Raw archive lines/events:** 329,310.
* **Relevant extracted events before deduplication:** 312,125.
* **Unique cleaned event records after deduplication:** 311,034.
* **Distinct actors in cleaned events:** 108,666.
* **Distinct repositories in cleaned events:** 158,960.
* **Actor-week rows before filtering:** 108,666.
* **Actor-week rows after filtering:** 31,500.
* **Filtering rule:** Retain actor-week records with at least two events (`event_count >= 2`) so repository transitions can be analyzed.
* **Missing files:** 0 of 4 expected archives.
* **Archive read failures:** 0.
* **Invalid JSON lines:** 0.
* **Missing values in full actor-week feature dataset:** 0.
* **Missing values in filtered feature dataset:** 0.
* **Bot filtering:** No explicit bot filtering was performed. Results may include automated accounts.
* **Observation limitation:** The data covers four sampled UTC hours on a single day, not continuous activity throughout a full week. The actor-week feature name reflects the grouping scheme, not a complete weekly observation.

**Interpretation:** The four official archives were successfully validated and processed. The cleaned dataset contains 311,034 unique relevant event records. Applying the minimum-event filter produced 31,500 actor-week records for the main anomaly analysis. The results describe activity observed in the sampled archive hours and should not be generalized to complete weekly behavior.

## C. Real-Data Descriptive and Anomaly Analysis — Completed

### C1. SHRR Window Sensitivity

Short-horizon re-entry rate (SHRR) was recalculated using four time windows.

| Re-entry window | Actor-week rows | Rows with nonzero SHRR | Percentage with nonzero SHRR | Mean SHRR | Median SHRR |
| --------------- | --------------: | ---------------------: | ---------------------------: | --------: | ----------: |
| 15 minutes      |         108,666 |                    569 |                       0.524% |  0.001766 |           0 |
| 30 minutes      |         108,666 |                    759 |                       0.698% |  0.002739 |           0 |
| 60 minutes      |         108,666 |                    910 |                       0.837% |  0.003410 |           0 |
| 120 minutes     |         108,666 |                    910 |                       0.837% |  0.003410 |           0 |

**Interpretation:** The number of actor-week records with nonzero SHRR increases as the window expands from 15 to 60 minutes. The 120-minute window identifies no additional records with nonzero SHRR compared with the 60-minute window in this sample. The median remains zero at every setting, indicating that nonzero re-entry rates are uncommon. These results do not establish 60 minutes as a universally optimal window.

### C2. Baseline and Extended Anomaly Models

Both models used Isolation Forest with a contamination parameter of 0.05 on the same 31,500 filtered actor-week records.

| Metric                        | Baseline Isolation Forest | Extended Isolation Forest |
| ----------------------------- | ------------------------: | ------------------------: |
| Number of features            |                         3 |                         6 |
| Records analyzed              |                    31,500 |                    31,500 |
| Anomalies flagged             |                     1,555 |                     1,570 |
| Flagged fraction              |                    4.937% |                    4.984% |
| Mean anomaly score            |                   -0.1754 |                   -0.1948 |
| Median anomaly score          |                   -0.2108 |                   -0.2364 |
| 95th-percentile anomaly score |                    0.0000 |                    0.0000 |

**Baseline features:**

* Event count
* Distinct repositories
* Normalized repository entropy

**Extended features:**

* Event count
* Distinct repositories
* Normalized repository entropy
* Repository transitions
* Transitions per observed span hour
* Short-horizon re-entry rate (SHRR)

**Interpretation:** The extended model flagged 15 more records than the baseline. This difference indicates that the models do not produce identical anomaly sets; it does not establish that the extended model is more accurate.

### C3. Anomaly-Flag Overlap

| Model comparison           | Shared flagged records | Jaccard overlap |
| -------------------------- | ---------------------: | --------------: |
| Baseline vs. temporal-only |                    780 |          33.55% |
| Baseline vs. combined      |                  1,103 |          54.55% |
| Temporal-only vs. combined |                  1,246 |          66.49% |

**Interpretation:** The overlap results indicate that feature selection affects which records are flagged. The combined model shares more flagged records with the temporal-only model than with the baseline model in this experiment. Without independent ground-truth labels, these differences cannot determine which model is better at identifying real workflow fragmentation.

### C4. Contamination Sensitivity

The temporal-only Isolation Forest was evaluated at four contamination settings.

| Contamination setting | Records analyzed | Flagged records | Actual flagged fraction | Median anomaly score |
| --------------------- | ---------------: | --------------: | ----------------------: | -------------------: |
| 1%                    |           31,500 |             315 |                  1.000% |              -0.3553 |
| 2%                    |           31,500 |             601 |                  1.908% |              -0.3250 |
| 5%                    |           31,500 |           1,550 |                  4.921% |              -0.2368 |
| 10%                   |           31,500 |           3,146 |                  9.987% |              -0.1705 |

The flagged sets were nested across the tested settings: all records flagged at a lower contamination setting were retained among those flagged at each higher setting.

**Interpretation:** Increasing the contamination setting increases the number of records classified as anomalous. The nesting observed in this experiment suggests that higher settings expand the flagged set rather than replacing the lower-setting set. This is evidence about model behavior under the tested settings, not about detection accuracy.

### C5. Limitations

1. **No real-world ground-truth labels:** No independently verified labels identify genuinely fragmented workflows. Precision, recall, F1, ROC-AUC, and classification accuracy must not be reported for the real-data anomaly task.
2. **Limited temporal coverage:** Only four hourly archives from one UTC date were analyzed. These observations do not represent a complete week of developer activity.
3. **No explicit bot filtering:** Automated accounts may be present.
4. **Minimum-event filtering:** Records with fewer than two events were excluded from the main anomaly analysis because they cannot provide repository transitions. This limits the analyzed population.
5. **Sparse SHRR:** At least 99% of actor-week records had zero SHRR in the window sensitivity analysis. This feature is therefore zero for most observations in this sample.
6. **Synthetic benchmark limitations:** Synthetic labels were constructed from the patterns being evaluated. Perfect synthetic performance is not evidence of real-world generalization.
7. **Exploratory anomaly scores:** Isolation Forest flags unusual observations relative to the feature distribution. They are not probabilities of fragmentation, productivity, burnout, stress, or harm.
8. **Exploratory WFS:** The exploratory workflow-fragmentation score is a descriptive composite and has not been independently validated.
9. **Observational limitations:** Public GitHub activity is an incomplete proxy for developer work. Activity may occur in private repositories, local environments, code review tools, or other systems that are not represented in these archives.

## D. Overall Conclusion

The project implements a reproducible pipeline for extracting repository-switching features from public GitHub Archive events and comparing conventional activity features with temporal workflow features using Isolation Forest.

The real-data experiments show that feature selection and contamination settings affect anomaly flags. SHRR window sensitivity also shows that nonzero re-entry rates are uncommon in the sampled records. The synthetic benchmark demonstrates that the implementation can distinguish deliberately constructed patterns, but it does not establish real-world detection performance.

**The findings should therefore be presented as an exploratory analysis of unusual public GitHub activity patterns, not as validated detection of developer workflow fragmentation or any psychological or productivity-related condition.**
