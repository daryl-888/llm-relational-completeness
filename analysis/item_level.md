# Item-level omissions and the independent-error baseline

In-context enumeration, E1: 495 valid answers, 15,700 correct items. Excluded as identifier failures: MiMo-V2.6 Flash enum-ctx-064.

## 1. Per-item miss rate (%) by answer size

|   n |   DeepSeek V4 Pro |   DeepSeek V4 Flash |   Claude Haiku 5.5 |   MiMo-V2.6 Pro |   MiMo-V2.6 Flash |
|----:|------------------:|--------------------:|-------------------:|----------------:|------------------:|
|   5 |              0    |                0.87 |               0    |            0    |              0    |
|  10 |              0    |                0    |               0    |            0.43 |              1.3  |
|  20 |              0    |                0    |               0.45 |            1.59 |              1.9  |
|  50 |              0.25 |                0.12 |               1.88 |            1.47 |              6    |
| 100 |              0.12 |                0    |               2.12 |            0.06 |             11.53 |

## Item-level logistic regression

Odds ratio that a given correct item is missed, per doubling, adjusted for the other size and for input format; 95% intervals from resampling whole answers. Models with fewer than 20 missed items are omitted.

| Model                            |   Missed items | OR: answer size   | OR: table size     |
|:---------------------------------|---------------:|:------------------|:-------------------|
| Claude Haiku 5.5                 |             51 | 1.67 [1.15, 2.96] | 3.38 [2.05, 7.95]  |
| MiMo-V2.6 Pro                    |             20 | 0.65 [0.41, 1.09] | 3.80 [1.82, 11.16] |
| MiMo-V2.6 Flash                  |            229 | 2.24 [1.58, 3.47] | 4.26 [2.81, 10.26] |
| All models (model fixed effects) |            306 | 1.69 [1.30, 2.10] | 3.71 [2.76, 5.83]  |

## 2. Answers with no omission (%): observed (baseline)

Baseline = (1 - q)^n, with q each model's per-item miss rate on answer sets of 10 or fewer items.

| Model             | n = 5     | n = 10    | n = 20    | n = 50   | n = 100   |
|:------------------|:----------|:----------|:----------|:---------|:----------|
| DeepSeek V4 Pro   | 100 (100) | 100 (100) | 100 (100) | 88 (100) | 88 (100)  |
| DeepSeek V4 Flash | 96 (99)   | 100 (97)  | 100 (94)  | 94 (86)  | 100 (75)  |
| Claude Haiku 5.5  | 100 (100) | 100 (100) | 91 (100)  | 50 (100) | 56 (100)  |
| MiMo-V2.6 Pro     | 100 (99)  | 96 (97)   | 77 (94)   | 80 (86)  | 94 (74)   |
| MiMo-V2.6 Flash   | 100 (96)  | 87 (92)   | 71 (84)   | 33 (65)  | 47 (42)   |

## 3. Dispersion of missed counts (variance / binomial variance)

1 = misses strike independently; much larger = they clump in some answers.

|   n |   DeepSeek V4 Pro |   DeepSeek V4 Flash |   Claude Haiku 5.5 |   MiMo-V2.6 Pro |   MiMo-V2.6 Flash |
|----:|------------------:|--------------------:|-------------------:|----------------:|------------------:|
|   5 |             nan   |                   1 |              nan   |           nan   |             nan   |
|  10 |             nan   |                 nan |              nan   |             1   |               0.9 |
|  20 |             nan   |                 nan |                1   |             1.3 |               1.2 |
|  50 |               0.9 |                   1 |                1.7 |             5   |               4.9 |
| 100 |               0.9 |                 nan |                5.4 |             1   |              18.6 |

In answers with 3 or more misses (22), the missed rows sit in the first, middle and last third of the table 17%, 34% and 49% of the time (33% each if spread evenly).

