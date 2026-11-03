# Power of the fully crossed answer-size experiment

Design: answer size g in [5, 10, 20, 50, 100, 200, 500] x table size N in [1000, 2000, 4000] (21 cells), one model, one prompt. 300 simulated experiments per setting. base = per-item miss rate at g = 50, N = 1000; sigma = spread of answer-level difficulty (logit scale).

|   base |   sigma |   instances per cell |   answers | false positives at |z| > 1.96   | power, OR 1.2 per doubling   | power, OR 1.5 per doubling   |   dispersion at g = 100 |
|-------:|--------:|---------------------:|----------:|:--------------------------------|:-----------------------------|:-----------------------------|------------------------:|
|   0.01 |     0.5 |                    3 |        63 | 8%                              | 65%                          | 100%                         |                     2.2 |
|   0.01 |     0.5 |                    5 |       105 | 7%                              | 89%                          | 100%                         |                     2.3 |
|   0.01 |     0.5 |                   10 |       210 | 9%                              | 100%                         | 100%                         |                     2.3 |
|   0.01 |     1   |                    3 |        63 | 4%                              | 53%                          | 100%                         |                     4   |
|   0.01 |     1   |                    5 |       105 | 8%                              | 68%                          | 100%                         |                     4.7 |
|   0.01 |     1   |                   10 |       210 | 8%                              | 94%                          | 100%                         |                     5.3 |
|   0.02 |     0.5 |                    3 |        63 | 9%                              | 74%                          | 100%                         |                     3.1 |
|   0.02 |     0.5 |                    5 |       105 | 7%                              | 97%                          | 100%                         |                     3.5 |
|   0.02 |     0.5 |                   10 |       210 | 7%                              | 100%                         | 100%                         |                     3.6 |
|   0.02 |     1   |                    3 |        63 | 5%                              | 58%                          | 100%                         |                     6.2 |
|   0.02 |     1   |                    5 |       105 | 8%                              | 77%                          | 100%                         |                     7   |
|   0.02 |     1   |                   10 |       210 | 8%                              | 97%                          | 100%                         |                     9   |

Power is size-adjusted (critical value from the simulated null). For comparison, item_level.md estimates the pooled odds ratio for answer size at 1.50 (beta-binomial) to 1.69 (cluster bootstrap), and observes dispersion of 1-19x at g = 100.

