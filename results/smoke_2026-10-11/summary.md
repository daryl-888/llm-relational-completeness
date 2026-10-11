## Overall (set and group answers)

|                                             |   n |   recall |   precision |   exact |
|:--------------------------------------------|----:|---------:|------------:|--------:|
| ('claude-cli/claude-haiku-4-5', 'direct')   |  19 |    0.957 |       0.992 |   0.789 |
| ('claude-cli/claude-haiku-5-5', 'direct')   |  19 |    0.996 |       1     |   0.947 |
| ('opencode-go/deepseek-v4-flash', 'direct') |  19 |    1     |       1     |   1     |
| ('opencode-go/deepseek-v4-pro', 'direct')   |  19 |    1     |       1     |   1     |
| ('opencode-go/mimo-v2.6-flash', 'direct')   |  19 |    0.984 |       0.975 |   0.842 |
| ('opencode-go/mimo-v2.6-pro', 'direct')     |  19 |    0.982 |       0.998 |   0.842 |


## Recall by slice (direct prompting)

| slice                    |   claude-cli/claude-haiku-4-5 |   claude-cli/claude-haiku-5-5 |   opencode-go/deepseek-v4-flash |   opencode-go/deepseek-v4-pro |   opencode-go/mimo-v2.6-flash |   opencode-go/mimo-v2.6-pro |
|:-------------------------|------------------------------:|------------------------------:|--------------------------------:|------------------------------:|------------------------------:|----------------------------:|
| aggregation              |                         0.875 |                         1     |                               1 |                             1 |                         0.75  |                       1     |
| enumeration              |                         0.925 |                         1     |                               1 |                             1 |                         1     |                       1     |
| enumeration (parametric) |                         1     |                         1     |                               1 |                             1 |                         1     |                       1     |
| enumeration (rag)        |                         1     |                         1     |                               1 |                             1 |                         1     |                       1     |
| join                     |                         0.969 |                         1     |                               1 |                             1 |                         0.979 |                       1     |
| nested                   |                         1     |                         1     |                               1 |                             1 |                         1     |                       0.75  |
| projection               |                         1     |                         1     |                               1 |                             1 |                         1     |                       0.909 |
| selection                |                         0.897 |                         0.974 |                               1 |                             1 |                         1     |                       1     |
| setop                    |                         1     |                         1     |                               1 |                             1 |                         1     |                       1     |


## Exact-set accuracy by slice (direct prompting)

| slice                    |   claude-cli/claude-haiku-4-5 |   claude-cli/claude-haiku-5-5 |   opencode-go/deepseek-v4-flash |   opencode-go/deepseek-v4-pro |   opencode-go/mimo-v2.6-flash |   opencode-go/mimo-v2.6-pro |
|:-------------------------|------------------------------:|------------------------------:|--------------------------------:|------------------------------:|------------------------------:|----------------------------:|
| aggregation              |                         0     |                         1     |                               1 |                             1 |                         0     |                       1     |
| enumeration              |                         0.75  |                         1     |                               1 |                             1 |                         1     |                       1     |
| enumeration (parametric) |                         1     |                         1     |                               1 |                             1 |                         0.667 |                       1     |
| enumeration (rag)        |                         1     |                         1     |                               1 |                             1 |                         1     |                       1     |
| join                     |                         0.667 |                         1     |                               1 |                             1 |                         0.667 |                       0.667 |
| nested                   |                         1     |                         1     |                               1 |                             1 |                         1     |                       0     |
| projection               |                         1     |                         1     |                               1 |                             1 |                         1     |                       0     |
| selection                |                         0.667 |                         0.667 |                               1 |                             1 |                         1     |                       1     |
| setop                    |                         1     |                         1     |                               1 |                             1 |                         1     |                       1     |


## Error counts (direct prompting, set answers)

| model                         |   tp |   missing |   wrong |   made_up |   duplicates |
|:------------------------------|-----:|----------:|--------:|----------:|-------------:|
| claude-cli/claude-haiku-4-5   |  312 |        11 |       1 |         1 |            0 |
| claude-cli/claude-haiku-5-5   |  322 |         1 |       0 |         0 |            0 |
| opencode-go/deepseek-v4-flash |  323 |         0 |       0 |         0 |            0 |
| opencode-go/deepseek-v4-pro   |  323 |         0 |       0 |         0 |            0 |
| opencode-go/mimo-v2.6-flash   |  319 |         4 |       4 |         0 |            0 |
| opencode-go/mimo-v2.6-pro     |  321 |         2 |       1 |         0 |            0 |


## Scalar aggregation

|                                                      |   n |   exact |   undercount |   overcount |
|:-----------------------------------------------------|----:|--------:|-------------:|------------:|
| ('claude-cli/claude-haiku-4-5', 'direct', 'count')   |   1 |       0 |            1 |           0 |
| ('claude-cli/claude-haiku-5-5', 'direct', 'count')   |   1 |       1 |            0 |           0 |
| ('opencode-go/deepseek-v4-flash', 'direct', 'count') |   1 |       1 |            0 |           0 |
| ('opencode-go/deepseek-v4-pro', 'direct', 'count')   |   1 |       1 |            0 |           0 |
| ('opencode-go/mimo-v2.6-flash', 'direct', 'count')   |   1 |       0 |            1 |           0 |
| ('opencode-go/mimo-v2.6-pro', 'direct', 'count')     |   1 |       1 |            0 |           0 |


## Usage

|                                             |   calls | served_by                 |     input_tokens |   cached_input |   output_tokens |   reasoning_tokens |   cost_usd |   mean_latency_s |
|:--------------------------------------------|--------:|:--------------------------|-----------------:|---------------:|----------------:|-------------------:|-----------:|-----------------:|
| ('claude-cli/claude-haiku-4-5', 'direct')   |      20 | claude-haiku-4-5-20251001 | 171198           |              0 |          102808 |             100023 |     0.6852 |            nan   |
| ('claude-cli/claude-haiku-5-5', 'direct')   |      20 | claude-haiku-5-5          | 224968           |              0 |           33241 |              29779 |   nan      |            nan   |
| ('opencode-go/deepseek-v4-flash', 'direct') |      21 | deepseek-v4-flash         | 157116           |              0 |           64514 |              62881 |     0.0623 |             21.1 |
| ('opencode-go/deepseek-v4-pro', 'direct')   |      23 | deepseek-v4-pro           | 172374           |              0 |           94275 |              92339 |     0.3004 |             35.7 |
| ('opencode-go/mimo-v2.6-flash', 'direct')   |      21 | mimo-v2.6-flash           | 177183           |          13056 |           39437 |              37478 |     0.0341 |             46.3 |
| ('opencode-go/mimo-v2.6-pro', 'direct')     |      21 | mimo-v2.6-pro             | 177183           |          13056 |           41965 |              39760 |     0.108  |             39   |
| ('TOTAL', '')                               |     126 | nan                       |      1.08002e+06 |          26112 |          376240 |             362260 |     1.19   |             35.5 |


Tokens are as reported by each provider. Cost is the provider's figure when it reports one, otherwise computed from `usage.PRICES`. Claude CLI costs are list-price equivalents: a subscription is not billed per call. Calls counts every API call made, including retried attempts.



## Parse failures and API errors

|                                             |   parse_fail |   api_error |   truncated |
|:--------------------------------------------|-------------:|------------:|------------:|
| ('claude-cli/claude-haiku-4-5', 'direct')   |            0 |           0 |           0 |
| ('claude-cli/claude-haiku-5-5', 'direct')   |            0 |           0 |           0 |
| ('opencode-go/deepseek-v4-flash', 'direct') |            0 |           0 |           0 |
| ('opencode-go/deepseek-v4-pro', 'direct')   |            0 |           0 |           0 |
| ('opencode-go/mimo-v2.6-flash', 'direct')   |            0 |           0 |           0 |
| ('opencode-go/mimo-v2.6-pro', 'direct')     |            0 |           0 |           0 |
