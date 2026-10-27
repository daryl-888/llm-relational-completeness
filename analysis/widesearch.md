# Answer sizes in WideSearch's gold tables

Source: Hugging Face `ByteDance-Seed/WideSearch`, revision `6531a7e5b497d44c8912407e0cb3dc95bd98cc09` (CC0), 200 tasks with one gold table each. Rows per gold table is the answer size a model must get exactly right. Task types (D = determined by the query, O = open set) are hand labels from `analysis/widesearch_labels.csv`; see the script's docstring for the rule.

## 1. Rows per task

| Tasks           |   n |   Median rows |   Mean rows |   90th pct |   Max |   Median cells |
|:----------------|----:|--------------:|------------:|-----------:|------:|---------------:|
| All             | 200 |          37   |        73.2 |        130 |  1458 |            200 |
| en, determined  |  44 |          25   |        31.9 |         50 |   100 |            175 |
| en, open set    |  56 |          30.5 |        60.5 |         98 |   534 |            185 |
| zh, determined  |  36 |          40   |        61.4 |        158 |   240 |            222 |
| zh, open set    |  64 |          52.5 |       119.4 |        219 |  1458 |            230 |
| All, determined |  80 |          30   |        45.2 |         77 |   240 |            200 |
| All, open set   | 120 |          38   |        91.9 |        145 |  1458 |            214 |

## 2. Tasks by number of rows

|            |   1-10 |   11-20 |   21-50 |   51-100 |   101-200 |   201-500 |   501+ |
|:-----------|-------:|--------:|--------:|---------:|----------:|----------:|-------:|
| determined |      7 |      13 |      45 |       10 |         2 |         3 |      0 |
| open set   |      7 |      22 |      40 |       28 |        11 |         7 |      5 |
| all        |     14 |      35 |      85 |       38 |        13 |        10 |      5 |

## 3. Share of tasks at or beyond our answer-size tiers

benchmark_v0's largest answer has 100 items; the proposal adds tiers of 200 and 500.

|            | >= 50 rows   | > 100 rows   | >= 200 rows   | >= 500 rows   |
|:-----------|:-------------|:-------------|:--------------|:--------------|
| determined | 34%          | 6%           | 4%            | 0%            |
| open set   | 43%          | 19%          | 10%           | 4%            |
| all        | 40%          | 14%          | 8%            | 2%            |

## 4. The 15 largest open-set tasks

| id        |   rows |   cols | query                                                                                         |
|:----------|-------:|-------:|:----------------------------------------------------------------------------------------------|
| ws_zh_008 |   1458 |      6 | 我是25年北京高考考生，帮我梳理一下20-24年北京普通批中600至690分区间的录取投档线信息，包括各院校及专业组，以及对应的投档线总分，总分由高到低进行降序统计。并且，我还需要... |
| ws_zh_086 |   1037 |      4 | 最近我是歌手的单依纯很火，歌手这个节目热度依旧。所以我想知道从2013年开始办歌手的节目，到2024年（包含2013年和2024年）的歌手，每一季每一期每位歌手唱了什么歌。你帮我以... |
| ws_en_006 |    534 |      5 | Could you list every single concert on Taylor Swift’s official tour from January 1, 2010,...  |
| ws_zh_007 |    515 |     10 | 帮我梳理一份Nvidia自1990年至2024年以来（包含1990年和2024年）发布的桌面级显卡产品清单，包括具体的产品系列、产品名称、芯片名称、发布日期、总线接口、显存容量、显... |
| ws_en_073 |    510 |     10 | Please help me sort out a list of NVIDIA's desktop graphics card products released from 19... |
| ws_zh_081 |    428 |      5 | 我读了迟子建的《额尔古纳河右岸》，对鄂温克族文化产生了浓厚兴趣，计划写一篇公众号文章。为充实内容，需要搜集相关学术资料。 请帮我在万方数据库检索 2005–2024年 发表的期刊论... |
| ws_en_091 |    331 |     12 | It is said that "Mom Su" has saved AMD. Please help me sort out the specific processor (CP... |
| ws_zh_061 |    331 |     12 | 都说"苏妈"挽救了AMD，请帮我整理一份苏姿丰出任AMD公司CEO推出"Zen"架构至2023年以来（包含2023年），AMD具体发布了哪些处理器(CPU)产品，以及产品的具体信息... |
| ws_zh_006 |    285 |      6 | 我要一份贵州茅台、五粮液、泸州老窖三个品牌截至2025年6月的产品清单，请分别列出品牌、产品系列、产品名称、香型、规格(/瓶)、酒精度，请将结果整理成一个markdown表格，哪一... |
| ws_zh_014 |    224 |      6 | 帮我把邓紫棋从2010年1月1日-2025年5月1日期间举行的所有巡回演唱会的具体日期（xx年xx月xx日格式，不要用区间表示）、演唱会中文名、演唱会英文名、举办国家、举办城市、举... |
| ws_zh_098 |    208 |      4 | 截止到2025年5月31日，给我整理所有体育学北大核心期刊各自最新一刊所有研究论文，包括期刊名称、论文名称、论文关键词、论文作者。请以一整个Markdown表格的格式输出整理后的数... |
| ws_zh_074 |    205 |      5 | 浪姐1-6季每季公演（一公-五公）的节目（不含个人表演，特指多个姐姐组成的团体表演）、表演的姐姐、及其票数都是什么？票数不包含加成例如合作秀、赛前秀等加成，如果没有具体票数，可以用... |
| ws_zh_088 |    139 |      5 | 保定最大的汽车产业就是长城汽车了，所以我想看下截止到2025年5月31日长城汽车所有在售车型的信息，我希望获取其完整的产品线数据。统一车型可能存在多种配置版本请全部分行列出 需要整... |
| ws_zh_012 |    133 |      5 | 整合《财富》给出的世界2024年世界500强企业中的中国公司名单，包括榜单排名、企业名称，注意企业名称需要同时给出中英文，例如国家电网有限公司（STATE GRID)、营业收入、利... |
| ws_zh_010 |    132 |     20 | 帮我整理汇总羽毛球明星林丹在2010-2020年的赛事情况，生成一张清晰的markdown表，包括时间、赛事名称、级别、项目、赛事成绩、对局信息(包括对手、比分、胜负)，对局信息需... |

## 5. Scientific-publication tasks

| id        | type   |   rows |   cols | task                                                                     |
|:----------|:-------|-------:|-------:|:-------------------------------------------------------------------------|
| ws_zh_081 | O      |    428 |      5 | Wanfang papers 2005-2024 with 'Evenki' (鄂温克族) in the title               |
| ws_zh_098 | O      |    208 |      4 | every paper in the latest issue of each PKU-core sports journal          |
| ws_zh_069 | O      |    129 |      4 | LLM papers by ByteDance Seed and DeepSeek (Chinese version of ws_en_093) |
| ws_en_093 | O      |    129 |      4 | LLM papers by ByteDance Seed and DeepSeek                                |
| ws_en_023 | O      |     56 |      7 | journal articles with a given title keyword                              |
| ws_zh_077 | O      |     26 |      7 | papers by one author (Zhang Linghan) indexed in CNKI                     |

No task concerns Magic: The Gathering (the only match for "Magic" is the AppMagic games chart, ws_en_060).

## 6. Label checks

- 12 English-Chinese translation pairs found by gold-table overlap all carry the same label.
- 19 labels are flagged borderline. Flipping all of them gives 125 open-set tasks (62%); 18% of open-set and 7% of determined tasks then need more than 100 rows.

## Findings

- 120 of 200 tasks (60%) are open-set; the other 80 have a size fixed by the query.
- Typical sizes are similar (median 38 rows open-set, 30 determined), but open-set tasks have the long tail: 90th percentile 145 against 77 rows, and every task above 240 rows is open-set.
- 19% of open-set tasks need more than 100 rows, the largest answer in benchmark_v0, and 4% need 500 or more (largest: 1458 rows). Tiers of 200 and 500 match the upper tail of real tasks rather than being artificial stress tests.
- Chinese tasks are about twice as large (median 49.5 rows against 25.5 for English).
- All 6 scientific-publication tasks are open-set (5 distinct, as some are translations), from 26 to 428 rows, and bibliographic databases give a checkable gold set. They are candidates for our own real-world tasks; MTG has no counterpart here, so its tasks must be built from scratch.

## Caveats

- The gold tables are WideSearch's own answers. For open-set tasks they are the set an agent is scored against, not a proven complete set, so row counts may undercount the true answer.
- The 12 translation pairs count twice in every table above.
- Rows are the unit WideSearch scores; some tasks have several rows per entity (e.g. one per entity and year), so rows overstate the number of distinct entities.

