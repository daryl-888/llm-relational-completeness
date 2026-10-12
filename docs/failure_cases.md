# Failure cases

Each case is a question a model got wrong in the main run (`results/full_v0`, repeat 0), re-run three
more times on the same model (`results/failure_cases`, repeats 1–3). Everything is re-scored with the
current scorer. Models first run on OpenCode Go were re-run on the same model through OpenRouter.

## Claude Haiku 5.5: `enum-par-013`

**Question** (enumeration, none format, 50 records, ~0.0k tokens of data, 2 correct answers): List all U.S. states whose name ends with the letter y. Answer from your own knowledge.

- main run: recall 0.50, 1 missing
- re-run 1: recall 0.50, 1 missing
- re-run 2: correct
- re-run 3: recall 0.50, 1 missing
- Missed in every failed run: new jersey
- **Reproduced in 2 of 3 re-runs.**

## Claude Haiku 5.5: `enum-par-021`

**Question** (enumeration, none format, 50 records, ~0.0k tokens of data, 11 correct answers): List all U.S. states whose name contains the letter sequence "in" (case-insensitive). Answer from your own knowledge.

- main run: recall 0.73, 3 missing
- re-run 1: recall 0.73, 3 missing
- re-run 2: recall 0.91, 1 missing
- re-run 3: recall 0.73, 3 missing
- Missed in every failed run: maine
- **Reproduced in 3 of 3 re-runs.**

## Claude Haiku 5.5: `sel-029`

**Question** (selection, csv format, 400 records, ~4.6k tokens of data, 13 correct answers): List the names of all products whose units in stock is less than 22.

- main run: recall 0.85, 2 missing
- re-run 1: correct
- re-run 2: correct
- re-run 3: correct
- Missed in every failed run: fendorlo k-579, soltor f-628
- **Reproduced in 0 of 3 re-runs.**

## Claude Haiku 5.5: `join-004`

**Question** (join, csv format, 100 records, ~1.5k tokens of data, 55 correct answers): List every (employee name, project name) pair such that the employee's office city is the same as the project's city, the employee's remote-work status is yes, and the project's status is Completed.

- main run: recall 0.93, 4 missing
- re-run 1: correct
- re-run 2: correct
- re-run 3: correct
- Missed in every failed run: marissa carroll / orquin, marissa carroll / solquinfen, marissa carroll / thaloqua, marissa carroll / zenrasen
- **Reproduced in 0 of 3 re-runs.**

## Claude Haiku 5.5: `nest-022`

**Question** (nested, csv format, 200 records, ~2.3k tokens of data, 13 correct answers): List the names of all products made by Draphe whose price is higher than the average price of all products (any brand) in the same category.

- main run: recall 0.92, 1 missing
- re-run 1: recall 0.92, 1 missing
- re-run 2: correct
- re-run 3: recall 0.15, 11 missing, 1 wrong
- **Reproduced in 2 of 3 re-runs.**

## Claude Haiku 5.5: `proj-031`

**Question** (projection, nl format, 300 records, ~8.5k tokens of data, 37 correct answers): For every book whose publisher is Ralys Press, give its title and publication year and genre.

- main run: recall 0.89, 4 missing
- re-run 1: correct
- re-run 2: correct
- re-run 3: correct
- Missed in every failed run: archive of crimson zenfenmi / 2007 / poetry, garden of velvet lysul / 2011 / romance, orchard of hollow dralysvex / 2017 / fantasy, winter of endless raty / 1987 / mystery
- **Reproduced in 0 of 3 re-runs.**

## Claude Haiku 5.5: `agg-015`

**Question** (aggregation, nl format, 400 records, ~11.3k tokens of data, 12 correct answers): How many books whose author is David Gutierrez and whose page count is less than 859 are there?

- main run: answered 11 (gold 12)
- re-run 1: correct
- re-run 2: correct
- re-run 3: correct
- **Reproduced in 0 of 3 re-runs.**

## Claude Haiku 5.5: `enum-ctx-029`

**Question** (enumeration, csv format, 500 records, ~7.9k tokens of data, 50 correct answers): List the names of all employees whose department is Support.

- main run: recall 0.96, 2 missing
- re-run 1: correct
- re-run 2: correct
- re-run 3: correct
- Missed in every failed run: david fletcher, teresa cannon
- Table position of those rows (% from the top): [39, 59]
- **Reproduced in 0 of 3 re-runs.**

## DeepSeek V4 Pro: `enum-ctx-029`

**Question** (enumeration, csv format, 500 records, ~7.9k tokens of data, 50 correct answers): List the names of all employees whose department is Support.

- main run: recall 0.98, 1 missing
- re-run 1: recall 0.90, 5 missing, 1 wrong
- re-run 2: recall 0.96, 2 missing, 2 wrong
- re-run 3: recall 0.86, 7 missing
- **Reproduced in 3 of 3 re-runs.**

## MiMo-V2.6 Flash: `enum-ctx-029`

**Question** (enumeration, csv format, 500 records, ~7.9k tokens of data, 50 correct answers): List the names of all employees whose department is Support.

- main run: recall 0.94, 3 missing, 3 wrong
- re-run 1: recall 0.92, 3 missing, 1 near-miss, 1 wrong
- re-run 2: no parseable answer
- re-run 3: recall 1.00, 2 wrong
- Near-misses: "devon hill" for "devin hill"
- **Reproduced in 2 of 3 re-runs (1 re-run gave no parseable answer).**

## MiMo-V2.6 Pro: `enum-ctx-029`

**Question** (enumeration, csv format, 500 records, ~7.9k tokens of data, 50 correct answers): List the names of all employees whose department is Support.

- main run: recall 0.92, 4 missing
- re-run 1: correct
- re-run 2: recall 0.84, 8 missing
- re-run 3: correct
- Missed in every failed run: daniel gamble, donald mitchell, joshua rice
- Table position of those rows (% from the top): [13, 15, 58]
- **Reproduced in 1 of 3 re-runs.**

## MiMo-V2.6 Flash: `enum-ctx-035`

**Question** (enumeration, csv format, 1000 records, ~14.2k tokens of data, 100 correct answers): List the names of all students whose major is Philosophy.

- main run: recall 0.52, 46 missing, 2 near-miss, 19 wrong, 371 made-up, 34 duplicate
- re-run 1: correct
- re-run 2: recall 0.86, 13 missing, 1 near-miss
- re-run 3: recall 0.79, 21 missing, 28 wrong, 2 made-up, 64 duplicate
- Missed in every failed run: alexandra richardson, alexandra thompson, eric charles, glenda esparza
- Table position of those rows (% from the top): [28, 41, 46, 71]
- Near-misses: "karen smith" for "kari smith"; "kristina mcdowell" for "christina mcdowell"; "philip rodriguez" for "phillip rodriguez"
- **Reproduced in 2 of 3 re-runs.**

## DeepSeek V4 Flash: `enum-ctx-018`

**Question** (enumeration, csv format, 50 records, ~0.6k tokens of data, 5 correct answers): List the names of all products whose category is Kitchen.

- main run: recall 0.80, 1 near-miss
- re-run 1: correct
- re-run 2: correct
- re-run 3: correct
- Missed in every failed run: ratygri s-799
- Table position of those rows (% from the top): [86]
- Near-misses: "raty gri s-799" for "ratygri s-799"
- **Reproduced in 0 of 3 re-runs.**

