"""How many items do real broad-search tasks require, and how many are open-set?

  .venv/bin/python analysis/widesearch.py [--data data/external/widesearch]

WideSearch (Wong et al., arXiv 2508.07999) has 200 tasks, 100 English and 100 Chinese, each asking a search agent
to fill a table, with one gold table per task. Its tasks and gold tables are released under CC0 on Hugging Face;
its per-task model results are not. This script downloads the pinned revision (checking every file against the git
blob hash in the repository listing), counts the rows of each gold table, which is the answer size n a model must
get exactly right, and splits the tasks by the hand labels in analysis/widesearch_labels.csv:
  D (determined): the rows are fixed by the query's structure or by one named list it points to, e.g. the top 10
    of a named ranking for each year 2020-2024, every member of the OECD, or entities the query names. Once the
    agent finds that list it knows how many rows it owes.
  O (open set): the rows are every entity meeting the conditions, and no single list enumerates them, e.g. every
    offshore wind farm of at least 300 MW commissioned 2010-2024. The agent must find and check candidates and
    cannot tell when it has all of them. This is the set-completeness problem proper.
The labels are one annotator's reading of each query (Claude, from the full query text); borderline calls are
flagged in the labels file, and the findings are recomputed with every borderline label flipped. Many Chinese
tasks are translations of English ones; pairs found by gold-table overlap must carry the same label.

Writes analysis/widesearch.md and analysis/widesearch.pdf.
"""
import argparse
import hashlib
import importlib.util
import json
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("ma", ROOT / "report" / "make_assets.py")
ma = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ma)  # reuse the report's plot style
REPO = "ByteDance-Seed/WideSearch"
REVISION = "6531a7e5b497d44c8912407e0cb3dc95bd98cc09"
BINS = [(1, 10), (11, 20), (21, 50), (51, 100), (101, 200), (201, 500), (501, None)]
TYPES = {"D": "determined", "O": "open set"}
TYPE_COLOR = {"D": "#2a78d6", "O": "#eb6834"}
OUR_MAX = 100  # largest answer size in benchmark_v0
TIERS = [200, 500]  # answer sizes added in the proposal


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "relcomplete-benchmark/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def download(data: Path) -> None:
    """Fetch the pinned revision into `data`, skipping files already present with the right hash."""
    api = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}"
    files = [f for f in json.loads(fetch(api)) if f["path"] == "widesearch.jsonl"]
    files += json.loads(fetch(f"{api}/widesearch_gold"))
    assert len(files) == 201, f"expected the task file and 200 gold tables, got {len(files)}"
    for f in files:
        out = data / f["path"]
        body = out.read_bytes() if out.exists() else None
        if body is None or blob_sha1(body) != f["oid"]:
            body = fetch(f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{f['path']}")
            if blob_sha1(body) != f["oid"]:
                raise ValueError(f"{f['path']}: hash mismatch")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(body)
    (data / "REVISION").write_text(REVISION + "\n")


def blob_sha1(body: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(body) + body).hexdigest()


def load(data: Path) -> pd.DataFrame:
    rows = []
    for line in (data / "widesearch.jsonl").read_text().splitlines():
        t = json.loads(line)
        gold = pd.read_csv(data / "widesearch_gold" / f"{t['instance_id']}.csv", encoding="utf-8-sig")
        keys = json.loads(t["evaluation"])["unique_columns"]
        rows.append({"id": t["instance_id"], "lang": t["language"], "rows": len(gold), "cols": gold.shape[1],
                     "keys": len(keys), "query": t["query"]})
    d = pd.DataFrame(rows)
    labels = pd.read_csv(ROOT / "analysis" / "widesearch_labels.csv", keep_default_na=False)
    d = d.merge(labels, on="id", how="left", validate="1:1")
    assert d.type.isin(TYPES).all(), "every task needs a D or O label"
    return d


def bin_label(lo: int, hi: int | None) -> str:
    return f"{lo}+" if hi is None else f"{lo}-{hi}"


def snippet(q: str, n: int = 90) -> str:
    q = " ".join(q.split()).replace("|", "/")
    return q if len(q) <= n else q[:n].rstrip() + "..."


def translation_pairs(data: Path, d: pd.DataFrame) -> list[tuple[str, str]]:
    """English-Chinese task pairs whose gold tables share most of their ASCII cells and have about the same rows."""
    cells = {}
    for i in d.id:
        g = pd.read_csv(data / "widesearch_gold" / f"{i}.csv", encoding="utf-8-sig", dtype=str).fillna("")
        cells[i] = {c for c in g.values.ravel() if c.isascii() and len(c) > 2}
    rows = d.set_index("id").rows
    jac = lambda a, b: len(cells[a] & cells[b]) / (1 + len(cells[a] | cells[b]))  # noqa: E731
    pairs = []
    for e in d.id[d.lang == "en"]:
        z = max(d.id[d.lang == "zh"], key=lambda z: jac(e, z))
        if jac(e, z) > 0.3 and abs(rows[e] - rows[z]) <= 0.1 * rows[e] + 1:
            pairs.append((e, z))
    return pairs


def size_table(d: pd.DataFrame) -> pd.DataFrame:
    out = []
    groups = [("All", d)] + [(f"{lang}, {TYPES[t]}", g) for (lang, t), g in d.groupby(["lang", "type"])] + \
             [(f"All, {TYPES[t]}", g) for t, g in d.groupby("type")]
    for name, g in groups:
        out.append([name, len(g), g.rows.median(), round(g.rows.mean(), 1), int(g.rows.quantile(0.9)), g.rows.max(),
                    int((g.rows * g.cols).median())])
    return pd.DataFrame(out, columns=["Tasks", "n", "Median rows", "Mean rows", "90th pct", "Max",
                                      "Median cells"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "external" / "widesearch")
    args = ap.parse_args()
    download(args.data)
    d = load(args.data)
    o, det = d[d.type == "O"], d[d.type == "D"]
    md = ["# Answer sizes in WideSearch's gold tables", "",
          f"Source: Hugging Face `{REPO}`, revision `{REVISION}` (CC0), {len(d)} tasks with one gold table each. "
          "Rows per gold table is the answer size a model must get exactly right. Task types (D = determined by "
          "the query, O = open set) are hand labels from `analysis/widesearch_labels.csv`; see the script's "
          "docstring for the rule.", ""]

    md += ["## 1. Rows per task", "", size_table(d).to_markdown(index=False), ""]

    counts = pd.DataFrame({bin_label(lo, hi): [int(((g.rows >= lo) & ((hi is None) | (g.rows <= (hi or 0)))).sum())
                                               for g in (det, o, d)] for lo, hi in BINS},
                          index=[TYPES["D"], TYPES["O"], "all"])
    md += ["## 2. Tasks by number of rows", "", counts.to_markdown(), ""]

    thr = [50, OUR_MAX, *TIERS]
    share = pd.DataFrame({f"> {t} rows" if t == OUR_MAX else f">= {t} rows":
                          [f"{100 * ((g.rows > t) if t == OUR_MAX else (g.rows >= t)).mean():.0f}%" for g in (det, o, d)]
                          for t in thr}, index=[TYPES["D"], TYPES["O"], "all"])
    md += ["## 3. Share of tasks at or beyond our answer-size tiers", "",
           f"benchmark_v0's largest answer has {OUR_MAX} items; the proposal adds tiers of {TIERS[0]} and {TIERS[1]}.",
           "", share.to_markdown(), ""]

    big = o.sort_values("rows", ascending=False).head(15)
    md += ["## 4. The 15 largest open-set tasks", "",
           big.assign(query=big["query"].map(snippet))[["id", "rows", "cols", "query"]].to_markdown(index=False), ""]

    sci = d[d.domain == "scientific publications"].sort_values("rows", ascending=False)
    md += ["## 5. Scientific-publication tasks", "",
           sci[["id", "type", "rows", "cols", "note"]].rename(columns={"note": "task"}).to_markdown(index=False), "",
           "No task concerns Magic: The Gathering (the only match for \"Magic\" is the AppMagic games chart, "
           "ws_en_060).", ""]

    pairs = translation_pairs(args.data, d)
    t = d.set_index("id").type
    bad = [p for p in pairs if t[p[0]] != t[p[1]]]
    assert not bad, f"translated pairs with different labels: {bad}"
    flip = d.assign(type=d.type.where(d.borderline != "yes", d.type.map({"D": "O", "O": "D"})))
    fo, fd = flip[flip.type == "O"], flip[flip.type == "D"]
    md += ["## 6. Label checks", "",
           f"- {len(pairs)} English-Chinese translation pairs found by gold-table overlap all carry the same label.",
           f"- {(d.borderline == 'yes').sum()} labels are flagged borderline. Flipping all of them gives "
           f"{len(fo)} open-set tasks ({100 * len(fo) / len(d):.0f}%); {100 * (fo.rows > OUR_MAX).mean():.0f}% of "
           f"open-set and {100 * (fd.rows > OUR_MAX).mean():.0f}% of determined tasks then need more than "
           f"{OUR_MAX} rows.", ""]

    sci_distinct = len(sci) - sum(a in set(sci.id) and b in set(sci.id) for a, b in pairs)
    md += ["## Findings", "",
           f"- {len(o)} of {len(d)} tasks ({100 * len(o) / len(d):.0f}%) are open-set; the other {len(det)} have a "
           "size fixed by the query.",
           f"- Typical sizes are similar (median {o.rows.median():g} rows open-set, {det.rows.median():g} "
           f"determined), but open-set tasks have the long tail: 90th percentile {int(o.rows.quantile(0.9))} "
           f"against {int(det.rows.quantile(0.9))} rows, and every task above {det.rows.max()} rows is open-set.",
           f"- {100 * (o.rows > OUR_MAX).mean():.0f}% of open-set tasks need more than {OUR_MAX} rows, the "
           f"largest answer in benchmark_v0, and {100 * (o.rows >= TIERS[1]).mean():.0f}% need {TIERS[1]} or "
           f"more (largest: {o.rows.max()} rows). Tiers of {TIERS[0]} and {TIERS[1]} match the upper tail of "
           "real tasks rather than being artificial stress tests.",
           f"- Chinese tasks are about twice as large (median {d[d.lang == 'zh'].rows.median():g} rows against "
           f"{d[d.lang == 'en'].rows.median():g} for English).",
           f"- All {len(sci)} scientific-publication tasks are open-set ({sci_distinct} distinct, as some are "
           f"translations), from {sci.rows.min()} to "
           f"{sci.rows.max()} rows, and bibliographic databases give a checkable gold set. They are candidates for "
           "our own real-world tasks; MTG has no counterpart here, so its tasks must be built from scratch.", "",
           "## Caveats", "",
           "- The gold tables are WideSearch's own answers. For open-set tasks they are the set an agent is scored "
           "against, not a proven complete set, so row counts may undercount the true answer.",
           f"- The {len(pairs)} translation pairs count twice in every table above.",
           "- Rows are the unit WideSearch scores; some tasks have several rows per entity (e.g. one per entity "
           "and year), so rows overstate the number of distinct entities.", ""]
    (ROOT / "analysis" / "widesearch.md").write_text("\n".join(md) + "\n")

    # Figure: tasks per size bin by type (left), share of tasks with at least n rows (right)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5))
    x = np.arange(len(BINS))
    for j, t in enumerate(TYPES):
        axes[0].bar(x + (j - 0.5) * 0.4, counts.loc[TYPES[t]].values, width=0.4, color=TYPE_COLOR[t],
                    label=TYPES[t])
    axes[0].set_xticks(x, [bin_label(lo, hi) for lo, hi in BINS], fontsize=7, rotation=30)
    axes[0].set(xlabel="Rows in the gold table", ylabel="Tasks")
    axes[0].legend(fontsize=7)
    axes[0].grid(axis="x", visible=False)
    grid = np.unique(np.geomspace(1, d.rows.max(), 200).astype(int))
    for t, g in d.groupby("type"):
        axes[1].plot(grid, [100 * (g.rows >= v).mean() for v in grid], color=TYPE_COLOR[t], label=TYPES[t])
    for v, s in [(OUR_MAX, f"benchmark_v0 max ({OUR_MAX})"), *[(t, f"proposed tier ({t})") for t in TIERS]]:
        axes[1].axvline(v, color=ma.INK2, linestyle="--", linewidth=0.8)
        axes[1].text(v * 1.1, 97, s, fontsize=6, color=ma.INK2, va="top", rotation=90)
    axes[1].set(xscale="log", xlabel="Rows (n)", ylabel="Tasks with at least n rows (%)", ylim=(0, 102))
    axes[1].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(ROOT / "analysis" / "widesearch.pdf")
    plt.close(fig)
    print("\n".join(md))


if __name__ == "__main__":
    main()
