"""PEC 2 - Data Visualization: build the three chart pages (timeline, radar, violin).

Reads the Epoch AI files in ./data and writes static HTML into ./docs (the folder GitHub Pages serves).

    pip install -r requirements.txt
    python build.py

Data preparation (the "T" in ETL) is done here, in plain pandas, so every rule is visible:
  * timeline: frontier models since 2015; lane = first listed organisation (Google entities merged);
              open = downloadable weights, closed = API / hosted / limited / unreleased.
  * radar:    six Epoch-run benchmarks; axis order = median score over the models that have all six.
  * violin:   notable models since 2018 with a compute estimate; domain = "Multimodal" if listed,
              otherwise the first-listed domain; groups with fewer than 30 models are not drawn.
"""
import html
import os
import shutil

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.environ.get("PEC2_OUT", os.path.join(HERE, "docs"))
os.makedirs(OUT, exist_ok=True)

# ---- design tokens (light mode) ---------------------------------------------------------------
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
OPEN, CLOSED, UNKNOWN = "#2a78d6", "#eb6834", "#a8a7a1"
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
CREDIT = ("Data: Epoch AI, &ldquo;Data on AI Models&rdquo; and &ldquo;Capabilities &amp; benchmarking&rdquo;, "
          "epoch.ai, accessed 30 Oct 2026, licensed CC BY 4.0 (changes: filtered and regrouped).")

CSS = f"""
:root {{ --surface:{SURFACE}; --ink:{INK}; --ink2:{INK2}; --grid:{GRID}; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--surface); color:var(--ink); font-family:{FONT}; line-height:1.45; }}
main {{ max-width:1080px; margin:0 auto; padding:28px 20px 48px; }}
h1 {{ font-size:1.55rem; line-height:1.2; margin:0 0 8px; font-weight:650; }}
.sub {{ color:var(--ink2); margin:0 0 18px; max-width:62em; }}
.notes {{ color:var(--ink2); font-size:.92rem; max-width:62em; }}
.notes li {{ margin:4px 0; }}
details {{ margin:18px 0; }}
summary {{ cursor:pointer; color:var(--ink2); font-size:.92rem; }}
table {{ border-collapse:collapse; font-size:.85rem; margin-top:8px; }}
th, td {{ border-bottom:1px solid var(--grid); padding:4px 10px; text-align:left; }}
footer {{ margin-top:26px; padding-top:14px; border-top:1px solid var(--grid); color:var(--ink2); font-size:.85rem; }}
.tag {{ display:inline-block; font-size:.75rem; color:var(--ink2); border:1px solid var(--grid); border-radius:10px; padding:1px 8px; margin-bottom:10px; }}
.nav {{ font-size:.85rem; margin-bottom:6px; }}
a {{ color:#256abf; }}
.cards a {{ display:block; padding:14px 16px; margin:10px 0; border:1px solid var(--grid); border-radius:8px; text-decoration:none; color:var(--ink); }}
.cards a span {{ display:block; color:var(--ink2); font-size:.9rem; margin-top:2px; }}
"""


def write_page(filename, tag, title, subtitle, fig, notes, table_html):
    chart = fig.to_html(full_html=False, include_plotlyjs="directory",
                        config={"displaylogo": False, "responsive": True})
    notes_html = "".join(f"<li>{n}</li>" for n in notes)
    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title><style>{CSS}</style></head>
<body><main>
<div class="nav"><a href="index.html">&larr; All charts</a></div>
<span class="tag">{tag}</span>
<h1>{title}</h1>
<p class="sub">{subtitle}</p>
{chart}
<ul class="notes">{notes_html}</ul>
<details><summary>Show the data behind this chart (table view)</summary>{table_html}</details>
<footer>{CREDIT} PEC 2, Data Visualization (UOC).</footer>
</main></body></html>"""
    with open(os.path.join(OUT, filename), "w", encoding="utf-8") as fh:
        fh.write(doc)


def base_layout(**kw):
    lay = dict(paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font=dict(family=FONT, size=13, color=INK2),
               hoverlabel=dict(bgcolor="white", font=dict(family=FONT, size=13, color=INK), bordercolor=GRID),
               margin=dict(l=20, r=20, t=10, b=20))
    lay.update(kw)
    return lay


def uniq(s):
    """Split a comma-separated multi-select cell into unique values, keeping order."""
    out = []
    for x in ([] if pd.isna(s) else str(s).split(",")):
        x = x.strip()
        if x and x not in out:
            out.append(x)
    return out


# =================================================================================================
# 1. TIMELINE: frontier models since 2015
# =================================================================================================
def build_timeline():
    fr = pd.read_csv(os.path.join(DATA, "epoch", "frontier_ai_models.csv"))
    nb = pd.read_csv(os.path.join(DATA, "epoch", "notable_ai_models.csv"))
    for d in (fr, nb):
        d["date"] = pd.to_datetime(d["Publication date"])

    def lane(v):
        l = uniq(v)
        if not l:
            return "Other"
        o = l[0]
        if "Google" in o or "DeepMind" in o:
            return "Google (incl. DeepMind)"
        if o in ("Meta AI", "Facebook", "Facebook AI"):
            return "Meta"
        return o if o in ("OpenAI", "Anthropic", "xAI", "Nvidia") else "Other"

    def access(v):
        if pd.isna(v):
            return "Unknown"
        return "Open weights" if "Open weights" in str(v) else "Closed"

    lanes = ["Google (incl. DeepMind)", "OpenAI", "Anthropic", "xAI", "Meta", "Nvidia", "Other"]
    f = fr[fr.date.dt.year >= 2015].sort_values("date").copy()
    f["lane"], f["acc"] = f["Organization"].map(lane), f["Model accessibility"].map(access)

    def share(lo, hi):
        s = f[(f.date.dt.year >= lo) & (f.date.dt.year <= hi)]
        return int((s.acc == "Open weights").sum()), len(s), int((s.acc == "Unknown").sum())

    o1, n1, u1 = share(2015, 2018)
    o2, n2, _ = share(2019, 2021)
    o3, n3, _ = share(2022, 2026)
    p2, p3 = round(100 * o2 / n2), round(100 * o3 / n3)

    # 2026 restricted releases from the Notable file (Epoch labels: Limited access / Unreleased), three big labs only
    r = nb[(nb.date >= "2026-01-01") & (nb["Model accessibility"].isin(["Limited access", "Unreleased"]))].copy()
    r["lane"] = r["Organization"].map(lane)
    r = r[r.lane.isin(["Google (incl. DeepMind)", "OpenAI", "Anthropic"])]

    label_pos = {"AlphaGo Zero": "top center", "GPT-2 (774M)": "top center", "GPT-3 175B (davinci)": "bottom center",
                 "PaLM (540B)": "top center", "GPT-4 (Mar 2023)": "bottom center", "Gemini 1.0 Ultra": "top center",
                 "Claude 3 Opus": "bottom center", "Llama 3.1-405B": "top center", "Falcon-180B": "bottom center",
                 "Nemotron-4 340B": "top center", "Grok 3": "top center", "Llama 4 Behemoth (preview)": "bottom center",
                 "GPT-6 Astra": "top center"}
    short = {"GPT-2 (774M)": "GPT-2", "GPT-3 175B (davinci)": "GPT-3", "PaLM (540B)": "PaLM",
             "GPT-4 (Mar 2023)": "GPT-4", "Llama 4 Behemoth (preview)": "Llama 4 Behemoth"}
    rlabel_pos = {"Claude Mythos Preview": "top left", "Claude Mythos 5": "bottom left", "GPT-5.4-Cyber": "top left",
                  "GPT-5.5 Cyber": "bottom left", "Gemini 4 Argon": "top left", "Gemini 4 Carbon": "bottom left"}

    fig = go.Figure()
    for key, color, name in (("Open weights", OPEN, "Open weights (downloadable)"),
                             ("Closed", CLOSED, "Closed (API, hosted or unreleased)"),
                             ("Unknown", UNKNOWN, "Accessibility not recorded")):
        d = f[f.acc == key]
        fig.add_trace(go.Scatter(
            x=d.date, y=d.lane, mode="markers+text", name=name,
            text=[short.get(m, m) if m in label_pos else "" for m in d.Model],
            textposition=[label_pos.get(m, "top center") for m in d.Model], textfont=dict(size=11, color=INK2),
            marker=dict(size=11, color=color, line=dict(width=2, color=SURFACE)),
            customdata=np.stack([d.Model, d["Organization"].fillna("n/a"), d["Model accessibility"].fillna("not recorded"),
                                 d["Training compute (FLOP)"].map(lambda v: "n/a" if pd.isna(v) else f"{v:.1e} FLOP")], axis=-1),
            hovertemplate="<b>%{customdata[0]}</b><br>%{x|%d %b %Y}<br>Organisation: %{customdata[1]}<br>"
                          "Access: %{customdata[2]}<br>Training compute: %{customdata[3]}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=r.date, y=r.lane, mode="markers+text", name="Restricted 2026 release (Notable file)",
        text=[m if m in rlabel_pos else "" for m in r.Model], textposition=[rlabel_pos.get(m, "top center") for m in r.Model],
        textfont=dict(size=11, color=INK2),
        marker=dict(size=13, symbol="diamond", color=SURFACE, line=dict(width=2.5, color=CLOSED)),
        customdata=np.stack([r.Model, r["Organization"].fillna("n/a"), r["Model accessibility"]], axis=-1),
        hovertemplate="<b>%{customdata[0]}</b><br>%{x|%d %b %Y}<br>Organisation: %{customdata[1]}<br>"
                      "Epoch access label: %{customdata[2]}<extra></extra>"))

    bands = [(pd.Timestamp("2015-01-01"), pd.Timestamp("2019-01-01"), f"2015–18: access unknown for {u1} of {n1}"),
             (pd.Timestamp("2019-01-01"), pd.Timestamp("2022-01-01"), f"2019–21: {o2} of {n2} open ({p2}%)"),
             (pd.Timestamp("2022-01-01"), pd.Timestamp("2027-01-01"), f"2022–26: {o3} of {n3} open ({p3}%)")]
    for i, (a, b, txt) in enumerate(bands):
        fig.add_vrect(x0=a, x1=b, fillcolor="#f0efec" if i % 2 == 0 else SURFACE, opacity=0.9, layer="below", line_width=0)
        fig.add_annotation(x=a + (b - a) / 2, y=1.0, yref="paper", yanchor="bottom", text=f"<b>{txt}</b>",
                           showarrow=False, font=dict(size=12.5, color=INK))
    fig.update_layout(**base_layout(
        height=660, margin=dict(l=150, r=20, t=40, b=20),
        xaxis=dict(range=[pd.Timestamp("2014-10-01"), pd.Timestamp("2027-04-30")], gridcolor=GRID, tickformat="%Y",
                   dtick="M12", showline=False, zeroline=False),
        yaxis=dict(categoryorder="array", categoryarray=lanes[::-1], gridcolor=GRID, zeroline=False,
                   tickfont=dict(color=INK, size=13)),
        legend=dict(orientation="h", yanchor="top", y=-0.04, x=0), hovermode="closest"))
    table = f[["Model", "Publication date", "lane", "acc", "Model accessibility"]].rename(
        columns={"lane": "Lane", "acc": "Open/closed group", "Model accessibility": "Epoch access label"}).to_html(index=False, border=0)
    write_page(
        "timeline.html", "Timeline (scaled)",
        f"Fewer frontier AI models are open-weights: {p2}% in 2019–21, {p3}% since 2022",
        "Each dot is a frontier model (top 5 by training compute when released), positioned by release date and by the lab that built it. "
        "Blue = weights downloadable, orange = closed.", fig,
        ["<b>Frontier</b> follows Epoch AI&rsquo;s definition: top 5 by training compute at release. It says nothing about capability.",
         "<b>Closed</b> groups Epoch&rsquo;s labels API access, hosted access, limited access and unreleased; <b>open</b> means downloadable weights.",
         "<b>Limit:</b> the frontier list has only 7 models in 2025 and 1 in 2026, because many recent models have no compute estimate. "
         "The diamonds add 2026 models that Epoch labels limited-access or unreleased, taken from its separate Notable file (OpenAI, Anthropic and Google only).",
         "Lanes use the first organisation listed for each model; Google, DeepMind, Brain and Research are merged. "
         "Descriptive only: the data do not show <i>why</i> models became closed."], table)


# =================================================================================================
# 2. RADAR: three 2026 models on six Epoch-run benchmarks
# =================================================================================================
BENCH_FILES = {  # file -> axis label
    "otis_mock_aime_2024_2025.csv": "Mock AIME<br>(competition maths)",
    "gpqa_diamond.csv": "GPQA Diamond<br>(grad-level science)",
    "frontiermath_tiers_1_3_v2.csv": "FrontierMath<br>Tiers 1–3",
    "simpleqa_verified.csv": "SimpleQA<br>Verified (facts)",
    "chess_puzzles.csv": "Chess puzzles",
    "frontiermath_tier_4_v2.csv": "FrontierMath<br>Tier 4 (hardest)",
}
# (model id, legend label, colour, line dash, marker symbol, line width, marker size, fill opacity)
RADAR_MODELS = [
    ("gpt-6-astra_max", "GPT-6 Astra (closed flagship)", CLOSED, "solid", "circle", 5, 15, 0.04),
    ("gemini-3.7-flash_high", "Gemini 3.7 Flash (closed, lighter tier)", CLOSED, "dash", "diamond", 4, 17, 0.0),
    ("deepseek-v4.1-flash-1017_max", "DeepSeek V4.1 Flash (open weights)", OPEN, "solid", "square", 5.5, 14, 0.05),
]


def build_radar():
    cols = {}
    for fn, label in BENCH_FILES.items():
        d = pd.read_csv(os.path.join(DATA, "benchmarks", fn))
        d = d.dropna(subset=["Best score (across scorers)"])
        cols[label] = d.groupby("Model version")["Best score (across scorers)"].max()
        cols[(label, "rel")] = pd.to_datetime(d.groupby("Model version")["Release date"].max())
    labels = list(BENCH_FILES.values())
    piv = pd.DataFrame({l: cols[l] for l in labels})
    rel = pd.concat([cols[(l, "rel")] for l in labels], axis=1).max(axis=1)
    complete = piv.dropna()
    complete = complete[rel.reindex(complete.index) >= "2026-01-01"]
    order = complete.median().sort_values(ascending=False).index.tolist()  # saturated first, hardest last

    fig = go.Figure()
    for mid, label, col, dash, sym, width, msize, alpha in RADAR_MODELS:
        vals = [float(piv.loc[mid, c]) for c in order]
        rgb = tuple(int(col[i:i + 2], 16) for i in (1, 3, 5))
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=order + [order[0]], name=label, mode="lines+markers",
            fill="toself" if alpha > 0 else "none", fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha})",
            line=dict(color=col, width=width, dash=dash),
            marker=dict(size=msize, symbol=sym, color=col, line=dict(width=2.5, color=SURFACE)),
            hovertemplate="<b>" + label + "</b><br>%{theta}<br>Score: %{r:.0%}<extra></extra>"))
    # value labels on the hardest axis only (that is where the story is); an invisible marker keeps the text off the point
    hard = order[-1]
    for (mid, label, col, _d, _s, _w, msize, _a), pos in zip(RADAR_MODELS, ("middle right", "bottom right", "middle right")):
        fig.add_trace(go.Scatterpolar(r=[float(piv.loc[mid, hard])], theta=[hard], mode="markers+text",
                                      marker=dict(size=msize + 14, opacity=0), text=[f"<b>{piv.loc[mid, hard]:.2f}</b>"],
                                      textposition=pos, textfont=dict(size=13, color=col), showlegend=False, hoverinfo="skip"))
    fig.update_layout(**base_layout(
        height=680, margin=dict(l=100, r=100, t=40, b=70),
        polar=dict(bgcolor=SURFACE,
                   radialaxis=dict(range=[0, 1], tickvals=[0.25, 0.5, 0.75], tickformat=".0%", gridcolor=GRID, linecolor=GRID,
                                   tickfont=dict(size=11, color=INK2), angle=90),
                   angularaxis=dict(gridcolor=GRID, linecolor=GRID, direction="clockwise", rotation=90,
                                    tickfont=dict(size=12.5, color=INK))),
        legend=dict(orientation="h", yanchor="top", y=-0.06, x=0.5, xanchor="center")))
    ast, gm, ds = (piv.loc[m[0], order] for m in RADAR_MODELS)
    names = {c: c.replace("<br>", " ") for c in order}
    table = pd.DataFrame({names[c]: [f"{piv.loc[m[0], c]:.2f}" for m in RADAR_MODELS] for c in order},
                         index=[m[1] for m in RADAR_MODELS]).reset_index().rename(columns={"index": "Model"}).to_html(index=False, border=0)
    write_page(
        "radar.html", "Radar chart",
        "The open-weights model trails the closed flagship on every benchmark, most on FrontierMath Tier 4",
        "Share of questions solved, 0–100%, for three models released in 2026. Axes run clockwise from the most saturated benchmark (top) "
        f"to the hardest (upper left), so the gap to the flagship widens along the way: DeepSeek scores {ds.iloc[-1]:.2f} on Tier 4 "
        f"against {ast.iloc[-1]:.2f} for GPT-6 Astra.", fig,
        ["<b>Not every closed model beats the open one:</b> Gemini 3.7 Flash, a lighter closed model, scores below DeepSeek on FrontierMath "
         f"Tier 4 ({gm.iloc[-1]:.2f} vs {ds.iloc[-1]:.2f}), so &ldquo;closed&rdquo; alone does not explain the gap; model tier matters.",
         f"<b>Limits of the radar:</b> axis order changes the shape (here: ordered by median score across the {len(complete)} 2026 models with all six scores); "
         "the top two axes are almost saturated, so they separate nothing; benchmarks have different chance levels (GPQA 25%, chess puzzles about 5%).",
         "Model names include the reasoning-effort setting used by Epoch (max, high, xhigh). Scores are single-run results; "
         "differences of a few points may be within noise."], table)


# =================================================================================================
# 3. VIOLIN: training compute by domain
# =================================================================================================
def build_violin():
    nb = pd.read_csv(os.path.join(DATA, "epoch", "notable_ai_models.csv"))
    nb["date"] = pd.to_datetime(nb["Publication date"])
    nb["dom"] = nb["Domain"].map(lambda s: None if not uniq(s) else ("Multimodal" if "Multimodal" in uniq(s) else uniq(s)[0]))
    c = nb[(nb["Training compute (FLOP)"].notna()) & (nb.date.dt.year >= 2018)].copy()
    c["lc"] = np.log10(c["Training compute (FLOP)"])
    cnt = c.dom.value_counts()
    keep = [d for d in cnt.index if cnt[d] >= 30]
    excluded = {d: int(cnt[d]) for d in cnt.index if cnt[d] < 30}
    med = c[c.dom.isin(keep)].groupby("dom").lc.median().sort_values(ascending=False)

    fig = go.Figure()
    for d in med.index:
        s = c[c.dom == d]
        fig.add_trace(go.Violin(
            y=s.lc, name=f"{d}<br>n = {len(s)}", points="all", pointpos=0, jitter=0.45, spanmode="hard",
            box=dict(visible=True, width=0.12, fillcolor=SURFACE, line=dict(color=INK, width=1.5)), meanline=dict(visible=False),
            line=dict(color=INK2, width=1.5), fillcolor="rgba(82,81,78,0.22)",
            marker=dict(size=5, color=INK2, opacity=0.45), text=s.Model, customdata=s["Training compute (FLOP)"],
            hovertemplate="<b>%{text}</b><br>%{customdata:.2e} FLOP<extra></extra>", showlegend=False, width=0.85))
    ticks = list(range(int(np.floor(c.lc.min() / 2) * 2), 28, 2))
    fig.update_layout(**base_layout(
        height=620, margin=dict(l=90, r=20, t=20, b=40), violingap=0.2,
        yaxis=dict(title="Training compute (FLOP, log scale)", tickvals=ticks, ticktext=[f"10<sup>{t}</sup>" for t in ticks],
                   gridcolor=GRID, zeroline=False, range=[np.floor(c.lc.min()) - 0.5, np.ceil(c.lc.max()) + 0.3]),
        xaxis=dict(tickfont=dict(color=INK, size=13), showgrid=False)))
    lo = 10 ** (med[["Language", "Multimodal"]].min() - med[["Vision", "Biology"]].max())
    hi = 10 ** (med[["Language", "Multimodal"]].max() - med[["Vision", "Biology"]].min())
    summ = c[c.dom.isin(keep)].groupby("dom").lc.agg(n="count", median="median", q1=lambda x: np.percentile(x, 25),
                                                     q3=lambda x: np.percentile(x, 75)).loc[med.index]
    table = summ.assign(**{"median (FLOP)": (10 ** summ["median"]).map(lambda v: f"{v:.1e}")}).round(2).reset_index() \
        .rename(columns={"dom": "Domain", "q1": "Q1 (log10)", "q3": "Q3 (log10)", "median": "median (log10)"}).to_html(index=False, border=0)
    write_page(
        "violin.html", "Violin plot",
        f"Language and multimodal models train with roughly {round(lo, -1):.0f}–{round(hi, -2):.0f}× the compute of vision and biology models (medians)",
        "Distribution of estimated training compute for notable AI models released since 2018, by domain. Each dot is a model; "
        "the shape shows where models cluster, the box marks the median and quartiles.", fig,
        ["<b>Only groups with at least 30 models are drawn</b> (a violin of a small sample is mostly smoothing). Left out: "
         + ", ".join(f"{k} ({v})" for k, v in excluded.items()) + ".",
         "<b>Limit:</b> only models with a compute estimate appear (about half of Epoch&rsquo;s notable models), so this is not every model. "
         "Multi-domain models are assigned to Multimodal if listed, otherwise to their first-listed domain. The shape depends on the "
         "smoothing bandwidth (plot default); the density is cut at the data range.",
         "Every violin has the same area whatever its n, so read n from the axis label."], table)


# =================================================================================================
# index + shared assets
# =================================================================================================
def build_index():
    cards = [("timeline.html", "Timeline", "Frontier AI models by lab and release date: how many are open-weights?"),
             ("radar.html", "Radar chart", "An open-weights model against two closed models on six benchmarks."),
             ("violin.html", "Violin plot", "Training compute by AI domain: where models cluster.")]
    items = "".join(f"<a href='{h}'><b>{t}</b><span>{d}</span></a>" for h, t, d in cards)
    with open(os.path.join(OUT, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
                 f"<title>PEC 2 – Data Visualization</title><style>{CSS}</style></head><body><main>"
                 "<h1>PEC 2 – Data Visualization: the AI frontier</h1>"
                 "<p class='sub'>Three charts, one per assigned technique, built from open data by Epoch AI.</p>"
                 f"<div class='cards'>{items}</div><footer>{CREDIT}</footer></main></body></html>")


if __name__ == "__main__":
    build_timeline()
    build_radar()
    build_violin()
    build_index()
    # shared Plotly library (written once, referenced by every page) + switch off Jekyll on GitHub Pages
    shutil.copy(os.path.join(os.path.dirname(plotly.__file__), "package_data", "plotly.min.js"), os.path.join(OUT, "plotly.min.js"))
    open(os.path.join(OUT, ".nojekyll"), "w").close()
    print("built into", OUT)
