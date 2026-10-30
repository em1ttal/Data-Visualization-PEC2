# PEC 2 – Data Visualization: the AI frontier

Three charts, one per assigned technique (timeline, radar chart, violin plot), built from open data by Epoch AI.
Published with GitHub Pages from the `docs/` folder.

| Chart | Question | Data |
|---|---|---|
| `docs/timeline.html` | How many frontier AI models are open-weights? | Epoch AI frontier models (+ 2026 restricted releases from the notable file) |
| `docs/radar.html` | How far does an open-weights model trail the closed flagship? | Epoch AI benchmark results, six Epoch-run benchmarks |
| `docs/violin.html` | Where do AI models cluster in training compute, by domain? | Epoch AI notable models with a compute estimate |

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python build.py            # writes the static site into docs/
python -m http.server -d docs 8000   # optional: preview at http://localhost:8000
```

## Publish on GitHub Pages
1. Push this repository to GitHub (public).
2. Settings, Pages, "Deploy from a branch", branch `main`, folder `/docs`, Save.
3. The site appears at `https://<your-user>.github.io/<repo-name>/` after a minute or two.

## Data preparation (all in `build.py`)
- **Timeline:** frontier models since 2015; lane = first listed organisation (Google, DeepMind, Brain and Research merged);
  open = downloadable weights, closed = API, hosted, limited access or unreleased; accessibility missing = "not recorded".
  Diamonds = 2026 models labelled limited access or unreleased in the notable file (OpenAI, Anthropic, Google only).
- **Radar:** six Epoch-run benchmarks (GPQA Diamond, Mock AIME, Chess puzzles, FrontierMath Tiers 1–3 v2, SimpleQA Verified,
  FrontierMath Tier 4 v2); axis order = median score over the 2026 models that have all six; three models chosen for distinct profiles.
- **Violin:** notable models since 2018 with a compute estimate; domain = Multimodal if listed, otherwise the first-listed domain;
  groups with fewer than 30 models are not drawn; compute shown on a log10 scale.

## Data and licence
Data: Epoch AI, "Data on AI Models" (`data/epoch/`) and "Capabilities & benchmarking" (`data/benchmarks/`), epoch.ai,
accessed 30 Oct 2026, CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Files are copied unchanged; the charts show
filtered and regrouped data. External benchmarks inside Epoch's hub keep their own licences; only Epoch-run benchmarks are used here.
Add a licence for your own code (for example MIT) if you want to share it.

## AI-use statement (complete before submitting)
Claude (Anthropic) helped with research, dataset checks, data profiling, the chart code and this README.
Tool, objectives, prompts, responses and how each output was reviewed and edited must be listed per section of the submission.
