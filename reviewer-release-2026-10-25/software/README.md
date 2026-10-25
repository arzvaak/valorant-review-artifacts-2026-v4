# Software map

`final_colab/` contains the nine source modules for the bounded retrospective execution and local continuation. They are included as source and are not self-contained without the corresponding compatible, lawful phase inputs, frozen package artifacts, and environment recorded in the methods. No data rows, row-level targets or forecasts, model binaries, or checkpoints are included.

`aggregate-renderers/` contains the reader-runnable commands that consume only the supplied aggregate CSVs. See the top-level README for exact commands. The score renderer uses the Python standard library; the figure renderers require Matplotlib.

`audit-renderers/` contains provenance/audit source that normally reads restricted saved run reports and forecasts. It is distributed for code inspection only and is not required to reproduce any shared aggregate table or figure. `presentation_saved_outputs.py` requires explicit paths to restricted frozen later outputs; it cannot run from this release.

The original source snapshot, feature dictionaries, schemas, dependency notices, and pinned aggregate reconstruction runtime are preserved in the repository root and its previous v4.1 artifacts; links are provided in the dated release README.
