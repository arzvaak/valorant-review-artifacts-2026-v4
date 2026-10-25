# Software map

`final_colab/` contains the nine source modules used for the bounded retrospective analyses and their local continuation. They are provided for inspection and are not self-contained without compatible, lawfully obtained phase inputs, frozen package artifacts, and the environment described in the methods. The release contains no data rows, row-level targets or forecasts, model binaries, or checkpoints.

`aggregate-renderers/` contains the reader-runnable commands that consume only the supplied aggregate CSVs. See the top-level README for exact commands. The score renderer uses the Python standard library; the figure renderers require Matplotlib.

`audit-renderers/` contains provenance and audit source for saved run reports that are not included here. These files are provided for inspection and are not needed to reproduce the shared aggregate tables or figures. `presentation_saved_outputs.py` requires explicit paths to restricted saved outputs and cannot run from this release.

The original source snapshot, feature dictionaries, schemas, dependency notices, and pinned aggregate reconstruction runtime are preserved in the repository root and its previous v4.1 artifacts; links are provided in the dated release README.
