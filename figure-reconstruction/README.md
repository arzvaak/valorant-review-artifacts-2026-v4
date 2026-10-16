# Figure and aggregate reconstruction

From this directory, run the pinned aggregate-only reconstruction:

```sh
docker compose run --build --rm render
```

The default output directory is a new `./reconstructed/` tree. The bundled `outputs-v3/` is retained as a read-only qualified reference and is not mounted over. For repeat runs, choose a fresh empty destination, for example:

```sh
VALORANT_RENDER_OUTPUT=./reconstructed-v4.1b docker compose run --build --rm render
```

The command exports grouped records to CSV, renders figures from the included aggregate JSON, writes the secondary-metric presentation table, and stores a pathless input/code/output receipt in the selected output directory.

The v3-qualified plotting code, projected aggregate inputs, dependency lock and carried-forward output tree are unchanged. The v4.1 Compose file changes only the host output bind so a fresh reconstruction can run without overwriting the bundled reference. One bounded v4.1 aggregate-only reconstruction completed; see `reconstruction-qualification-v4.1.json`. No new models, scores, intervals or resamples were produced. The renderer processes only included grouped aggregate inputs and does not access source rows.

Figure qualification establishes source-record/caption equivalence, not raster/vector byte identity across environments. Scientific refitting requires compatible lawfully obtained phase inputs, CUDA-enabled PyTorch and a complete matched runtime; none is included here.
