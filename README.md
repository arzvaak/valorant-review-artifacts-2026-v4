# Review revision software and aggregate artifact v4.1

This public artifact contains project-authored scientific helper source, grouped aggregate results and schemas, figure/table specifications and outputs, and a pinned aggregate-only reconstruction environment. It is separate from the local double-anonymous review archive; no manuscript or supplement is included here.

## Included material

- `software/revision-20261016/`: current scientific helper source plus historical and execution source identity records.
- `aggregate-review/revision-2026-10-16/`: methods and findings supporting this revision, with portable relative references.
- `figure-reconstruction/package/`: allowlisted grouped JSON, presentation-only renderer/exporter, and aggregate schemas.
- `figure-reconstruction/outputs-v3/`: grouped CSV tables and figures, including 40 retuned-removal contrasts, 16 denominator groups, 160 fit-package summaries, and 104 secondary-metric records.
- `schemas/` and `notices/`: field dictionaries/layouts and preserved third-party notices.

No raw source responses, row-level targets or labels, row-level forecasts, prediction arrays, keys, checkpoints, model binaries, private correspondence, credentials, internal authorization records, or protected data are included. Grouped outputs retain only aggregate grain used in the manuscript.

## Reconstruct the aggregate tables and figures

From this repository root run:

```sh
cd figure-reconstruction
docker compose run --build --rm render
```

The pinned Python 3.12.14 Linux/amd64 environment and hash-pinned plotting dependencies are in this directory. The v3-qualified aggregate inputs, renderer, dependency lock and carried-forward `outputs-v3/` tree are unchanged. The v4.1 Compose output bind writes to a fresh `reconstructed/` directory rather than overlaying the reference tree; one bounded aggregate-only reconstruction completed and is separately receipted. See `figure-reconstruction/v4.1-qualification-receipt.json` and `figure-reconstruction/outputs-v3/reconstruction-receipt.json`. Figure qualification establishes source-record/caption identity, not cross-environment image-byte identity.

This command uses included grouped records only. It does not refit models, rescore forecasts, recalculate intervals, resample, or access source rows. Standalone scientific refitting requires compatible lawfully obtained phase inputs, CUDA-enabled PyTorch and a complete matching runtime, none of which is provided here.

## License and rights scope

Project-authored code is offered under MIT; authored methods/findings documents, schemas, grouped aggregate tables and figures are offered under CC BY 4.0 to the extent the project owner holds relevant rights. See `LICENSE`, `LICENSES/README.md` and `LICENSES/CC-BY-4.0.txt`. Owner authorization for sharing and license selection was recorded on 2026-10-17; that is not independent verification of data acquisition, ownership or third-party copyrights. Third-party notices remain in `notices/` and take precedence for their components.
