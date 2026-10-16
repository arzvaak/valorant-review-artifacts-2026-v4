# Dependency notices in the local plotting reconstruction candidate

The `requirements-plotting.lock` pins the observed Python 3.12 Linux/amd64 wheel versions and SHA-256 hashes. License metadata was read from installed distribution `METADATA`; the raw notices below are copied from those installed distributions. This is an inventory, not a legal clearance. The final container image and any redistributed binaries still need their full applicable notices.

| Distribution | Version | Observed license metadata / notice |
|---|---:|---|
| contourpy | 1.4.0 | BSD-3-Clause |
| cycler | 0.12.1 | metadata supplies project copyright; license notice copied |
| fonttools | 4.66.1 | MIT; external notice copied |
| kiwisolver | 1.5.1 | license text copied; normalized SPDX not assigned |
| matplotlib | 3.10.8 | Matplotlib license text copied; DejaVu and STIX notices copied |
| numpy | 2.5.3 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause |
| pillow | 12.3.0 | MIT-CMU |
| pyparsing | 3.3.3 | MIT |
| python-dateutil | 2.9.0.post0 | dual license; text copied |
| six | 1.17.0 | MIT |

Copied notices are under `python-distributions/<distribution>.dist-info/`; Matplotlib bundled-font notices are under `matplotlib-3.10.8/bundled-fonts/`. The three third-party license files containing an email-like string are external package notice text, not submitter identity. `SET_TRANSFORMER_LICENSE` covers only the upstream Set Transformer contribution; the project-authored code still has no root license.

The v2 aggregate/figure reconstruction uses the same pinned renderer environment as v1: `python:3.12.14-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e`, `linux/amd64`, the Python standard library, and the plotting distributions listed above. No PyTorch, CUDA runtime, model binary, or model-refit dependency lock is bundled. The scientific refit sources import CUDA-enabled PyTorch and NumPy and require their compatible historical phase inputs; the full refit environment and applicable notices have not been inventoried or qualified here.
