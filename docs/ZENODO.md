# Zenodo Archive

Software archive for citation and long-term preservation.

## DOI

| Type | DOI | Link |
|------|-----|------|
| **Concept (always latest)** | [10.5281/zenodo.20495459](https://doi.org/10.5281/zenodo.20495459) | https://zenodo.org/records/20495459 |
| Release v1.0.0 | [10.5281/zenodo.20495460](https://doi.org/10.5281/zenodo.20495460) | https://zenodo.org/records/20495460 |

New GitHub releases (e.g. `v0.2.0`) receive a version-specific DOI under the same concept record when the [Zenodo–GitHub integration](https://zenodo.org/account/settings/github/) is enabled for `keisuke58/wafer-proc-sim`.

## Publish a new release on Zenodo

1. Create a GitHub release (tag + notes), e.g. `gh release create v0.2.0 …`
2. Open https://zenodo.org/account/settings/github/ and confirm the repo is ON
3. After a few minutes, check https://zenodo.org/deposit — a draft may appear
4. Review metadata (auto-filled from `.zenodo.json` + `CITATION.cff`) and click **Publish**
5. Update `CITATION.cff` `version` and `doi` if Zenodo assigns a new version DOI

## Cite

Prefer the **concept DOI** in papers so readers resolve to the latest version:

```bibtex
@software{nishioka2026wafer,
  author    = {Nishioka, Keisuke},
  title     = {wafer-proc-sim},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20495459},
  url       = {https://github.com/keisuke58/wafer-proc-sim}
}
```

For the *Precision Engineering* paper, cite both the manuscript and this software archive.