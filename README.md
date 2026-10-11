# Computational Fluid Dynamics

Lecture notes and exercises for the Computational Fluid Dynamics course at TU Dresden.

The course website is authored as Quarto-compatible Python notebooks in `notebooks/` and rendered to `docs/`:

```bash
python -m venv .venv-quarto
source .venv-quarto/bin/activate
python -m pip install -r requirements.txt -r requirements-quarto.txt
export QUARTO_PYTHON="$PWD/.venv-quarto/bin/python"
quarto render
quarto preview --port 8002
```

See [authoring.qmd](authoring.qmd) for the full workflow. Historical PDF material remains in `old_material/` as a migration reference.

The [course website](https://andreweiner.github.io/cfd-lecture/) is deployed by GitHub Actions on pushes to `main`. Rendered `docs/` files remain ignored; `_freeze/` results are versioned so unchanged notebook computations can be reused. Local figure regeneration requires LaTeX and dvipng. The separate slides repository is not yet published.
