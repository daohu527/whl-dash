**Whl-Dash**

Lightweight browser-based data analysis dashboard for visualizing and comparing autonomous-driving `.record` data (built with Dash + Plotly).

**Quick Start**
- **Install (development)**:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e .
```

- **Run (installed console script)**:

```bash
# After installing the package you can run the console script
whl-dash
# Or run in-place while developing
python3 -m whl_dash.main
```

**Usage**
- Open `http://127.0.0.1:8050` in your browser after the server starts.
- Use the left panel to choose or scan a directory containing `.record` files (Data Source), configure templates on the right, and click `Render Canvas`.
- Note: the CLI `--record_path` option has been removed — choose workspaces/records via the UI.

**Performance Optimizations**
- The app automatically down-samples very long time series and map trajectories for interactive performance, and switches to WebGL-backed traces (`Scattergl`) for large traces.
- For more aggressive down-sampling, adjust `max_points` in `whl_dash/dashboard.py` or implement multi-resolution caching in `RecordDataLoader`.

**Packaging & Publishing**
- The project uses `pyproject.toml` (setuptools) for builds.
- A GitHub Actions workflow `.github/workflows/python-publish.yml` is included to build and publish distributions when a Release is created — ensure `PYPI_API_TOKEN` is configured in repository Secrets.

**Troubleshooting**
- If you see `ModuleNotFoundError: No module named 'dashboard'`, reinstall the package in your virtualenv (`python3 -m pip install --upgrade .`) and run the installed `whl-dash` script; the package uses package-qualified imports (`whl_dash.*`) to avoid collisions.

**Developer Notes**
- Key modules: `whl_dash/dashboard.py` (UI & callbacks), `whl_dash/data.py` (RecordDataLoader), `whl_dash/template.py` (template management).
- When changing dependencies or CI, update `pyproject.toml` and `.github/workflows/python-publish.yml` accordingly.
