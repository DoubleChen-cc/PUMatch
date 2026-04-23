# Plot Toolkit Overview

This folder contains the plotting and data-preparation pipeline for Fig9–Fig12.

## 1) Unified Data Convention

To reduce script-specific formats, the plotting pipeline now uses:

- A single Excel file: `plot/result.xlsx`
- One sheet per figure family:
  - `Sheet1` -> Fig9 + Fig11 data
  - `Sheet2` -> Fig10 data
  - `Sheet3` -> Fig12 data

### Raw benchmark result conventions

- **Single-GPU style result** (`result.txt`):
  - `<graph> <query> <time> <matches>`
  - Used by Fig9 preparation.
- **Distributed result** (`pu_result.txt`, `gamma_result.txt`):
  - `<graph> <query> t0 t1 ... t(n-1) <matches>`
  - Effective run time is `max(t0..t(n-1))`.
  - Used by Fig10 preparation.

## 2) Core Scripts

### Unified preparer

- `prepare_results.py`
  - `fig9` subcommand: writes `Sheet1`
  - `fig10` subcommand: writes `Sheet2`
- `prepare_fig11_input.py`
  - updates `Sheet1` fig11-specific columns from raw TSV
- `prepare_fig12_input.py`
  - builds `Sheet3` from transfer TSV + `Sheet1` time speedup

Compatibility wrappers (old entry names, still usable):

- `prepare_fig9_input.py` -> forwards to `prepare_results.py fig9`
- `prepare_fig10_input.py` -> forwards to `prepare_results.py fig10`

### Plot scripts

- `plot_fig9.py`   (reads `Sheet1`)
- `plot_fig10.py`  (reads `Sheet2`)
- `plot_fig11.py`  (reads `Sheet1`)
- `plot_fig12.py`  (reads `Sheet3`)

### Pipeline scripts

- `run_fig9_pipeline.py` / `.sh`
  - Run PUMatch + GAMMA + STMatch
  - Build `Sheet1`
  - Plot Fig9
- `run_fig10_pipeline.py` / `.sh`
  - Run distributed PUMatch + distributed GAMMA
  - Build `Sheet2`
  - Plot Fig10
- `plot_fig11.sh`
  - Run 3 configs: `Package_only`, `non-prefetch`, `fixed D a`
  - Generate raw TSV + call `prepare_fig11_input.py`
  - Plot Fig11
- `plot_fig12.sh`
  - Nsight Systems data-transfer analysis for STMatch_UM and PUMatch
  - Generate raw TSV + call `prepare_fig12_input.py` (build `Sheet3`)
  - Plot Fig12

## 3) Typical Usage

From project root:

```bash
# Fig9
python3 plot/run_fig9_pipeline.py --graph <graph_path> --clear-result

# Fig10
python3 plot/run_fig10_pipeline.py --graph <graph_path> --clear-result

# Fig11
bash plot/plot_fig11.sh --graph <graph_path>

# Fig12
bash plot/plot_fig12.sh --graph <graph_path>
```

## 4) Dependencies

- Python: `pandas`, `numpy`, `matplotlib`, `openpyxl`
- For Fig12 profiling: `nsys` (Nsight Systems CLI)

Install Python deps:

```bash
pip install pandas numpy matplotlib openpyxl
```

## 5) Notes

- Keep `result.xlsx` in `plot/`; scripts assume this location.
- Wrapper scripts are retained for backward compatibility.
- If you add new figure scripts, prefer:
  - writing to a dedicated sheet
  - reusing `prepare_results.py` when possible
  - avoiding ad-hoc temporary formats.

