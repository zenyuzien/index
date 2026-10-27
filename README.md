# HAI - Healthcare Accessibility Index (Louisiana parishes)

    pip install -r requirements.txt
    python run_hai.py            # writes hai_dataset.csv

Files
- `run_hai.py`            driver: dataset -> rank-average index -> rotated-PCA index -> `hai_dataset.csv`
- `dataset.py`            builds the seven indicators from raw sources (sources are documented at the top of the file)
- `index_rank_average.py` equal-weight rank average index
- `index_rotated_pca.py`  varimax-rotated PCA aggregation index
- `data/`                 raw input files (copied from the project)
- `raw_cache/`            saved downloads (HRSA API, Census files) and OSRM drive times, so runs are offline and repeatable.
                          Delete a file in here and it is downloaded/recomputed again (needs internet; OSRM takes ~2 min).

Needs Python 3.9+ with numpy and pandas only. All paths are relative to this folder.
