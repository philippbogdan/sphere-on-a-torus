# Figure data

`throughput.csv` contains modelled speed ratios against a polar-limited lat-lon baseline. `filter_trade.csv` holds a separate optimistic same-kernel comparison with a whole-latitude polar filter. Do not combine their denominators.

`accuracy.csv` contains all 18 paired height-l2 ratios, with each raw error and cell count. `spectral.csv` gives the TC6 day-14 spectral comparison. `stable_step.csv` contains the three measured TC5 stability limits. `model.json` holds the kernel and patch costs and the model calibration values.

The data is rebuilt from the saved measurements by `scripts/prepare_figures.py`. The plotting script reads only this pack. Missing or untested results have not been interpolated into it. Speeds are modelled; accuracy and stability scores come from numerical runs.
