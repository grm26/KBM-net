
## Run

```bash
python main.py --dataset TADPOLE --task SMCI_PMCI
python main.py --dataset TADPOLE --task AD_CN_SMCI_PMCI
```

Useful options:

```bash
python main.py \
  --dataset TADPOLE \
  --task SMCI_PMCI \
  --epochs 500 \
  --n-splits 10 \
  --selection-metric bac \
  --save-model
```

The result files are written to `RESULT/<dataset>/<task>/`.

## Dependencies

- Python 3.10+
- PyTorch 2.1+
- `mamba-ssm`
- `einops`
- NumPy
- pandas
- scikit-learn
- tqdm
