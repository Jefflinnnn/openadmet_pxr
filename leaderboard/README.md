# Leaderboard experiment log

This folder tracks **what we submitted** and **how it performed** on the OpenADMET Activity Track leaderboard.

## Files

- `experiments.csv`: append-only log of submissions and scores.

## How to use

After you submit a file and get leaderboard metrics, append a row with:

```bash
python scripts/log_leaderboard.py \
  --submission-csv submissions/<file>.csv \
  --source <run_dir_or_description> \
  --model-family <chemeleon|analog|blend|...> \
  --targets <comma-separated> \
  --lb-mae <float> \
  --lb-r2 <float> \
  --lb-spearman <float> \
  --lb-kendall <float>
```

You can also include optional fields like `--local-cv-mae`, `--task-weights`, `--ensemble-desc`, etc.
