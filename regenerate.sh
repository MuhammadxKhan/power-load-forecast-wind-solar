#!/usr/bin/env bash
# Regenerate every committed result, in the order the README reports them.
# Roughly 20 minutes: three models, four weather modes, two targets, four folds.
set -e
PY="C:/Users/mikah/AppData/Local/Python/pythoncore-3.14-64/python.exe"

# Order matters: every run writes results/scores.csv and results/predictions.csv,
# so the run that should end up committed there has to go last. The headline
# table is the no-weather run, so that one goes after the two weather runs.
echo "=== 1/5  demand, weather figures ==="
"$PY" run_comparison.py --weather noisy

echo "=== 2/5  demand, rolling-origin backtest ==="
"$PY" run_comparison.py --weather noisy --backtest --no-plots

echo "=== 3/5  demand, headline table (no weather) - writes scores.csv last ==="
"$PY" run_comparison.py --weather none --no-plots

echo "=== 4/5  solar ==="
"$PY" run_comparison.py --target solar --weather clearsky

echo "=== 5/5  ablations for ANALYSIS.md ==="
"$PY" ablations.py

echo "=== done ==="
