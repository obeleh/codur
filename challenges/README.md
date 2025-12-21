# Challenges

This folder contains small, self-contained coding challenges intended for Codur to solve.

## Structure
Each challenge lives in its own subfolder and must contain exactly these files:
- `prompt.txt` — the instruction sent to Codur
- `main.py` — the executable harness that prints a deterministic output
- `expected.txt` — the exact expected stdout from `main.py`

No other files should exist in a challenge folder.

## How it works
The test suite runs `codur` with the challenge prompt (using the challenge folder as the working directory). After Codur makes changes, the suite executes `main.py` and compares its output to `expected.txt`.

## Running the tests
From the repo root:
```
pytest tests/test_challenges.py
```

## Running a challenge manually
From the repo root, you can run a single challenge like this:
```
cd challenges/01-fix-off-by-onerror
python -m codur.cli --command "$(cat prompt.txt)" --raw --config ../../codur.yaml --verbose
python main.py
```

## Reset behavior
The test harness resets files under `challenges/` back to the git state after each run to keep the repo clean.
