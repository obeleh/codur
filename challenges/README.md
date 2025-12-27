# Challenges

This folder contains small, self-contained coding challenges used to validate Codur changes.

## Structure

Each challenge lives in its own subfolder and usually contains:

- `prompt.txt` - the instruction sent to Codur
- `expected.txt` - expected stdout from the entry point
- `main.py` or `app.py` (or a single .py file) - the entry point Codur edits

Keep challenge folders minimal. If you add extra files, commit them and ensure the harness still passes.

## How it works

The test harness:

1. Runs Codur in the challenge directory using `prompt.txt`.
2. Executes the entry point and compares stdout to `expected.txt`.
3. Writes `codur_debug.log` in the challenge folder for inspection.

## Running the tests

From the repo root:

```bash
pytest tests/with_several_llm_calls/test_challenges.py
```

## Running a challenge manually

From the repo root:

```bash
cd challenges/01-fix-off-by-onerror
python -m codur.cli --command "$(cat prompt.txt)" --max-llm-calls 10 --raw --config ../../codur.yaml --verbose --fail-early
python main.py
```

## Reset behavior

The test harness resets files under `challenges/` back to the git state after each run and cleans up `codur_debug.log`. Untracked files under `challenges/` (except this README) will fail the tests.
