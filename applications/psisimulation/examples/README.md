# Examples

## GSS 2018 progressive-choice example

`gss2018_choices.json` applies two persistent hard observations:

1. column 0 = `"working fulltime"`
2. column 3 = `"yes"`

Each answer is followed by five centered-LDP mode sweeps. These labels were
verified against the GSS-2018 native model during development.

Run from the repository root:

```bash
python3 applications/psisimulation/simulate_progressive.py \
  --model gss/gss_2018 \
  --choices applications/psisimulation/examples/gss2018_choices.json \
  --threads "$(nproc)" \
  --gif
```

To construct a different sequence, first inspect the alphabet at the empty
state:

```bash
python3 applications/psisimulation/inspect_column.py \
  --model gss/gss_2018 \
  --column 0
```

Then copy `gss2018_choices.json` and replace the column/value pairs.
