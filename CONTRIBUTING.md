# Contributing to HPCcoWorklab

Thanks for considering it. This project is early and built in the open on
purpose — cluster quirks and real-world testing from other institutions
are exactly what make it better.

## Ways to contribute

- **A config file for your own cluster** — see [`hpccoworklab/configs/libra.yaml`](hpccoworklab/configs/libra.yaml)
  for the full shape. If your cluster has partitions, limits, or rules
  hpccoworklab doesn't yet model well, that's useful signal even without a
  code change.
- **New checks** — parallelism patterns, common mistakes on your own
  cluster, anything in the [README's roadmap](README.md#the-idea) that
  isn't built yet.
- **Bug reports** — especially false positives/negatives found on a real
  cluster. Include the (sanitized) job script and what you expected vs.
  what hpccoworklab said.
- **Real-world testing** — the monitoring (`watch`) and diagnosis
  (`diagnose`) commands are built against realistic sample Slurm output,
  but haven't been verified against every Slurm version's exact output
  format. If something parses wrong on your cluster, that's valuable to
  know.

## Before submitting a PR

- Open an issue first for anything beyond a small fix, so we can agree on
  direction before you put in the work.
- Run the test suite (`python3 -m pytest tests/ -v`) and make sure it's
  green.
- If you're adding a check, add a test for it — see `tests/test_hpccoworklab.py`
  and `tests/test_monitor.py` for the pattern: pure logic functions tested
  against fixture text/data, so tests don't require a live Slurm cluster.
- Match the existing tone in error messages: explain *why* something's
  flagged, not just that it is.

## Development setup

```bash
git clone https://github.com/vaniwalvekar/hpccoworklab.git
cd hpccoworklab
pip install -e ".[dev]"
python3 -m pytest tests/ -v
```
