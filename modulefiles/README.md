# hpclint modulefile (draft)

This is a personal-testing draft, not a DST-supported install yet.

## To try it yourself on Libra

1. Create a dedicated virtual environment (don't touch shared/system Python):
   ```bash
   python3 -m venv ~/.local/hpclint-venv
   ~/.local/hpclint-venv/bin/pip install hpclint   # or pip install -e . from a clone
   ```
   Cluster configs ship inside the package (`hpclint/configs/`), so there is
   nothing to copy manually.

2. Add this repo's `modulefiles/` directory to your personal module path and load:
   ```bash
   module use /path/to/hpclint/modulefiles
   module load hpclint
   ```
   The modulefile sets `HPCLINT_DEFAULT_CONFIG=libra`, which hpclint resolves
   from the packaged config, so `--config` is unnecessary.

3. Use it:
   ```bash
   hpclint check some_script.sh      # uses the bundled libra.yaml
   ```

