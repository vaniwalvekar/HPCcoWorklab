# hpccoworklab modulefile (draft)

This is a personal-testing draft, not a DST-supported install yet.

## To try it yourself on Libra

1. Create a dedicated virtual environment (don't touch shared/system Python):
   ```bash
   python3 -m venv ~/.local/hpccoworklab-venv
   ~/.local/hpccoworklab-venv/bin/pip install hpccoworklab   # or pip install -e . from a clone
   ```
   Cluster configs ship inside the package (`hpccoworklab/configs/`), so there is
   nothing to copy manually.

2. Add this repo's `modulefiles/` directory to your personal module path and load:
   ```bash
   module use /path/to/hpccoworklab/modulefiles
   module load hpccoworklab
   ```
   The modulefile sets `HPCCOWORKLAB_DEFAULT_CONFIG=libra`, which hpccoworklab resolves
   from the packaged config, so `--config` is unnecessary.

3. Use it:
   ```bash
   hpccoworklab check some_script.sh      # uses the bundled libra.yaml
   ```

