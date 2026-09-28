# hpclint modulefile (draft)

This is a personal-testing draft, not a DST-supported install yet.

## To try it yourself on Libra

1. Create a dedicated virtual environment (don't touch shared/system Python):
   ```bash
   python3 -m venv ~/.local/hpclint-venv
   ~/.local/hpclint-venv/bin/pip install hpclint  # or pip install -e . from a repo clone
   ```

2. Copy this project's Libra config into that environment's folder so the
   modulefile's `HPCLINT_DEFAULT_CONFIG` path resolves:
   ```bash
   mkdir -p ~/.local/hpclint-venv/configs
   cp configs/libra.yaml ~/.local/hpclint-venv/configs/
   ```

3. Add this repo's `modulefiles/` directory to your personal module path:
   ```bash
   module use /path/to/hpclint/modulefiles
   module load hpclint
   ```

4. Test it — no `--config` needed now, since it's picked up automatically:
   ```bash
   hpclint check some_script.sh
   ```

