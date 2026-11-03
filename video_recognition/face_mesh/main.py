"""Convenience shim so ``python main.py ...`` still works without installing.

The real CLI lives in ``facemesh/cli.py``. After ``pip install .`` you can also
run the ``facemesh`` command or ``python -m facemesh``.
"""
from facemesh.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
