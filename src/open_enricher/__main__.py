"""Package entrypoint: ``python -m open_enricher``.

Delegates to the local sequential experiment loop. Install the package in
editable mode (``pip install -e .``) or set ``PYTHONPATH=src`` so the
``open_enricher`` package resolves.
"""

from open_enricher.experiments.loop import main

if __name__ == "__main__":
    main()
