"""Module entrypoint: ``python -m open_enricher.experiments``.

Thin alias of the package CLI so either module path starts the local loop.
"""

from open_enricher.experiments.loop import main

if __name__ == "__main__":
    main()
