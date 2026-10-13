"""Allow `python -m open_enricher.remote.server` via package entry.

Prefer: python -m open_enricher.remote.server
This module starts the worker when invoked as open_enricher.remote.
"""

from open_enricher.remote.server import main

main()
