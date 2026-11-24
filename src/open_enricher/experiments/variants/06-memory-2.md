# 06-memory-2

This variant is the fixed-window control for the optional encoders. It uses
300-second windows. The VLM receives the current dialogue description and
metadata for up to two preceding windows. The first window has no prior records.

The historical `02-memory-2` configuration had the same inference settings.
Its original configuration, notes and outputs are in the local experiment archive.
Use `--only 06-memory-2` for a new run of this control.
