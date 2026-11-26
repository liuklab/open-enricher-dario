# Copy results from the cluster

The active experiment directory is `runs/local/<experiment_id>/<video>/` unless
`OPEN_ENRICHER_RUNS_ROOT` selects another root. Completed historical experiments
on the original cluster are under `local-archive/` in the main checkout. The
archive index gives their directory and run provenance.

Run rsync on the destination computer. Substitute the SSH host alias, remote
repository path, experiment ID and episode filename for your installation.

```bash
rsync -avP 'cluster:/path/to/open-enricher/runs/local/12-combined/suits_s01e02.mkv/' ./results/suits_s01e02/
```

An episode directory contains the stage JSONs, timings and, for combined runs,
the manifest and frame index. JPEGs used by a historical run can live separately;
use the archive index to locate its input evidence. Keep video, licensed frames
and local evaluation data in access-controlled storage.
