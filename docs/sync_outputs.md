# Sync experiment outputs

Experiment archives live on the GPU host under `runs/local/`. Copy them to your PC with `rsync`. Run the command in a terminal on your PC, from your local clone of this repository (the directory that contains `runs/`).

The integrated terminal of this workspace is already an SSH session on the host: the prompt is `st_mdp26@adalovelace`. Running the command there connects back to the same machine (`adalovelace` resolves to `127.0.0.1`) and writes into the folder it just read. Nothing is copied to your PC.

`adalovelace` is the name of this machine inside the cluster. It is not in public DNS. From a PC, use the `Host` alias already configured for Cursor. On this lab setup that alias is `OpenEnricher-Lab` (`HostName 10.79.23.33` in the SSH config).

Cursor on Windows reads `C:\Users\user\.ssh\config`. In the commands below, `user` is a placeholder: replace it with your Windows account name, the folder under `C:\Users\`. A Linux shell on the same PC, such as WSL, reads `~/.ssh/config` in its own home and does not see that alias. `rsync` then tries to resolve `OpenEnricher-Lab` as a DNS name and fails with `Could not resolve hostname`. Pass the Windows file with `ssh -F`:

```bash
rsync -avP \
  -e "ssh -F /mnt/c/Users/user/.ssh/config" \
  st_mdp26@OpenEnricher-Lab:/SLURM_shared/st_mdp26/open-enricher/runs/local/ \
  ./runs/local/
```

`-F` tells `ssh` which config file to read instead of the default `~/.ssh/config`. With that file, the alias `OpenEnricher-Lab` expands to `10.79.23.33` and the user `st_mdp26`. Confirm before copying:

```bash
ssh -G -F /mnt/c/Users/user/.ssh/config OpenEnricher-Lab | grep ^hostname
```

The line must be `hostname 10.79.23.33`. `hostname openenricher-lab` means `ssh` is still ignoring the Windows file. The lab address is on a private network, so the VPN has to be connected.

## From PowerShell

Run the command in PowerShell from the local clone. Windows OpenSSH reads `C:\Users\user\.ssh\config` by itself (`user` is again your Windows account name), so `-F` is not required. Pass that `ssh` to `rsync` explicitly. A `rsync` installed beside Git Bash otherwise calls a different `ssh`, which does not see the Cursor host alias.

```powershell
rsync -avP `
  -e "C:\Windows\System32\OpenSSH\ssh.exe" `
  st_mdp26@OpenEnricher-Lab:/SLURM_shared/st_mdp26/open-enricher/runs/local/ `
  ./runs/local/
```

In PowerShell the backtick continues the command on the next line. Confirm the alias before copying:

```powershell
ssh -G OpenEnricher-Lab | Select-String '^hostname'
```

The matching line is `hostname 10.79.23.33`. PowerShell does not ship `rsync`; the program has to be installed and on `PATH`. The VPN has to be connected, as in the Linux shell above.

The remote path is the source. `./runs/local/` is the destination, relative to your current directory. The trailing slash on both paths copies the contents of `local` into `./runs/local/`. A second `local` directory is not created.

- `-a` keeps timestamps and permissions.
- `-v` lists each transferred file.
- `-P` shows progress and keeps a partial file, so running the command again resumes an interrupted copy.
- `runs/local/_cache/` is part of the copy. It holds the shared audio and the scene frames.

`rsync` sends only files that are new or changed. A later run updates `./runs/local/` in place.

`dataset/outputs/` on the host is the live working copy used during a run. The loop archives the same JSON into `runs/local/<experiment_id>/`, which is the directory this command syncs. Video files, model weights, and `dataset/` stay on the host.
