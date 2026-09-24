"""Networking strategies for transferring files and executing remote commands."""

import subprocess
import sys
from pathlib import Path
from typing import Protocol

REMOTE_TARBALL_PATH = "/content/open-enricher.tgz"
REMOTE_ENV_PATH = "/content/.env"

def _run_cmd(cmd: list[str], input_data: str | None = None) -> None:
    print(f"[RUN] Running: {' '.join(cmd)}")
    try:
        stdin_bytes = input_data.encode("utf-8") if input_data else None
        
        subprocess.run(cmd, input=stdin_bytes, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed with exit code {e.returncode}.")
        sys.exit(1)

class DeployStrategy(Protocol):
    """Interface for all deployment network strategies (Strategy Pattern)."""
    def deploy(self, tarball_path: Path, env_path: Path, target: str, remote_script: str) -> None:
        ...

class TailscaleStrategy:
    """Deploys over a direct Tailscale SSH connection."""
    def deploy(self, tarball_path: Path, env_path: Path, target: str, remote_script: str) -> None:
        print(f"[INFO] Deploying via Tailscale SSH to {target}...")
        remote_host = f"root@{target}"
        
        # 1. Invia il codice sorgente
        _run_cmd(["scp", "-o", "StrictHostKeyChecking=no", str(tarball_path), f"{remote_host}:{REMOTE_TARBALL_PATH}"])
        
        # 2. Invia i segreti (se il file .env esiste)
        if env_path.exists():
            print("[INFO] Trovato file .env locale: invio sicuro dei segreti al server...")
            _run_cmd(["scp", "-o", "StrictHostKeyChecking=no", str(env_path), f"{remote_host}:{REMOTE_ENV_PATH}"])
        else:
            print("[WARNING] Nessun file .env locale trovato. Deploy senza segreti extra.")
            
        # 3. Lancia lo script remoto
        _run_cmd(["ssh", "-o", "StrictHostKeyChecking=no", remote_host, "bash", "-s"], input_data=remote_script)

class ColabCliStrategy:
    """Deploys utilizing Google Colab CLI tunneling."""
    def deploy(self, tarball_path: Path, env_path: Path, target: str, remote_script: str) -> None:
        print(f"[INFO] Deploying via Colab CLI to session '{target}'...")
        
        # 1. Invia il codice sorgente
        _run_cmd(["colab", "upload", "-s", target, str(tarball_path), REMOTE_TARBALL_PATH])
        
        # 2. Invia i segreti (se il file .env esiste)
        if env_path.exists():
            print("[INFO] Trovato file .env locale: invio sicuro dei segreti al server...")
            _run_cmd(["colab", "upload", "-s", target, str(env_path), REMOTE_ENV_PATH])
        else:
            print("[WARNING] Nessun file .env locale trovato. Deploy senza segreti extra.")
        
        # 3. Lancia lo script remoto
        script_path = tarball_path.parent / "launch.sh"
        script_path.write_text(remote_script, encoding="utf-8")
        
        _run_cmd(["colab", "upload", "-s", target, str(script_path), "/content/launch.sh"])
        _run_cmd(["colab", "exec", "-s", target, "--", "bash", "/content/launch.sh"])