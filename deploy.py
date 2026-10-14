"""Entrypoint to execute the remote deployment."""

from pathlib import Path

from deployment.cli_parser import parse_args
from deployment.packager import ProjectPackager
from deployment.network import ColabCliStrategy, DeployStrategy, TailscaleStrategy

def main() -> None:
    args = parse_args()
    
    root_dir = Path(__file__).resolve().parent
    tarball_path = root_dir / "open-enricher.tgz"
    script_path = root_dir / "deployment" / "launch.sh"
    
    # 0. Cerchiamo il file .env locale
    env_path = root_dir / ".env"
    
    # 1. Read the bash script
    remote_script = script_path.read_text(encoding="utf-8").replace("\r\n", "\n")
    
    # 2. Package the project
    packager = ProjectPackager(root_dir)
    packager.create_tarball(tarball_path)
    
    # 3. Select the network strategy
    strategy: DeployStrategy
    if args.mode == "tailscale":
        strategy = TailscaleStrategy()
    else:
        strategy = ColabCliStrategy()
        
    # 4. Execute deployment
    try:
        # Passiamo anche env_path alla strategia
        strategy.deploy(tarball_path, env_path, args.target, remote_script)
    finally:
        if tarball_path.exists():
            tarball_path.unlink()

if __name__ == "__main__":
    main()