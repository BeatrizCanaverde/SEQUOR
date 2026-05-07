import jsonargparse
import jinja2
import os
import json

from dataclasses import asdict, dataclass, field
from pathlib import Path

from typing import Optional



@dataclass(frozen=True)
class JobConfig:
    run_dir: str
    testset_dir: str
    output_dir: str
    model_path: str
    log_dir: Optional[str] = None
    judge_dir: Optional[str] = None
    api_key: str = "None"
    tp_size: int = 1
    dp_size: int = 1
    max_turns: int = 20
    max_tokens: int = 32768  # 16384  # 8192
    batch_size: int = 50
    max_concurrent: int = 100
    context_length: int = 131072
    health_check_max_wait_minutes: int = 120
    health_check_interval_seconds: int = 15


@dataclass(frozen=True)
class EnvConfig:
    env_vars: dict[str, str] = field(default_factory=dict)


def main(
    job: JobConfig,
    env: EnvConfig,
    no_submit: bool = False,
):
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    jinja_env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(scripts_dir),
        undefined=jinja2.StrictUndefined,
    )
    script = "submit_generate_evals.sh"
    template = jinja_env.get_template(f"{script}.j2")

    run_dir = Path(job.run_dir)
    # Create run directory
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        **asdict(job),
        **asdict(env),
    }

    # Path form of output_dir for creating subdirectories
    output_dir = Path(cfg["output_dir"])

    # Calculate total GPUs needed: gres = tp_size * dp_size
    # If gres is explicitly provided, verify it matches
    tp_size = cfg.get("tp_size", job.tp_size)
    dp_size = cfg.get("dp_size", job.dp_size)
    calculated_gres = tp_size * dp_size
    
    gres_val = cfg.get("gres")
    if gres_val is not None:
        # User explicitly provided gres, validate it matches tp_size * dp_size
        try:
            provided_gres = int(gres_val)
            if provided_gres != calculated_gres:
                print(f"Warning: Provided gres={provided_gres} doesn't match tp_size({tp_size}) * dp_size({dp_size}) = {calculated_gres}")
                print(f"Using calculated value: {calculated_gres}")
        except Exception:
            pass
    
    cfg["gres"] = str(calculated_gres)
    cfg["tp_size"] = tp_size
    cfg["dp_size"] = dp_size

    # Default judge/log dirs to output_dir if not provided
    if not cfg.get("judge_dir"):
        cfg["judge_dir"] = str(output_dir / "judge")
    if not cfg.get("log_dir"):
        cfg["log_dir"] = str(output_dir / "logs")

    # Create required directories
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["judge_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["log_dir"]).mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    script_path = output_dir / script

    with open(script_path, "w", encoding="utf-8") as f:
        f.write(template.render(cfg))

    # Make script executable
    os.chmod(script_path, 0o755)

    if not no_submit:
        os.system(f"bash {script_path}")
    

if __name__ == "__main__":
    jsonargparse.CLI(main, as_positional=False)
