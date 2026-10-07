import logging
import os
import sys
from pathlib import Path
from typing import Dict


logger = logging.getLogger(__name__)

_RUNTIME_CONFIGURED = False


def _workspace_root() -> Path:
    env_root = os.environ.get("WORKSPACE_DIR") or os.environ.get("PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def _prepend_sys_path(path: Path) -> None:
    path_str = str(path)
    if path.is_dir() and path_str not in sys.path:
        sys.path.insert(0, path_str)


def configure_hpc_runtime() -> None:
    global _RUNTIME_CONFIGURED
    if _RUNTIME_CONFIGURED:
        return

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    workspace_dir = _workspace_root()
    hf_home = Path(os.environ.setdefault("HF_HOME", str(workspace_dir / "hf_cache")))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home))
    os.environ.setdefault("XDG_CACHE_HOME", str(hf_home))
    os.environ.setdefault("TORCH_HOME", str(hf_home))
    os.environ.setdefault("OFFLOAD_DIR", str(workspace_dir / "offload_dir"))

    if os.environ.get("HPC_OFFLINE", "1") == "1":
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    devroot = Path(os.environ.get("DEVROOT", str(workspace_dir / "dev"))).expanduser()
    _prepend_sys_path(devroot / "hf_deps")
    _prepend_sys_path(devroot / "transformers" / "src")

    try:
        import huggingface_hub.utils._validators as _hf_validators

        _hf_validators.validate_repo_id = lambda *a, **k: None
    except Exception:
        logger.debug("huggingface_hub validator patch unavailable", exc_info=True)

    try:
        import transformers.utils.hub as _tf_hub

        _tf_hub.validate_repo_id = lambda *a, **k: None
    except Exception:
        logger.debug("transformers validator patch unavailable", exc_info=True)

    _RUNTIME_CONFIGURED = True


def _sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "--")


def _resolve_snapshot_dir(cache_root: Path) -> Path | None:
    refs_main = cache_root / "refs" / "main"
    if refs_main.exists():
        snapshot_id = refs_main.read_text().strip()
        snapshot_dir = cache_root / "snapshots" / snapshot_id
        if snapshot_dir.exists():
            return snapshot_dir.resolve()

    snapshots_dir = cache_root / "snapshots"
    if snapshots_dir.exists():
        snapshot_dirs = sorted((p for p in snapshots_dir.iterdir() if p.is_dir()), key=lambda p: p.name)
        if snapshot_dirs:
            return snapshot_dirs[-1].resolve()

    return None


def resolve_model_source(model_name: str) -> str:
    configure_hpc_runtime()

    model_path = Path(model_name).expanduser()
    if model_path.exists():
        return str(model_path.resolve())

    local_override = os.environ.get(f"MODEL_PATH_{_sanitize_model_name(model_name).upper().replace('-', '_')}")
    if local_override and Path(local_override).expanduser().exists():
        return str(Path(local_override).expanduser().resolve())

    search_roots = [
        Path(os.environ["HF_HOME"]),
        _workspace_root(),
    ]
    cache_dir_name = f"models--{_sanitize_model_name(model_name)}"
    for root in search_roots:
        local_repo = root / cache_dir_name / "local-repo"
        if local_repo.exists():
            return str(local_repo.resolve())

        cache_root = root / cache_dir_name
        if cache_root.exists():
            snapshot_dir = _resolve_snapshot_dir(cache_root)
            if snapshot_dir is not None:
                return str(snapshot_dir)
            return str(cache_root.resolve())

    return model_name


def build_hf_load_args(model_name: str, model_type: str = "default") -> Dict[str, object]:
    configure_hpc_runtime()
    # Created here, when a model is actually about to be loaded, rather than in
    # configure_hpc_runtime(): importing the analysis modules must not leave
    # cache directories behind.
    Path(os.environ["OFFLOAD_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)

    resolved_source = resolve_model_source(model_name)
    load_args: Dict[str, object] = {
        "local_files_only": os.environ.get("HF_HUB_OFFLINE", "0") == "1",
    }

    if model_type == "auth":
        token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if token:
            load_args["token"] = token
            load_args["use_auth_token"] = token
        elif resolved_source == model_name:
            raise ValueError(
                f"Authentication token required for {model_name} and no local mirror was found."
            )

    import torch

    if torch.cuda.is_available():
        load_args["torch_dtype"] = torch.bfloat16
        load_args["low_cpu_mem_usage"] = True
        device_map = os.environ.get("HF_DEVICE_MAP", "auto")
        if device_map:
            load_args["device_map"] = device_map
            load_args["offload_folder"] = os.path.join(
                os.environ["OFFLOAD_DIR"], _sanitize_model_name(model_name)
            )
            load_args["offload_state_dict"] = True
    else:
        load_args["torch_dtype"] = torch.float32

    return {"source": resolved_source, "load_args": load_args}


def should_use_cuda_autocast() -> bool:
    import torch

    return torch.cuda.is_available()


def model_device(model: "torch.nn.Module") -> "torch.device":
    import torch

    if hasattr(model, "device"):
        try:
            return torch.device(model.device)
        except Exception:
            pass

    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cleanup_torch() -> None:
    import torch

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            logger.debug("torch.cuda.empty_cache failed", exc_info=True)
