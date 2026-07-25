"""Production-oriented molecular landscape analysis."""

from .config import WorkflowConfig

__all__ = ["WorkflowConfig", "run_workflow"]
__version__ = "0.5.0"


def run_workflow(config: WorkflowConfig):
    """Lazily import and run the workflow so callers can configure caches first."""
    from .workflow import run_workflow as _run_workflow

    return _run_workflow(config)
