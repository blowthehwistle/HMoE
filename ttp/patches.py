import inspect
import logging
import torch.multiprocessing as mp
from einops._torch_specific import allow_ops_in_compiled_graph


try:
    from rod.logging import get_logger
except ImportError:
    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)


mp.set_start_method('spawn', force=True)


allow_ops_in_compiled_graph()


logger = get_logger("patches")


def as_patch(p):
    frame = inspect.currentframe().f_back
    logger.warning(f"Monkey patch: {frame.f_code.co_filename}:{frame.f_lineno}@{p}")
    return p


def run_patches():
    import ttp.tools.utils
    ttp.tools.utils.do_patch()

    import ttp.components.checkpoint
    ttp.components.checkpoint.do_patch()

    import ttp.datasets.hf_datasets
    ttp.datasets.hf_datasets.do_patch()

    import ttp.protocols.train_spec
    ttp.protocols.train_spec.do_patch()

    import ttp.components.optimizer
    ttp.components.optimizer.do_patch()


def register_models():
    # flake8: noqa: F401
    import ttp.experiments.hmoe
