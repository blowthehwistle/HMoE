import torchtitan.protocols.train_spec
from ttp.patches import as_patch
from torchtitan.tools.logging import logger


class _NotSetType:

    def __str__(self):
        return "Not Set"

    def __repr__(self):
        return "Not Set"


NOT_SET = _NotSetType()


torchtitan_register_train_spec = torchtitan.protocols.train_spec.register_train_spec


@as_patch
def ttp_register_train_spec(train_spec: torchtitan.protocols.train_spec.TrainSpec):
    global torchtitan_register_train_spec
    logger.info(f"Register model: {train_spec.name}")
    torchtitan_register_train_spec(train_spec)


def do_patch():
    torchtitan.protocols.train_spec.register_train_spec = ttp_register_train_spec
