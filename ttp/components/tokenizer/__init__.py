from torchtitan.tools.logging import logger
from torchtitan.components.tokenizer import Tokenizer, build_hf_tokenizer
from ttp.components.tokenizer.hy_tokenizer import build_hunyuan_tokenizer
from ttp.config.job_config import TTPJobConfig


def build_tokenizer(job_config: TTPJobConfig) -> Tokenizer:
    tokenizer_type = job_config.model.tokenizer_type
    tokenizer_path = job_config.model.tokenizer_path
    logger.info(f"Building {tokenizer_type} tokenizer locally from {tokenizer_path}.")
    match tokenizer_type:
        case "hf_tokenizer":
            return build_hf_tokenizer(job_config)
        case "hunyuan_tokenizer":
            return build_hunyuan_tokenizer(job_config)
        case _:
            raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}.")
