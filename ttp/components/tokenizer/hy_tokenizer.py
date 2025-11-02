import math
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.tools.logging import logger
from ttp.config.job_config import TTPJobConfig
from transformers import AutoTokenizer
from typing import List


class HunYuanTokenizer(BaseTokenizer):
    """AutoTokenizer for Hf Pretrained model loading."""

    def __init__(self, tokenizer_name_or_path: str,
                 model_max_length: int = 8192,
                 use_fast: bool = True,):

        super().__init__()
        hf_tokenizer_kwargs = {}
        hf_tokenizer_kwargs["model_max_length"] = model_max_length
        hf_tokenizer_kwargs["use_fast"] = use_fast
        hf_tokenizer_kwargs["trust_remote_code"] = True
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, **hf_tokenizer_kwargs)

        if self.tokenizer.bos_token is None:
            self.tokenizer.add_special_tokens({'bos_token': '<|beginoftext|>'})
        if self.tokenizer.eos_token is None:
            self.tokenizer.add_special_tokens({'eos_token': '<|endoftext|>'})

        self.bos_id = self.tokenizer.bos_token_id
        self.eos_id = self.tokenizer.eos_token_id
        self._n_words = len(self.tokenizer)  # vocab_size doesn't contain additional tokens

        logger.info(
            f"Hunyuan Tokenizer built: #words {self.get_vocab_size()}, BOS ID {self.bos_id}, EOS ID {self.eos_id}"
        )

        # TODO: n_words (cannot devide by num_process) may lead to bug
        # temp solution: pad vocab size
        multiple = 8192  # a large number
        self._n_words = int(math.ceil(self.get_vocab_size() / multiple) * multiple)

        logger.info(
            f"Hunyuan Tokenizer built: #padded words {self.get_vocab_size()}"
        )

    def encode(self, text, add_bos=False, add_eos=False) -> List[int]:
        try:
            t = self.tokenizer.encode(text)
        except:  # noqa
            logger.error(f"Bad TEXT:\n{text}")
            t = []
        if add_bos:
            t.insert(0, self.bos_id)
        if add_eos:
            t.append(self.eos_id)
        return t

    def decode(self, token_ids):
        token_ids = list(token_ids)
        if token_ids and token_ids[-1] == self.eos_id:
            token_ids = token_ids[:-1]
        return self.tokenizer.decode(token_ids)

    def get_vocab_size(self) -> int:
        return self._n_words


def build_hunyuan_tokenizer(job_config: TTPJobConfig) -> HunYuanTokenizer:
    return HunYuanTokenizer(job_config.model.tokenizer_path, job_config.training.seq_len)
