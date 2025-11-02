# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inference module for TorchTitan."""

from .generation import generate
from .build_model import init_config, build_tokenizer_and_model

__all__ = ["generate", "init_config", "build_tokenizer_and_model"]
