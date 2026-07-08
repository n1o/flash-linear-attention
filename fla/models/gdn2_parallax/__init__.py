# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.gdn2_parallax.configuration_gdn2_parallax import GDN2ParallaxConfig
from fla.models.gdn2_parallax.modeling_gdn2_parallax import GDN2ParallaxForCausalLM, GDN2ParallaxModel

AutoConfig.register(GDN2ParallaxConfig.model_type, GDN2ParallaxConfig, exist_ok=True)
AutoModel.register(GDN2ParallaxConfig, GDN2ParallaxModel, exist_ok=True)
AutoModelForCausalLM.register(GDN2ParallaxConfig, GDN2ParallaxForCausalLM, exist_ok=True)

__all__ = ['GDN2ParallaxConfig', 'GDN2ParallaxForCausalLM', 'GDN2ParallaxModel']
