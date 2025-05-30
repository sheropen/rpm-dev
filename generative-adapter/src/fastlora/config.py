from dataclasses import asdict, dataclass, field
from peft import PeftConfig

from typing import Optional, Union, List

@dataclass
class FastLoraConfig(PeftConfig):

    fastlora_r: Optional[int] = field(default=32, metadata={"help": "The number of attention heads."})
    fastlora_inter_size: Optional[int] = field(default=None, metadata={"help": "The number of attention heads."})
    fastlora_window: Optional[int] = field(default=1024, metadata={"help": "The number of attention heads."})
    fastlora_max_rank: Optional[int] = field(default=128, metadata={"help": "The number of attention heads."})
    fastlora_attn_len: Optional[int] = field(default=8192, metadata={"help": "The number of attention heads."})
    fastlora_gist_len: Optional[int] = field(default=0, metadata={"help": "The number of attention heads."})
    fastlora_alpha: Optional[int] = field(default=64, metadata={"help": "The number of attention heads."})
    fastlora_dropout: Optional[float] = field(default=0.1, metadata={"help": "The dropout ratio for the attention probabilities."})
    fastlora_arch: Optional[str] = field(default="aassbb", metadata={"help": "The number of attention heads."})
    fastlora_norm: Optional[str] = field(default="forbenius", metadata={"help": "The number of attention heads."})
    fastlora_init: Optional[str] = field(default="random", metadata={"help": "The number of attention heads."})
    fastlora_merge: Optional[str] = field(default="mean", metadata={"help": "The number of attention heads."})
    fastlora_param: Optional[List[str]] = field(default=None, metadata={"help": "the target modules to apply fastlora"})
    fastlora_training_attention_mask: Optional[str] = field(default=None, metadata={"help": "the target modules to apply fastlora"})

    target_modules: Optional[List[str]] = field(default=None, metadata={"help": "The target modules to apply fastlora."})

    # obsolete parameters, for compatibility with the original code
    lora_r: Optional[int] = field(default=0, metadata={"help": "The number of attention heads."})
    lora_alpha: Optional[int] = field(default=0, metadata={"help": "The number of attention heads."})
    lora_dropout: Optional[float] = field(default=0.0, metadata={"help": "The dropout ratio for the attention probabilities."})
    lora_param: Optional[List[str]] = field(default=None, metadata={"help": "The number of attention heads."})
    layer_replication: Optional[list[tuple[int, int]]] = field(default=None, metadata={"help": "Enables replicating layers in a model to expand it to a larger model."})
    megatron_config: Optional[dict] = field(default=None, metadata={"help": "Megatron configuration."})
    bias: Optional[str] = field(default='none', metadata={"help": "Whether to use bias in the attention layer."})
    