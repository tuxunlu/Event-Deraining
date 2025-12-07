from dataclasses import dataclass, field
from typing import Any, Dict

from configs.config_tracker import TrackedConfigMixin

@dataclass
class ModelConfig(TrackedConfigMixin):
    file_name: str = "simple_net"
    class_name: str = "SimpleNet"
    in_chans: int = field(default=1, metadata={"help": "Number of input channels."})
    out_chans: int = field(default=1, metadata={"help": "Number of output channels."})
    dim: int = field(default=48, metadata={"help": "Base dimension of the model."})
    num_blocks: list = field(default_factory=lambda: [2, 2, 2, 2], metadata={"help": "Number of blocks at each stage."})
