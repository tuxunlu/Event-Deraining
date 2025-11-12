from dataclasses import dataclass, field
from typing import Any, Dict

from configs.config_tracker import TrackedConfigMixin

@dataclass
class ModelConfig(TrackedConfigMixin):
    file_name: str = "simple_net"
    class_name: str = "SimpleNet"
    H: int = field(default=32, metadata={"help": "Input height."})
    W: int = field(default=32, metadata={"help": "Input width."})
    N: int = field(default=5, metadata={"help": "Sequence length."})
    d_models: list = field(default_factory=lambda: [16, 32, 32, 64], metadata={"help": "Model depths."})
