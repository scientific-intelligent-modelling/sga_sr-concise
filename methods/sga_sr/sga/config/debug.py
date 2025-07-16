from dataclasses import dataclass, field
from .base import BaseConfig
from .optim import AdamOptimConfig
from .llm import GPT4LLMConfig
from .physics import DefaultPhysicsConfig

@dataclass(kw_only=True)
class DebugConfig(BaseConfig, name='debug'):
    optim: AdamOptimConfig = field(default_factory=AdamOptimConfig)
    llm: GPT4LLMConfig = field(default_factory=GPT4LLMConfig)
    physics: DefaultPhysicsConfig = field(default_factory=DefaultPhysicsConfig)

    overwrite: bool = True
