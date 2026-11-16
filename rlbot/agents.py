"""Registry of agents available to the rlbot tools (watch, play_local, play_online, ...).

Each entry maps a CLI name to an `Agent` class constructed as `cls(id=name)`.
An agent's `act(observation, key) -> int32[5]` must be JAX-traceable.
"""
from generals.agents import ExpanderAgent, HunterAgent, RandomAgent
from generals.agents.harvester_agent import HarvesterAgent  # not exported from generals.agents

AGENTS = {
    "random": RandomAgent,
    "expander": ExpanderAgent,
    "hunter": HunterAgent,
    "harvester": HarvesterAgent,
}
