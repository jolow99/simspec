"""simspec: a typed spec language + thin runtime for LLM multi-agent game studies.

The contributions live in:
- ``simspec.domains``  finite typed domains (the bridge to memo)
- ``simspec.spec``     GameSpec, StateSchema, ActionSpec, View (observation structure), Phase
- ``simspec.rules``    Rule / Predicate / RuleEngine / RuleSpace (contracts and norms)
- ``simspec.runtime``  the thin engine: vault-locked state, chance log, replay, spec checker
- ``simspec.log``      typed JSONL trajectory log
- ``simspec.card``     Harness Card
- ``simspec.agents``   one agent interface, many adapters (scripted, LLM, decision model, memo, replay)
"""
from .domains import Enum, IntRange, Grid, Bool, Product, Table, Text, Domain
from .spec import (Field, StateSchema, State, ActionSpec, View, Derived, Phase, GameSpec,
                   Observation, ObservationLeak)
from .rules import Rule, Predicate, Cmp, IsTrue, And, Or, Not, ActionIs, PyPred, Sanction, RuleEngine, RuleSpace
from .agents.base import Agent, ActionDistribution, ActContext, ScriptedAgent, ReplayAgent
from .runtime import run_episode, check_spec, replay_episode, Engine
from .card import HarnessCard
from .log import Trajectory, RunWriter

__version__ = "0.1.0"
