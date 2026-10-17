"""The rule_obeying_harvest gridworld (Oldenburg & Zhi-Xuan 2024): Commons Harvest + Clean Up +
Territory mechanics, as pure functions over a plain dict state so the same code serves the
transition, the planners' forward model, and tests.

Fidelity notes: 4-neighbour density regrowth with the paper's probabilities and pollution throttle;
dirt spawns on potential cells; beams (clean / zap / pay / claim) are straight lines ahead of the
agent; movement conflicts are resolved in agent order; see docs/decisions.md D-012.
"""
from __future__ import annotations

from typing import Any

import numpy as np

ACTIONS = ["NOOP", "FORWARD", "BACKWARD", "STEP_LEFT", "STEP_RIGHT", "TURN_LEFT", "TURN_RIGHT",
           "FIRE_ZAP", "FIRE_CLEAN", "FIRE_CLAIM", "EAT", "PAY"]
ACTION_CLASS = {"NOOP": "NOOP", "FORWARD": "MOVE", "BACKWARD": "MOVE", "STEP_LEFT": "MOVE", "STEP_RIGHT": "MOVE",
                "TURN_LEFT": "TURN", "TURN_RIGHT": "TURN", "FIRE_ZAP": "ZAP", "FIRE_CLEAN": "CLEAN", "FIRE_CLAIM": "CLAIM",
                "EAT": "EAT", "PAY": "PAY"}
ROLES = ["free", "cleaner", "farmer"]            # AGENT_LOOK: 0 free ("egalitarian"), 1 cleaner, 2 farmer
LOOK = {r: i for i, r in enumerate(ROLES)}

REGROWTH = [0.02, 0.05, 0.07, 0.09]
THRESH_DEPLETION, THRESH_RESTORATION = 0.7, 0.1
DIRT_SPAWN_PROB = 0.2
APPLE_REWARD, ACTION_COST, BLOCK_COST, ZAP_PENALTY = 1.0, 0.01, 0.1, 15.0
ZAP_FREEZE = 4
BEAM_LEN = 3
PROPERTY_RADIUS = 2

PAPER_MAP = """
WWWWWWWWWWWWWWWWWWWWWWW
WFHFHFHFHFHFHFHHFHFFHFW
WFHFHFHFHFHFHFHHFHFFHFW
W========+~FHFHHFHf===W
W   P     ===+~Sf     W
W      P     <~Sf  P  W
W          P <~S>     W
WT^A^T^T^T^T^;~S^T^A^TW
W_AAA_____________AAA_W
WAAAAA_____A_____AAAAAW
W_AAA_____AAA_____AAA_W
W__A_____AAAAA_____A__W
W_____A___AAA___A_____W
W_Q__AAA___A___AAA__G_W
W___AAAAA_____AAAAA___W
W____AAA____Q__AAA____W
W__G__A___Q_____A__G__W
WWWWWWWWWWWWWWWWWWWWWWW
""".strip("\n")

SMALL_MAP = """
WWWWWWWWWWWWWW
WFHFHFHFHFHFHW
W=====+~SSS==W
W   P   <~S  W
WT^A^T^T;~S^TW
W_AAA____AAA_W
WAAAAA__AAAAAW
W_AAA_Q__AAA_W
W__A___G__A__W
WWWWWWWWWWWWWW
""".strip("\n")

DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]       # N, E, S, W as (dx, dy)


class Layout:
    def __init__(self, ascii_map: str = PAPER_MAP):
        rows = ascii_map.split("\n")
        self.H, self.W = len(rows), len(rows[0])
        self.n = self.H * self.W
        self.wall = np.zeros(self.n, bool)
        self.river = np.zeros(self.n, bool)
        self.dirt_capable = np.zeros(self.n, bool)
        self.dirt_init = np.zeros(self.n, bool)
        self.apple_capable = np.zeros(self.n, bool)
        self.apple_init = np.zeros(self.n, bool)
        self.spawns: list[int] = []
        for y, row in enumerate(rows):
            for x, ch in enumerate(row):
                c = y * self.W + x
                if ch == "W":
                    self.wall[c] = True
                elif ch in "SH~F":
                    self.river[c] = True
                    if ch in "HF":
                        self.dirt_capable[c] = True
                        self.dirt_init[c] = ch == "F"
                elif ch in "A_":
                    self.apple_capable[c] = True
                    self.apple_init[c] = ch == "A"
                if ch in "PGQ":
                    self.spawns.append(c)
        self.walkable = ~self.wall & ~self.river
        self.cells = self.n

    def xy(self, c: int) -> tuple[int, int]:
        return c % self.W, c // self.W

    def cell(self, x: int, y: int) -> int | None:
        if 0 <= x < self.W and 0 <= y < self.H:
            return y * self.W + x
        return None

    def neighbours4(self, c: int) -> list[int]:
        x, y = self.xy(c)
        out = []
        for dx, dy in DIRS:
            n = self.cell(x + dx, y + dy)
            if n is not None:
                out.append(n)
        return out

    def neighbours8(self, c: int) -> list[int]:
        x, y = self.xy(c)
        out = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                n = self.cell(x + dx, y + dy)
                if n is not None:
                    out.append(n)
        return out


# ----------------------------------------------------------------------------- state helpers

def initial_world(layout: Layout, roles: list[str], chance) -> dict[str, Any]:
    n_agents = len(roles)
    spawns = list(layout.spawns)
    pos = []
    for i in range(n_agents):
        cands = [c for c in spawns if c not in pos] or [c for c in range(layout.n) if layout.walkable[c] and c not in pos and not layout.apple_init[c]]
        s = chance.choice(f"spawn:{i}", cands)
        pos.append(s)
    prop = [0] * layout.n
    for i, c in enumerate(pos):
        _claim_rect(layout, prop, i + 1, c)
    return dict(
        t=0, apple=[bool(v) for v in layout.apple_init], desiccated=[False] * layout.n,
        dirt=[("dirt" if layout.dirt_init[c] else ("potential" if layout.dirt_capable[c] else "none")) for c in range(layout.n)],
        property=prop, pos=pos, orient=[chance.integers(f"orient:{i}", 0, 3) for i in range(n_agents)],
        inventory=[0] * n_agents, since_cleaned=[0] * n_agents, since_paid=[0] * n_agents, frozen=[0] * n_agents,
        look=[LOOK[r] for r in roles], stolen=[0] * (n_agents * n_agents),
        harvested=[False] * n_agents, harvest_apples_around=[0] * n_agents, harvested_foreign=[False] * n_agents,
        ate=[False] * n_agents, zapped=[False] * n_agents, cleaned=[False] * n_agents, paid=[False] * n_agents,
    )


def _claim_rect(layout: Layout, prop: list[int], owner: int, centre: int, radius: int = PROPERTY_RADIUS) -> None:
    x0, y0 = layout.xy(centre)
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            c = layout.cell(x0 + dx, y0 + dy)
            if c is not None and layout.apple_capable[c]:
                prop[c] = owner


def dirt_fraction(layout: Layout, dirt: list[str]) -> float:
    cap = [c for c in range(layout.n) if layout.dirt_capable[c]]
    if not cap:
        return 0.0
    return sum(1 for c in cap if dirt[c] == "dirt") / len(cap)


def apples_around(layout: Layout, apple: list[bool], c: int) -> int:
    return sum(1 for n in layout.neighbours8(c) if apple[n])


def beam_cells(layout: Layout, pos: int, orient: int, length: int = BEAM_LEN) -> list[int]:
    x, y = layout.xy(pos)
    dx, dy = DIRS[orient]
    out = []
    for k in range(1, length + 1):
        c = layout.cell(x + dx * k, y + dy * k)
        if c is None or layout.wall[c]:
            break
        out.append(c)
    return out


# ----------------------------------------------------------------------------- single-agent step (forward model)

def move_target(layout: Layout, pos: int, orient: int, action: str) -> int | None:
    x, y = layout.xy(pos)
    if action == "FORWARD":
        d = DIRS[orient]
    elif action == "BACKWARD":
        d = DIRS[(orient + 2) % 4]
    elif action == "STEP_LEFT":
        d = DIRS[(orient + 3) % 4]
    elif action == "STEP_RIGHT":
        d = DIRS[(orient + 1) % 4]
    else:
        return None
    return layout.cell(x + d[0], y + d[1])


def apply_action(layout: Layout, w: dict[str, Any], i: int, action: str) -> tuple[dict[str, Any], float]:
    """Apply agent i's action to world dict w (mutated copy returned), with others fixed. Returns (w, reward)."""
    w = {k: (list(v) if isinstance(v, list) else v) for k, v in w.items()}
    r = 0.0
    n_agents = len(w["pos"])
    w["harvested"][i] = False
    w["harvested_foreign"][i] = False
    w["harvest_apples_around"][i] = 0
    w["ate"][i] = False
    w["cleaned"][i] = False
    w["paid"][i] = False
    if w["frozen"][i] > 0:
        return w, 0.0
    if action != "NOOP":
        r -= ACTION_COST
    cls = ACTION_CLASS[action]
    if cls == "MOVE":
        t = move_target(layout, w["pos"][i], w["orient"][i], action)
        if t is None or layout.wall[t]:
            r -= BLOCK_COST
        elif layout.river[t]:
            r -= BLOCK_COST * 10
        elif t in w["pos"]:
            r -= BLOCK_COST
        else:
            w["pos"][i] = t
            if w["apple"][t]:
                w["apple"][t] = False
                w["inventory"][i] += 1
                w["harvested"][i] = True
                w["harvest_apples_around"][i] = apples_around(layout, w["apple"], t)
                owner = w["property"][t]
                if owner not in (0, i + 1):
                    w["harvested_foreign"][i] = True
                    w["stolen"][(owner - 1) * n_agents + i] += 1
    elif action == "TURN_LEFT":
        w["orient"][i] = (w["orient"][i] + 3) % 4
    elif action == "TURN_RIGHT":
        w["orient"][i] = (w["orient"][i] + 1) % 4
    elif action == "EAT":
        if w["inventory"][i] > 0:
            w["inventory"][i] -= 1
            w["ate"][i] = True
            r += APPLE_REWARD
    elif action == "FIRE_CLEAN":
        hit = False
        for c in beam_cells(layout, w["pos"][i], w["orient"][i]):
            if w["dirt"][c] == "dirt":
                w["dirt"][c] = "potential"
                hit = True
        if hit:
            w["cleaned"][i] = True
            w["since_cleaned"][i] = -1          # incremented to 0 at end of step
    elif action == "FIRE_CLAIM":
        for c in beam_cells(layout, w["pos"][i], w["orient"][i], 1):
            if layout.apple_capable[c]:
                w["property"][c] = i + 1
    elif action == "PAY":
        if w["inventory"][i] > 0:
            cells = beam_cells(layout, w["pos"][i], w["orient"][i])
            for j in range(n_agents):
                if j != i and w["pos"][j] in cells and w["look"][j] == LOOK["cleaner"]:
                    w["inventory"][i] -= 1
                    w["inventory"][j] += 1
                    w["paid"][i] = True
                    w["since_paid"][i] = -1
                    break
    elif action == "FIRE_ZAP":
        cells = beam_cells(layout, w["pos"][i], w["orient"][i])
        for j in range(n_agents):
            if j != i and w["pos"][j] in cells:
                w["frozen"][j] = ZAP_FREEZE
                w["zapped"][j] = True
    return w, r


def env_step(layout: Layout, w: dict[str, Any], chance) -> dict[str, Any]:
    """Environment dynamics after all agents acted: counters, freeze, regrowth, dirt spawn, desiccation."""
    n_agents = len(w["pos"])
    for i in range(n_agents):
        w["since_cleaned"][i] += 1
        w["since_paid"][i] += 1
        if w["frozen"][i] > 0:
            w["frozen"][i] -= 1
    df = dirt_fraction(layout, w["dirt"])
    interp = min(max((df - THRESH_DEPLETION) / (THRESH_RESTORATION - THRESH_DEPLETION), 0.0), 1.0)
    apple = w["apple"]
    new_apple = list(apple)
    for c in range(layout.n):
        if not layout.apple_capable[c] or apple[c]:
            continue
        n = sum(1 for m in layout.neighbours4(c) if apple[m])
        p = REGROWTH[min(n, 3)] * interp
        if n == 0:
            w["desiccated"][c] = True
            p = REGROWTH[0] * interp * 0.1
        if p > 0 and chance.uniform(f"regrow:{c}") < p:
            new_apple[c] = True
            w["desiccated"][c] = False
    w["apple"] = new_apple
    potential = [c for c in range(layout.n) if w["dirt"][c] == "potential"]
    if potential and chance.uniform("dirt_spawn") < DIRT_SPAWN_PROB:
        c = chance.choice("dirt_cell", potential)
        w["dirt"][c] = "dirt"
    w["t"] += 1
    return w


def desiccated_ratio(layout: Layout, w: dict[str, Any]) -> float:
    cap = int(layout.apple_capable.sum())
    return sum(1 for c in range(layout.n) if layout.apple_capable[c] and w["desiccated"][c]) / cap if cap else 0.0


def bfs_distances(layout: Layout, w: dict[str, Any], start: int, blocked_agents: bool = True) -> np.ndarray:
    """Grid distances from start over walkable cells (other agents block)."""
    dist = np.full(layout.n, 10 ** 6, dtype=int)
    occupied = set(w["pos"]) if blocked_agents else set()
    dist[start] = 0
    frontier = [start]
    while frontier:
        nxt = []
        for c in frontier:
            for m in layout.neighbours4(c):
                if layout.walkable[m] and (m not in occupied or m == start) and dist[m] > dist[c] + 1:
                    dist[m] = dist[c] + 1
                    nxt.append(m)
        frontier = nxt
    return dist
