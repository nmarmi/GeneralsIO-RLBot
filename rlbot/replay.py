"""Open a saved local game (from `rlbot.play_local`) in the replay viewer.

    python -m rlbot.replay                                  # most recent file in runs/replays/
    python -m rlbot.replay runs/replays/20261112-153000_you-vs-hunter_seed0.npz
    python -m rlbot.replay FILE --fog                       # start from the human's fog of war

Controls: SPACE play/pause | ←/→ or H/L step (hold to run) | R restart | Q quit
Click a name in the scoreboard to toggle that player's view.

File format (.npz): `state_<field>` arrays stacked over time for each GameState field
(T+1 states), `actions` (T, 2, 5) indexed by player, plus `names`, `human_seat`, `agent`,
`seed`, `grid`, `result`. Use `load_replay` to get the states back as GameState objects.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from generals.core.game import GameState, get_info
from generals.gui import ReplayGUI
from generals.gui.properties import GuiMode


def load_replay(path: str | Path):
    """Return (states, actions, meta) for a saved game."""
    z = np.load(path)
    T = z["state_armies"].shape[0]
    states = [GameState(**{k: jnp.asarray(z["state_" + k][t]) for k in GameState._fields}) for t in range(T)]
    meta = {k: z[k].tolist() for k in ("names", "human_seat", "agent", "seed", "grid", "result")}
    return states, z["actions"], meta


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file", nargs="?", help="replay .npz (default: newest in runs/replays/)")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--fog", action="store_true", help="start with only the human's view")
    args = p.parse_args()

    path = args.file
    if path is None:
        files = sorted(Path("runs/replays").glob("*.npz"))
        if not files:
            raise SystemExit("No replays in runs/replays/ yet; play one with `python -m rlbot.play_local AGENT`.")
        path = files[-1]
    states, actions, meta = load_replay(path)
    print(f"{path}: {meta['names'][0]} vs {meta['names'][1]}, {len(actions)} turns, {meta['result']}")

    gui = ReplayGUI(states[0], agent_ids=meta["names"], fps=args.fps, mode=GuiMode.REPLAY, start_paused=True)
    if args.fog:
        for i, n in enumerate(gui.agent_ids):
            gui._gui.properties.agent_fov[n] = i == meta["human_seat"]
    gui.play(states, [get_info(s) for s in states])


if __name__ == "__main__":
    main()
