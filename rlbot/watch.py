"""Watch one game between two agents in the pygame replay viewer.

    python -m rlbot.watch hunter expander             # map seed 0
    python -m rlbot.watch harvester hunter --seed 3 --grid 15 15

The game is simulated headless first, then opened as a scrubbable replay.
Controls: SPACE play/pause | ←/→ or H/L step (hold to run) | R restart | Q quit
"""
import argparse

import jax.numpy as jnp
import jax.random as jr

from generals import GeneralsEnv, get_observation
from generals.core import game as game_module
from generals.gui import ReplayGUI
from generals.gui.properties import GuiMode
from rlbot.agents import AGENTS


def main():
    """Parse args, simulate one game headless, then open it in the replay viewer."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("agent0", choices=list(AGENTS))
    p.add_argument("agent1", choices=list(AGENTS))
    p.add_argument("--seed", type=int, default=0, help="map seed")
    p.add_argument("--grid", type=int, nargs=2, default=(10, 10))
    p.add_argument("--truncation", type=int, default=500)
    p.add_argument("--fps", type=int, default=8)
    args = p.parse_args()

    names = [args.agent0, args.agent1]
    agents = [AGENTS[n](id=n) for n in names]
    env = GeneralsEnv(grid_dims=tuple(args.grid), truncation=args.truncation, pool_size=8)
    pool, _ = env.reset(jr.PRNGKey(0))
    state = env.init_state(jr.PRNGKey(args.seed))

    states, infos = [state], [game_module.get_info(state)]
    key = jr.PRNGKey(args.seed + 1)
    while True:
        key, k0, k1 = jr.split(key, 3)
        actions = jnp.stack([agents[0].act(get_observation(state, 0), k0),
                             agents[1].act(get_observation(state, 1), k1)])
        ts, state = env.step(state, actions, pool)
        states.append(ts.last_state)
        infos.append(ts.info)
        if bool(ts.terminated) or bool(ts.truncated):
            break

    w = int(infos[-1].winner)
    print(f"{len(states) - 1} turns, winner: {names[w] if w >= 0 else 'draw'}")
    gui = ReplayGUI(states[0], agent_ids=[f"{n} (P{i})" for i, n in enumerate(names)],
                    fps=args.fps, mode=GuiMode.REPLAY, start_paused=True)
    gui.play(states, infos)


if __name__ == "__main__":
    main()
