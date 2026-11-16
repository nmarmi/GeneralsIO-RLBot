"""Play against one of our agents on the generals.io bot server, from your browser.

    python -m rlbot.play_online expander mylobby
    python -m rlbot.play_online hunter https://bot.generals.io/games/mylobby
    python -m rlbot.play_online harvester mylobby --name "[Bot] csc352-harvester"

Then open https://bot.generals.io/games/<lobby> in a browser and click "Force Start".
The bot rejoins the same lobby after every game; Ctrl-C to stop.

Lobby id: the last path segment of the invite link (https://bot.generals.io/games/<lobby_id>).
Any string works; if the lobby doesn't exist yet, joining creates it.

Bot user id: a secret that *is* the bot's account on the server. It should stay the same
across runs (it keeps the bot's name). By default one is generated on first use and saved
to `.generals_bot_id` (git-ignored); override with --user-id or $GENERALS_BOT_ID.
--name sets the bot's display name. The server adds the "[Bot] " prefix itself (don't include
it) and rejects long names. A user id can be named only once, ever, and must be named before
it can join a lobby.

1v1 only (the observation code assumes one opponent). Fixes over the upstream
`generals.remote.autopilot`: passes an RNG key to `act`, jit-compiles the agent, keeps
the lobby alive while waiting, and does not mistake unexpected events for a win.
"""
from __future__ import annotations

import argparse
import os
import secrets
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from socketio.exceptions import TimeoutError as SioTimeout

from generals.core.observation import Observation
from generals.remote.generalsio_client import DIRECTIONS, GeneralsIOClient
from rlbot.agents import AGENTS

ID_FILE = Path(__file__).resolve().parent.parent / ".generals_bot_id"


def load_user_id(cli_value: str | None) -> str:
    """CLI flag > $GENERALS_BOT_ID > saved file > freshly generated (and saved)."""
    if cli_value:
        return cli_value
    if os.environ.get("GENERALS_BOT_ID"):
        return os.environ["GENERALS_BOT_ID"]
    if ID_FILE.exists():
        return ID_FILE.read_text().strip()
    uid = secrets.token_hex(12)
    ID_FILE.write_text(uid + "\n")
    print(f"Generated a new bot user id and saved it to {ID_FILE}")
    return uid


def parse_lobby(s: str) -> str:
    """Accept either a bare lobby id or a full invite link."""
    return s.rstrip("/").split("/")[-1].split("?")[0]


def to_jax(obs: Observation) -> Observation:
    """Give the server-built numpy observation the dtypes our simulator produces."""
    as_bool = lambda x: jnp.asarray(np.asarray(x).astype(bool))
    as_int = lambda x: jnp.asarray(np.asarray(x), dtype=jnp.int32)
    return Observation(
        armies=as_int(obs.armies), generals=as_bool(obs.generals), castles=as_bool(obs.castles),
        mountains=as_bool(obs.mountains), neutral_cells=as_bool(obs.neutral_cells),
        owned_cells=as_bool(obs.owned_cells), opponent_cells=as_bool(obs.opponent_cells),
        fog_cells=as_bool(obs.fog_cells), structures_in_fog=as_bool(obs.structures_in_fog),
        owned_land_count=as_int(obs.owned_land_count), owned_army_count=as_int(obs.owned_army_count),
        opponent_land_count=as_int(obs.opponent_land_count),
        opponent_army_count=as_int(obs.opponent_army_count), timestep=as_int(obs.timestep),
        allied_cells=as_bool(obs.allied_cells), allied_land_count=as_int(obs.allied_land_count),
        allied_army_count=as_int(obs.allied_army_count),
    )


class BotClient(GeneralsIOClient):
    """Upstream client with a jitted, keyed agent and sturdier lobby/game loops."""

    def __init__(self, agent, user_id: str, seed: int = 0):
        super().__init__(agent, user_id)
        self._act = jax.jit(agent.act)  # recompiles once per new map size
        self._key = jr.PRNGKey(seed)

    def join_private_lobby(self, lobby_id: str) -> None:
        self._status = "lobby"
        self._queue_id = lobby_id
        self.emit("join_private", (lobby_id, self.user_id, self.bot_key))
        try:  # the server replies only on failure, e.g. gio_error "choose a username"
            event, *data = self.receive(timeout=2)
        except SioTimeout:
            print(f"Joined private lobby {lobby_id}.")
            return
        if event in ("gio_error", "error_set_username", "error"):
            msg = data[0] if data else event
            hint = ('\nThis bot id has no username yet: rerun once with --name "[Bot] something".'
                    if "username" in str(msg) else "")
            raise SystemExit(f"Server refused to join the lobby: {msg}{hint}")
        if event == "server_down":
            raise SystemExit("The bot server answered 'server_down': its game server is offline "
                             "(maintenance/restart on their side). Try again later.")
        print(f"Joined private lobby {lobby_id} (server said: {event}).")

    def join_game(self, force_start: bool = True) -> None:
        self._status = "queue"
        print("Waiting in lobby; open the link and press Force Start.")
        last_force = 0.0
        while True:
            if time.time() - last_force > 2:
                self.emit("set_force_start", (self.queue_id, force_start))
                last_force = time.time()
            try:
                event, *data = self.receive(timeout=1)
            except SioTimeout:
                continue
            if event == "gio_error":
                raise SystemExit(f"Server error: {data[0] if data else ''}")
            if event == "game_start":
                self._status = "game"
                self._initialize_game(data)
                self._play_game()
                return

    def _generate_action(self, observation: Observation):
        self._key, k = jr.split(self._key)
        t0 = time.perf_counter()
        action = np.asarray(self._act(to_jax(observation), k))
        dt = time.perf_counter() - t0
        if dt > 0.4:
            print(f"  (agent took {dt:.2f}s this turn, probably compiling)")
        if action[0]:  # pass
            return None
        i, j, d, split = map(int, action[1:])
        width = self.game_state.map[0]
        di, dj = DIRECTIONS[d].value
        return (i * width + j, (i + di) * width + (j + dj), split)

    def _play_game(self) -> None:
        while True:
            event, *data = self.receive()
            if event == "game_update":
                self.game_state.update(data[0])
                if len(self.game_state.scores) != 2:
                    raise RuntimeError("only 1v1 games are supported")
                action = self._generate_action(self.game_state.get_observation())
                if action:
                    self.emit("attack", action)
            elif event in ("game_won", "game_lost"):
                self._finish_game(event == "game_won")
                return


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("agent", choices=list(AGENTS))
    p.add_argument("lobby", help="lobby id or invite link")
    p.add_argument("--user-id", help="secret bot id (default: saved in .generals_bot_id)")
    p.add_argument("--name", help='register a display name, e.g. "[Bot] csc352" (once per user id)')
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--games", type=int, default=0, help="stop after N games (0 = forever)")
    args = p.parse_args()

    lobby = parse_lobby(args.lobby)
    client = BotClient(AGENTS[args.agent](id=args.agent), load_user_id(args.user_id), seed=args.seed)
    try:
        if args.name:
            client.register_agent(args.name)
        print(f"Play here: https://bot.generals.io/games/{lobby}")
        played = 0
        while args.games == 0 or played < args.games:
            client.join_private_lobby(lobby)
            client.join_game()
            played += 1
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
