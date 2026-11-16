"""Play a game against one of our agents in a local pygame window (no server needed).

    python -m rlbot.play_local expander
    python -m rlbot.play_local hunter --grid 15 15 --seed 3
    python -m rlbot.play_local harvester --step          # turn advances only when you act

Controls
    click one of your tiles      select it
    arrows / WASD                move the selected army; the selection follows, so you can
                                 chain moves (they queue, one executes per turn)
    click an adjacent tile       same as an arrow toward it
    Z                            next queued move sends only half the army
    E                            undo the last queued move     X  clear the queue
    Space                        pause (real-time) / pass a turn (step mode)
    + / -                        game speed                    R  new game    Q  quit

Every game (finished, or abandoned with R/Q) is saved to runs/replays/*.npz;
watch it with `python -m rlbot.replay FILE` (or `python -m rlbot.replay` for the latest).

You are red (player 0) unless --seat 1. You only see what your tiles see, as on generals.io.
Rules are the simulator's classic rules (same as training).
"""
from __future__ import annotations

import argparse
import os
from collections import deque
from datetime import datetime
from pathlib import Path as Path_

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pygame

from generals import GeneralsEnv, get_observation
from generals.core.action import compute_valid_move_mask
from generals.core.config import Dimension, Path
from generals.core.game import GameState, get_info
from generals.core.rendering import JaxGameAdapter
from generals.gui.properties import GuiMode, Properties
from generals.gui.rendering import Renderer
from rlbot.agents import AGENTS

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # U D L R, the simulator's direction order
KEY_DIR = {pygame.K_UP: 0, pygame.K_w: 0, pygame.K_DOWN: 1, pygame.K_s: 1,
           pygame.K_LEFT: 2, pygame.K_a: 2, pygame.K_RIGHT: 3, pygame.K_d: 3}
COLORS = [(220, 70, 70), (70, 100, 220)]
SPEEDS = [0.25, 0.5, 1.0, 2.0, 4.0]  # turns per second = 2 * speed (1x = generals.io pace)
SQ = Dimension.SQUARE_SIZE.value
GOLD, GREY, WHITE = (236, 206, 112), (170, 174, 178), (240, 240, 240)

HELP = [("click", "select tile"), ("arrows/WASD", "move"), ("Z", "half army"),
        ("E / X", "undo / clear"), ("Space", "pause / pass"), ("+ / -", "speed"),
        ("R", "new game"), ("Q", "quit")]


class LocalGame:
    def __init__(self, args):
        self.args = args
        self.me, self.bot = args.seat, 1 - args.seat
        self.names = ["", ""]
        self.names[self.me] = "You"
        self.names[self.bot] = args.agent
        self.agent = AGENTS[args.agent](id=args.agent)
        self.env = GeneralsEnv(grid_dims=tuple(args.grid), truncation=args.truncation, pool_size=4)
        self.pool, _ = self.env.reset(jr.PRNGKey(0))
        self._step = jax.jit(self.env.step)
        self._act = jax.jit(lambda s, k: self.agent.act(get_observation(s, self.bot), k))
        self._mask = jax.jit(lambda s: (lambda o: compute_valid_move_mask(o.armies, o.owned_cells, o.mountains))(
            get_observation(s, self.me)))
        self.seed = args.seed
        self.speed_i = SPEEDS.index(1.0)
        self.new_game()

    # ----- game state -----
    def new_game(self):
        self.state = self.env.init_state(jr.PRNGKey(self.seed))
        self.history = [self.state]          # every state, for the replay file
        self.action_log = []                 # (2, 5) actions per turn, indexed by player
        self.key = jr.PRNGKey(self.seed + 1000)
        self.queue: deque = deque()  # (row, col, dir, split)
        self.selected = None
        self.half = False
        self.paused = False
        self.result = None
        print(f"New game, map seed {self.seed}...", flush=True)  # first game also jit-compiles (a few s)
        self._act(self.state, self.key).block_until_ready()
        self._mask(self.state).block_until_ready()
        print("Ready.", flush=True)

    def save_replay(self):
        """Write the game so far to runs/replays/ (once per game; skipped if no turn was played)."""
        if self.args.no_save or len(self.history) < 2 or getattr(self, "_saved_len", None) == len(self.history):
            return
        self._saved_len = len(self.history)
        out = Path_(self.args.replay_dir)
        out.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        f = out / f"{stamp}_you-vs-{self.args.agent}_seed{self.seed}.npz"
        fields = {k: np.stack([np.asarray(getattr(st, k)) for st in self.history]) for k in GameState._fields}
        np.savez_compressed(
            f, **{"state_" + k: v for k, v in fields.items()},
            actions=np.stack(self.action_log), names=np.array(self.names), human_seat=self.me,
            agent=self.args.agent, seed=self.seed, grid=np.array(self.args.grid),
            result=self.result or "unfinished",
        )
        print(f"Replay saved: {f}   (view: python -m rlbot.replay {f})", flush=True)

    def my_general(self):
        g = np.argwhere(np.asarray(self.state.generals) & np.asarray(self.state.ownership[self.me]))
        return tuple(g[0]) if len(g) else None

    def advance(self):
        """Play one turn: my first valid queued move (or pass) vs the bot's move."""
        mask = np.asarray(self._mask(self.state))
        mine = np.array([1, 0, 0, 0, 0], dtype=np.int32)
        while self.queue:
            r, c, d, split = self.queue.popleft()
            if mask[r, c, d]:
                mine = np.array([0, r, c, d, split], dtype=np.int32)
                break
        self.key, k = jr.split(self.key)
        bot_action = np.asarray(self._act(self.state, k))
        actions = np.zeros((2, 5), dtype=np.int32)
        actions[self.me], actions[self.bot] = mine, bot_action
        ts, new_state = self._step(self.state, jnp.asarray(actions), self.pool)
        self.action_log.append(actions)
        self.history.append(ts.last_state)  # == new_state unless the game just ended
        if bool(ts.terminated) or bool(ts.truncated):
            self.state = ts.last_state  # state before the auto-reset
            w = int(ts.info.winner)
            self.result = "You win!" if w == self.me else "Draw (turn limit)" if w < 0 else f"{self.names[self.bot]} wins"
            print(self.result, f"(turn {int(self.state.time) // 2})", flush=True)
            self.props.agent_fov[self.names[self.bot]] = True  # reveal the map
            self.save_replay()
        else:
            self.state = new_state
        if self.selected is not None and not np.asarray(self.state.ownership[self.me])[self.selected] and not self.queue:
            self.selected = None  # lost the selected tile

    def queue_move(self, d):
        if self.selected is None:
            return
        r, c = self.selected
        nr, nc = r + DIRS[d][0], c + DIRS[d][1]
        H, W = self.args.grid
        if not (0 <= nr < H and 0 <= nc < W) or bool(np.asarray(self.state.mountains)[nr, nc]):
            return
        self.queue.append((r, c, d, int(self.half)))
        self.half = False
        self.selected = (nr, nc)
        if self.args.step and not self.result:
            self.advance()

    def undo(self):
        if self.queue:
            r, c, _, _ = self.queue.pop()
            self.selected = (r, c)

    # ----- input -----
    def on_click(self, x, y):
        H, W = self.args.grid
        if x >= W * SQ or y >= H * SQ:
            return
        cell = (y // SQ, x // SQ)
        if self.selected is not None and abs(cell[0] - self.selected[0]) + abs(cell[1] - self.selected[1]) == 1:
            self.queue_move(DIRS.index((cell[0] - self.selected[0], cell[1] - self.selected[1])))
        elif cell == self.selected:
            self.half = not self.half  # click the selection again: toggle half, like the site
        elif bool(np.asarray(self.state.ownership[self.me])[cell]):
            self.selected, self.half = cell, False
        else:
            self.selected = None

    def on_key(self, key):
        if key == pygame.K_q:
            self.save_replay()
            return False
        if key == pygame.K_r:
            self.save_replay()
            self.seed += 1
            self.new_game()
        elif self.result:
            return True
        elif key in KEY_DIR:
            self.queue_move(KEY_DIR[key])
        elif key == pygame.K_z:
            self.half = not self.half
        elif key == pygame.K_e:
            self.undo()
        elif key == pygame.K_x:
            self.queue.clear()
        elif key == pygame.K_SPACE:
            if self.args.step:
                self.advance()
            else:
                self.paused = not self.paused
        elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self.speed_i = min(len(SPEEDS) - 1, self.speed_i + 1)
        elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.speed_i = max(0, self.speed_i - 1)
        return True

    # ----- drawing -----
    def make_renderer(self):
        info = get_info(self.state)
        self.adapter = JaxGameAdapter(self.state, self.names, info)
        props = Properties(self.adapter, {n: {"color": COLORS[i]} for i, n in enumerate(self.names)}, GuiMode.TRAIN)
        props.agent_fov[self.names[self.bot]] = False  # your fog of war only
        self.props = props
        self.renderer = Renderer(props)
        self.font = pygame.font.Font(Path.FONT_PATH, 14)
        self.big = pygame.font.Font(Path.FONT_PATH, 34)

    def draw(self):
        self.adapter.update_from_state(self.state, get_info(self.state))
        self.props.game_speed = SPEEDS[self.speed_i]
        r = self.renderer
        r.render_grid()
        r.render_stats()
        screen = r.screen

        # Queued moves as arrows, then the selection box.
        for (qr, qc, d, split) in self.queue:
            cx, cy = qc * SQ + SQ // 2, qr * SQ + SQ // 2
            ex, ey = cx + DIRS[d][1] * SQ // 2, cy + DIRS[d][0] * SQ // 2
            pygame.draw.line(screen, GOLD if split else WHITE, (cx, cy), (ex, ey), 3)
            pygame.draw.circle(screen, GOLD if split else WHITE, (ex, ey), 4)
        if self.selected is not None:
            sr, sc = self.selected
            pygame.draw.rect(screen, GOLD if self.half else WHITE, (sc * SQ, sr * SQ, SQ, SQ), 3)
            if self.half:
                screen.blit(self.font.render("50%", True, GOLD), (sc * SQ + 3, sr * SQ + 2))

        # Status + controls under the scoreboard.
        px = r.display_grid_width + 12
        y = (r.num_score_rows + 1) * Dimension.GUI_CELL_HEIGHT.value + 10
        panel_h = screen.get_height() - y
        screen.fill((0, 0, 0), (r.display_grid_width, y, r.right_panel_width, panel_h))
        mode = "step mode" if self.args.step else ("PAUSED" if self.paused else f"speed {SPEEDS[self.speed_i]}x")
        screen.blit(self.font.render(f"{mode}  |  queued moves: {len(self.queue)}", True, WHITE), (px, y))
        y += 24
        for k, v in HELP:
            if y + 18 > screen.get_height():
                break
            screen.blit(self.font.render(k, True, GOLD), (px, y))
            screen.blit(self.font.render(v, True, GREY), (px + 110, y))
            y += 19

        if self.result:
            W = r.display_grid_width
            band = pygame.Surface((W, 90), pygame.SRCALPHA)
            band.fill((0, 0, 0, 190))
            screen.blit(band, (0, screen.get_height() // 2 - 45))
            t = self.big.render(self.result, True, WHITE)
            screen.blit(t, t.get_rect(center=(W // 2, screen.get_height() // 2 - 12)))
            s = self.font.render("R: new game   Q: quit", True, GREY)
            screen.blit(s, s.get_rect(center=(W // 2, screen.get_height() // 2 + 22)))
        pygame.display.flip()

    def run(self):
        pygame.init()
        pygame.display.set_caption(f"generals: you vs {self.args.agent}")
        self.make_renderer()
        g = self.my_general()
        self.selected = g
        clock = pygame.time.Clock()
        last_turn = pygame.time.get_ticks()
        running = True
        while running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    self.save_replay()
                    running = False
                elif ev.type == pygame.KEYDOWN:
                    seed_before = self.seed
                    running = self.on_key(ev.key)
                    if self.seed != seed_before:  # new game: fresh renderer for the new map
                        self.make_renderer()
                        self.selected = self.my_general()
                elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and not self.result:
                    self.on_click(*ev.pos)
            now = pygame.time.get_ticks()
            interval = 500 / SPEEDS[self.speed_i]  # 1x: one turn per 0.5 s, as on generals.io
            if not self.args.step and not self.paused and not self.result and now - last_turn >= interval:
                self.advance()
                last_turn = now
            self.draw()
            clock.tick(60)
        pygame.quit()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("agent", choices=list(AGENTS))
    p.add_argument("--grid", type=int, nargs=2, default=(15, 15))
    p.add_argument("--seed", type=int, default=0, help="map seed (R advances it)")
    p.add_argument("--truncation", type=int, default=1000, help="turn limit (draw)")
    p.add_argument("--seat", type=int, choices=(0, 1), default=0)
    p.add_argument("--step", action="store_true", help="the game only advances when you move or press Space")
    p.add_argument("--replay-dir", default="runs/replays", help="where finished games are saved")
    p.add_argument("--no-save", action="store_true", help="don't save replays")
    LocalGame(p.parse_args()).run()


if __name__ == "__main__":
    main()
