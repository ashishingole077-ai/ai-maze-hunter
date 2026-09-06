"""AI Maze Hunter - a small Pygame game about escaping an A* enemy."""

from __future__ import annotations

import argparse
import heapq
import json
import math
import random
import sys
from array import array
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pygame

Color = tuple[int, int, int]
Position = tuple[int, int]

WIDTH, HEIGHT = 1120, 720
CELL = 32
GRID_COLS, GRID_ROWS = 29, 17
BOARD_X = 32
BOARD_Y = 112
BOARD_W = GRID_COLS * CELL
BOARD_H = GRID_ROWS * CELL
TOP_SCORE_FILE = Path(__file__).with_name("leaderboard.json")

BG: Color = (11, 17, 28)
PANEL: Color = (18, 27, 43)
PANEL_LIGHT: Color = (25, 37, 57)
INK: Color = (231, 239, 248)
MUTED: Color = (135, 155, 178)
WALL: Color = (35, 49, 70)
WALL_EDGE: Color = (52, 72, 98)
CYAN: Color = (69, 219, 210)
CORAL: Color = (255, 109, 105)
GOLD: Color = (246, 194, 83)


def astar(
    grid: list[list[int]],
    start: Position,
    goal: Position,
    trace: dict[str, set[Position]] | None = None,
) -> list[Position] | None:
    """Return the shortest four-direction path, including start and goal."""
    if trace is not None:
        trace["open"] = {start}
        trace["closed"] = set()
        trace["goal"] = {goal}
    if start == goal:
        return [start]

    rows, cols = len(grid), len(grid[0])
    open_list: list[tuple[int, int, Position]] = []
    heapq.heappush(open_list, (0, 0, start))
    came_from: dict[Position, Position] = {}
    cost = {start: 0}
    sequence = 0

    while open_list:
        _, _, current = heapq.heappop(open_list)
        if trace is not None:
            trace["open"].discard(current)
            trace["closed"].add(current)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        row, col = current
        for next_row, next_col in (
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ):
            if not (0 <= next_row < rows and 0 <= next_col < cols):
                continue
            if grid[next_row][next_col] == 1:
                continue

            neighbor = (next_row, next_col)
            new_cost = cost[current] + 1
            if new_cost < cost.get(neighbor, sys.maxsize):
                cost[neighbor] = new_cost
                heuristic = abs(next_row - goal[0]) + abs(next_col - goal[1])
                sequence += 1
                heapq.heappush(open_list, (new_cost + heuristic, sequence, neighbor))
                if trace is not None:
                    trace["open"].add(neighbor)
                came_from[neighbor] = current

    return None


def generate_maze(seed: int | None = None) -> list[list[int]]:
    """Generate a perfect maze and open it up into a generous play area."""
    rng = random.Random(seed)
    grid = [[1 for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
    start = (1, 1)
    grid[start[0]][start[1]] = 0
    stack = [start]

    while stack:
        row, col = stack[-1]
        neighbors = []
        for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            nr, nc = row + dr, col + dc
            if 1 <= nr < GRID_ROWS - 1 and 1 <= nc < GRID_COLS - 1 and grid[nr][nc] == 1:
                neighbors.append((nr, nc, row + dr // 2, col + dc // 2))
        if neighbors:
            nr, nc, wall_row, wall_col = rng.choice(neighbors)
            grid[wall_row][wall_col] = 0
            grid[nr][nc] = 0
            stack.append((nr, nc))
        else:
            stack.pop()

    # A few openings make the generated maze less predictable and more fun to dodge in.
    for _ in range(18):
        row = rng.randrange(1, GRID_ROWS - 1)
        col = rng.randrange(1, GRID_COLS - 1)
        if grid[row][col] == 1 and (row + col) % 2 == 0:
            grid[row][col] = 0
    return grid


def load_scores() -> list[int]:
    try:
        scores = json.loads(TOP_SCORE_FILE.read_text(encoding="utf-8"))
        return sorted((int(score) for score in scores), reverse=True)[:5]
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return []


def save_score(score: int) -> None:
    scores = sorted(load_scores() + [score], reverse=True)[:5]
    try:
        TOP_SCORE_FILE.write_text(json.dumps(scores), encoding="utf-8")
    except OSError:
        pass


class HunterGame:
    def __init__(self, difficulty: str = "normal", seed: int | None = None) -> None:
        self.rng = random.Random(seed)
        self.difficulty = "medium" if difficulty == "normal" else difficulty
        self.grid = generate_maze(seed)
        self.player: Position = self._nearest_open((GRID_ROWS - 2, GRID_COLS - 2))
        self.enemies: list[Position] = [self._nearest_open((1, 1))]
        enemy_spawns = {
            "hard": [(GRID_ROWS - 2, 1)],
            "extreme": [(GRID_ROWS - 2, 1), (1, GRID_COLS - 2)],
        }
        for position in enemy_spawns.get(self.difficulty, []):
            enemy = self._nearest_open(position)
            if enemy != self.player and enemy not in self.enemies:
                self.enemies.append(enemy)
        self.max_health = {"easy": 5, "medium": 3, "hard": 2, "extreme": 1}[self.difficulty]
        self.ai_interval = {"easy": 0.75, "medium": 0.5, "hard": 0.3, "extreme": 0.2}[self.difficulty]
        self.prediction_steps = 3 if self.difficulty == "extreme" else 0
        self.coins = self._place_coins()
        self.health = self.max_health
        self.score = 0
        self.started_at = 0.0
        self.last_hit = -10.0
        self.last_ai_move = -10.0
        self.last_direction = (0, 1)
        self.dash_ready_at = 0.0
        self.paused = False
        self.game_over = False
        self.won = False
        self.ai_steps = 0
        self.events: list[str] = []
        self.open_nodes: set[Position] = set()
        self.closed_nodes: set[Position] = set()
        self.current_path: list[Position] = []
        self.predicted_target = self.player
        self.path_metrics = (0, 0, 0)

    def _place_coins(self) -> set[Position]:
        walkable = [
            (row, col)
            for row in range(1, GRID_ROWS - 1)
            for col in range(1, GRID_COLS - 1)
            if self.grid[row][col] == 0 and (row, col) != self.player
        ]
        self.rng.shuffle(walkable)
        coin_count = {"easy": 8, "medium": 10, "hard": 12, "extreme": 14}[self.difficulty]
        return set(walkable[:coin_count])

    def _nearest_open(self, position: Position) -> Position:
        if self.grid[position[0]][position[1]] == 0:
            return position
        candidates = [
            (abs(row - position[0]) + abs(col - position[1]), row, col)
            for row in range(GRID_ROWS)
            for col in range(GRID_COLS)
            if self.grid[row][col] == 0
        ]
        _, row, col = min(candidates)
        return (row, col)

    def restart(self) -> None:
        seed = self.rng.randrange(1_000_000)
        self.__init__(self.difficulty, seed)

    def move_player(self, delta: tuple[int, int], now: float) -> None:
        if self.game_over or self.won or self.paused:
            return
        self.last_direction = delta
        self._move_one(delta, now)

    def dash(self, now: float) -> None:
        if self.game_over or self.won or self.paused or now < self.dash_ready_at:
            return
        self.dash_ready_at = now + 2.5
        self.events.append("dash")
        for _ in range(2):
            if not self._move_one(self.last_direction, now):
                break
            if self.won:
                break

    def _move_one(self, delta: tuple[int, int], now: float) -> bool:
        row, col = self.player
        target = (row + delta[0], col + delta[1])
        if not self._open(target):
            return False
        self.player = target
        if target in self.coins:
            self.coins.remove(target)
            self.score += 100
            self.events.append("coin")

        if now - self.last_hit > 1.0:
            for enemy in self.enemies:
                if self.player == enemy:
                    self.health -= 1
                    self.last_hit = now
                    self.events.append("hit")
                    if self.health <= 0:
                        self.game_over = True
                        self.events.append("lose")
                    break
        if not self.coins:
            self.won = True
            self.score += max(0, 500 - int(now - self.started_at) * 5)
            save_score(self.score)
            self.events.append("win")
        return True

    def update_ai(self, now: float) -> None:
        if (
            self.game_over
            or self.won
            or self.paused
            or now - self.last_hit < 0.45
            or now - self.last_ai_move < self.ai_interval
        ):
            return
        self.last_ai_move = now
        for index, enemy in enumerate(self.enemies):
            target = self.player
            if self.prediction_steps:
                predicted = (
                    self.player[0] + self.last_direction[0] * self.prediction_steps,
                    self.player[1] + self.last_direction[1] * self.prediction_steps,
                )
                if self._open(predicted):
                    target = predicted
            trace: dict[str, set[Position]] = {"open": set(), "closed": set(), "goal": {target}}
            path = astar(self.grid, enemy, target, trace)
            if index == 0:
                self.open_nodes = trace["open"] - trace["closed"]
                self.closed_nodes = trace["closed"]
                self.current_path = path or []
                self.predicted_target = target
                g_cost = max(0, len(self.current_path) - 1)
                h_cost = abs(enemy[0] - target[0]) + abs(enemy[1] - target[1])
                self.path_metrics = (g_cost, h_cost, g_cost + h_cost)
            if path and len(path) > 1:
                self.enemies[index] = path[1]
                self.ai_steps += 1
            if self.enemies[index] == self.player and now - self.last_hit > 1.0:
                self.health -= 1
                self.last_hit = now
                self.events.append("hit")
                if self.health <= 0:
                    self.game_over = True
                    self.events.append("lose")

    def _open(self, position: Position) -> bool:
        row, col = position
        return 0 <= row < GRID_ROWS and 0 <= col < GRID_COLS and self.grid[row][col] == 0


def draw_text(surface: pygame.Surface, font: pygame.font.Font, text: str, position: tuple[int, int], color: Color = INK) -> None:
    surface.blit(font.render(text, True, color), position)


class SoundBank:
    """Generate tiny arcade tones at runtime, avoiding external sound assets."""

    def __init__(self, pygame_module: object) -> None:
        self.sounds = {}
        try:
            pygame_module.mixer.init(frequency=44100, size=-16, channels=1)
            self.sounds = {
                "coin": self._tone(pygame_module, 740, 0.10, 0.18, 1040),
                "dash": self._tone(pygame_module, 260, 0.14, 0.12, 520),
                "hit": self._tone(pygame_module, 150, 0.18, 0.20, 75),
                "win": self._tone(pygame_module, 520, 0.35, 0.16, 1040),
                "lose": self._tone(pygame_module, 220, 0.45, 0.16, 65),
            }
        except pygame_module.error:
            pass

    @staticmethod
    def _tone(pygame_module: object, start: int, duration: float, volume: float, end: int):
        sample_rate = 44100
        samples = array("h")
        for index in range(int(sample_rate * duration)):
            progress = index / (sample_rate * duration)
            frequency = start + (end - start) * progress
            envelope = min(1.0, index / 500) * min(1.0, (len(range(int(sample_rate * duration))) - index) / 1200)
            samples.append(int(32767 * volume * envelope * math.sin(2 * math.pi * frequency * index / sample_rate)))
        return pygame_module.mixer.Sound(buffer=samples.tobytes())

    def play_events(self, events: list[str]) -> None:
        for event in events:
            sound = self.sounds.get(event)
            if sound:
                sound.play()
        events.clear()


def draw_game(surface: pygame.Surface, game: HunterGame, fonts: dict[str, pygame.font.Font], now: float) -> None:
    import pygame

    surface.fill(BG)
    title_font, body_font, small_font = fonts["title"], fonts["body"], fonts["small"]
    draw_text(surface, title_font, "AI MAZE HUNTER", (32, 20), CYAN)
    draw_text(surface, small_font, "THE ALGORITHM IS TRACKING YOUR MOVES", (35, 61), MUTED)
    draw_text(surface, small_font, f"{game.difficulty.upper()} MODE", (850, 28), GOLD)
    draw_text(surface, small_font, "WASD / ARROWS MOVE   SPACE DASH   P PAUSE   R RESET", (630, 61), MUTED)

    pygame.draw.rect(surface, PANEL, (BOARD_X - 8, BOARD_Y - 8, BOARD_W + 16, BOARD_H + 16), border_radius=8)
    for row, line in enumerate(game.grid):
        for col, tile in enumerate(line):
            x, y = BOARD_X + col * CELL, BOARD_Y + row * CELL
            rect = pygame.Rect(x, y, CELL, CELL)
            if tile:
                pygame.draw.rect(surface, WALL, rect)
                pygame.draw.line(surface, WALL_EDGE, (x + 2, y + 2), (x + CELL - 2, y + 2), 1)
            else:
                pygame.draw.rect(surface, (17, 27, 42), rect)
                pygame.draw.line(surface, (22, 34, 51), (x, y + CELL - 1), (x + CELL, y + CELL - 1), 1)

    # The first hunter's latest A* search is rendered as an explainable overlay.
    for row, col in game.closed_nodes:
        if game.grid[row][col] == 0:
            pygame.draw.rect(surface, (92, 45, 57), (BOARD_X + col * CELL + 8, BOARD_Y + row * CELL + 8, 16, 16))
    for row, col in game.open_nodes:
        if game.grid[row][col] == 0:
            pygame.draw.rect(surface, (35, 83, 112), (BOARD_X + col * CELL + 8, BOARD_Y + row * CELL + 8, 16, 16))
    if len(game.current_path) > 1:
        points = [
            (BOARD_X + col * CELL + CELL // 2, BOARD_Y + row * CELL + CELL // 2)
            for row, col in game.current_path
        ]
        pygame.draw.lines(surface, GOLD, False, points, 4)
        for point in points[1:-1]:
            pygame.draw.circle(surface, GOLD, point, 4)

    for row, col in game.coins:
        center = (BOARD_X + col * CELL + CELL // 2, BOARD_Y + row * CELL + CELL // 2)
        pygame.draw.circle(surface, GOLD, center, 6)
        pygame.draw.circle(surface, (255, 229, 139), center, 3)

    for row, col in game.enemies:
        x, y = BOARD_X + col * CELL, BOARD_Y + row * CELL
        pygame.draw.rect(surface, CORAL, (x + 7, y + 7, CELL - 14, CELL - 10), border_radius=8)
        pygame.draw.circle(surface, BG, (x + 13, y + 14), 3)
        pygame.draw.circle(surface, BG, (x + 22, y + 14), 3)
        pygame.draw.line(surface, CORAL, (x + 5, y + 8), (x + 1, y + 3), 2)
        pygame.draw.line(surface, CORAL, (x + CELL - 5, y + 8), (x + CELL - 1, y + 3), 2)

    row, col = game.player
    center = (BOARD_X + col * CELL + CELL // 2, BOARD_Y + row * CELL + CELL // 2)
    pygame.draw.circle(surface, CYAN, center, 11)
    pygame.draw.circle(surface, (180, 255, 246), center, 5)

    side_x = BOARD_X + BOARD_W + 28
    pygame.draw.rect(surface, PANEL, (side_x, BOARD_Y, 160, BOARD_H), border_radius=8)
    draw_text(surface, small_font, "RUN STATUS", (side_x + 16, BOARD_Y + 18), MUTED)
    draw_text(surface, body_font, f"{game.score:05d}", (side_x + 16, BOARD_Y + 45), GOLD)
    draw_text(surface, small_font, "SCORE", (side_x + 17, BOARD_Y + 85), MUTED)
    draw_text(surface, body_font, "♥" * game.health + "·" * (game.max_health - game.health), (side_x + 16, BOARD_Y + 110), CORAL)
    draw_text(surface, small_font, "HEALTH", (side_x + 17, BOARD_Y + 148), MUTED)
    elapsed = max(0, int(now - game.started_at))
    draw_text(surface, body_font, f"{elapsed:02d}s", (side_x + 16, BOARD_Y + 174), INK)
    draw_text(surface, small_font, "TIME", (side_x + 17, BOARD_Y + 212), MUTED)
    draw_text(surface, body_font, f"{len(game.coins):02d}", (side_x + 16, BOARD_Y + 238), GOLD)
    draw_text(surface, small_font, "COINS LEFT", (side_x + 17, BOARD_Y + 276), MUTED)
    dash_label = "READY" if now >= game.dash_ready_at else f"{game.dash_ready_at - now:0.1f}s"
    draw_text(surface, body_font, dash_label, (side_x + 16, BOARD_Y + 302), CYAN)
    draw_text(surface, small_font, "DASH", (side_x + 17, BOARD_Y + 331), MUTED)
    draw_text(surface, small_font, "A* SEARCH", (side_x + 16, BOARD_Y + 369), MUTED)
    draw_text(surface, small_font, f"{game.ai_steps:04d} STEPS", (side_x + 16, BOARD_Y + 391), CYAN)
    g_cost, h_cost, f_cost = game.path_metrics
    draw_text(surface, small_font, f"G {g_cost:02d}  H {h_cost:02d}", (side_x + 16, BOARD_Y + 425), INK)
    draw_text(surface, small_font, f"F {f_cost:02d}  PREDICT", (side_x + 16, BOARD_Y + 445), GOLD if game.prediction_steps else MUTED)
    pygame.draw.rect(surface, (35, 83, 112), (side_x + 16, BOARD_Y + 478, 10, 10))
    draw_text(surface, small_font, "OPEN", (side_x + 32, BOARD_Y + 475), MUTED)
    pygame.draw.rect(surface, (92, 45, 57), (side_x + 78, BOARD_Y + 478, 10, 10))
    draw_text(surface, small_font, "CLOSED", (side_x + 94, BOARD_Y + 475), MUTED)

    if game.game_over or game.won:
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((7, 12, 20, 215))
        surface.blit(overlay, (0, 0))
        color = CYAN if game.won else CORAL
        card = pygame.Rect(WIDTH // 2 - 245, HEIGHT // 2 - 155, 490, 310)
        pygame.draw.rect(surface, PANEL, card, border_radius=12)
        pygame.draw.rect(surface, color, (card.x, card.y, card.width, 6), border_radius=12)
        headline = "MAZE CLEARED" if game.won else "HUNT TERMINATED"
        subtitle = "YOU OUTSMARTED THE ALGORITHM" if game.won else "THE HUNTERS FOUND YOU"
        draw_text(surface, title_font, headline, (card.x + 34, card.y + 34), color)
        draw_text(surface, small_font, subtitle, (card.x + 36, card.y + 78), MUTED)
        draw_text(surface, small_font, "FINAL SCORE", (card.x + 36, card.y + 123), MUTED)
        draw_text(surface, body_font, f"{game.score:05d}", (card.x + 36, card.y + 144), GOLD)
        draw_text(surface, small_font, "TIME", (card.x + 220, card.y + 123), MUTED)
        draw_text(surface, body_font, f"{elapsed:02d}s", (card.x + 220, card.y + 144), INK)
        draw_text(surface, small_font, "A* STEPS", (card.x + 345, card.y + 123), MUTED)
        draw_text(surface, body_font, f"{game.ai_steps:04d}", (card.x + 345, card.y + 144), CYAN)
        pygame.draw.line(surface, PANEL_LIGHT, (card.x + 34, card.y + 196), (card.right - 34, card.y + 196), 1)
        draw_text(surface, small_font, "R  RUN IT BACK", (card.x + 36, card.y + 222), INK)
        draw_text(surface, small_font, "ESC  EXIT", (card.x + 300, card.y + 222), MUTED)
    elif game.paused:
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((7, 12, 20, 175))
        surface.blit(overlay, (0, 0))
        draw_text(surface, title_font, "RUN PAUSED", (WIDTH // 2 - 105, HEIGHT // 2 - 32), GOLD)
        draw_text(surface, body_font, "PRESS P TO RESUME", (WIDTH // 2 - 120, HEIGHT // 2 + 18), INK)


def show_start_menu(surface: pygame.Surface, pygame_module: object, fonts: dict[str, pygame.font.Font]) -> bool:
    options = ["START GAME", "HOW TO PLAY", "LEADERBOARD", "EXIT"]
    selected = 0
    clock = pygame_module.time.Clock()
    while True:
        for event in pygame_module.event.get():
            if event.type == pygame_module.QUIT:
                return False
            if event.type == pygame_module.KEYDOWN:
                if event.key in (pygame_module.K_UP, pygame_module.K_w):
                    selected = (selected - 1) % len(options)
                elif event.key in (pygame_module.K_DOWN, pygame_module.K_s):
                    selected = (selected + 1) % len(options)
                elif event.key in (pygame_module.K_RETURN, pygame_module.K_SPACE):
                    if selected == 0:
                        return True
                    if selected == 1:
                        show_help(surface, pygame_module, fonts)
                    elif selected == 2:
                        show_leaderboard(surface, pygame_module, fonts)
                    else:
                        return False

        surface.fill(BG)
        draw_text(surface, fonts["title"], "AI MAZE HUNTER", (WIDTH // 2 - 140, 105), CYAN)
        draw_text(surface, fonts["small"], "ESCAPE THE A* HUNTERS", (WIDTH // 2 - 92, 145), MUTED)
        for index, label in enumerate(options):
            y = 220 + index * 62
            color = CYAN if index == selected else INK
            pygame_module.draw.rect(surface, PANEL_LIGHT if index == selected else PANEL, (WIDTH // 2 - 180, y, 360, 46), border_radius=8)
            draw_text(surface, fonts["body"], label, (WIDTH // 2 - 125, y + 10), color)
        draw_text(surface, fonts["small"], "UP / DOWN SELECT     ENTER CONFIRM", (WIDTH // 2 - 145, 510), MUTED)
        pygame_module.display.flip()
        clock.tick(30)


def show_help(surface: pygame.Surface, pygame_module: object, fonts: dict[str, pygame.font.Font]) -> None:
    clock = pygame_module.time.Clock()
    while True:
        for event in pygame_module.event.get():
            if event.type == pygame_module.QUIT:
                return
            if event.type == pygame_module.KEYDOWN and event.key in (pygame_module.K_ESCAPE, pygame_module.K_RETURN, pygame_module.K_SPACE):
                return
        surface.fill(BG)
        draw_text(surface, fonts["title"], "HOW TO PLAY", (WIDTH // 2 - 100, 100), CYAN)
        lines = [
            "Collect every coin before the A* hunters catch you.",
            "WASD / ARROWS   move one tile",
            "SPACE           dash two tiles",
            "P               pause the hunt",
            "The overlay shows open, closed, and final A* nodes.",
            "G = movement cost     H = Manhattan estimate     F = G + H",
            "",
            "PRESS ESC OR ENTER TO RETURN",
        ]
        for index, line in enumerate(lines):
            draw_text(surface, fonts["small"], line, (WIDTH // 2 - 230, 205 + index * 34), INK if index < 6 else MUTED)
        pygame_module.display.flip()
        clock.tick(30)


def show_leaderboard(surface: pygame.Surface, pygame_module: object, fonts: dict[str, pygame.font.Font]) -> None:
    clock = pygame_module.time.Clock()
    while True:
        for event in pygame_module.event.get():
            if event.type == pygame_module.QUIT:
                return
            if event.type == pygame_module.KEYDOWN and event.key in (pygame_module.K_ESCAPE, pygame_module.K_RETURN, pygame_module.K_SPACE):
                return
        surface.fill(BG)
        draw_text(surface, fonts["title"], "TOP HUNTERS", (WIDTH // 2 - 105, 100), GOLD)
        scores = load_scores()
        if not scores:
            draw_text(surface, fonts["body"], "NO SCORES YET", (WIDTH // 2 - 90, 235), MUTED)
        else:
            for index, score in enumerate(scores):
                draw_text(surface, fonts["body"], f"{index + 1}.   {score:,}", (WIDTH // 2 - 100, 190 + index * 48), INK)
        draw_text(surface, fonts["small"], "PRESS ESC OR ENTER TO RETURN", (WIDTH // 2 - 120, 510), MUTED)
        pygame_module.display.flip()
        clock.tick(30)


def choose_difficulty(surface: pygame.Surface, pygame_module: object, fonts: dict[str, pygame.font.Font]) -> str | None:
    options = [
        ("easy", "5 health  /  8 coins  /  slow hunter"),
        ("medium", "3 health  /  10 coins  /  balanced"),
        ("hard", "2 health  /  12 coins  /  2 fast hunters"),
        ("extreme", "1 health  /  14 coins  /  3 predictive hunters"),
    ]
    selected = 1
    clock = pygame_module.time.Clock()
    while True:
        for event in pygame_module.event.get():
            if event.type == pygame_module.QUIT:
                return None
            if event.type == pygame_module.KEYDOWN:
                if event.key in (pygame_module.K_UP, pygame_module.K_w):
                    selected = (selected - 1) % len(options)
                elif event.key in (pygame_module.K_DOWN, pygame_module.K_s):
                    selected = (selected + 1) % len(options)
                elif event.key in (pygame_module.K_RETURN, pygame_module.K_SPACE):
                    return options[selected][0]
                elif event.key in (pygame_module.K_1, pygame_module.K_2, pygame_module.K_3):
                    return options[event.key - pygame_module.K_1][0]

        surface.fill(BG)
        draw_text(surface, fonts["title"], "AI MAZE HUNTER", (WIDTH // 2 - 140, 110), CYAN)
        draw_text(surface, fonts["small"], "CHOOSE YOUR RUN", (WIDTH // 2 - 67, 155), MUTED)
        for index, (name, details) in enumerate(options):
            y = 230 + index * 76
            color = CYAN if index == selected else INK
            pygame_module.draw.rect(surface, PANEL_LIGHT if index == selected else PANEL, (WIDTH // 2 - 230, y, 460, 58), border_radius=8)
            draw_text(surface, fonts["body"], f"{index + 1}  {name.upper()}", (WIDTH // 2 - 205, y + 8), color)
            draw_text(surface, fonts["small"], details, (WIDTH // 2 - 205, y + 36), MUTED)
        draw_text(surface, fonts["small"], "UP / DOWN TO SELECT     ENTER OR SPACE TO START", (WIDTH // 2 - 190, 570), MUTED)
        pygame_module.display.flip()
        clock.tick(30)


def run_game(difficulty: str | None) -> None:
    import pygame

    pygame.init()
    pygame.display.set_caption("AI Maze Hunter")
    surface = pygame.display.set_mode((WIDTH, HEIGHT))
    fonts = {
        "title": pygame.font.SysFont("consolas", 27, bold=True),
        "body": pygame.font.SysFont("consolas", 22, bold=True),
        "small": pygame.font.SysFont("consolas", 13, bold=True),
    }
    if not show_start_menu(surface, pygame, fonts):
        pygame.quit()
        return
    if difficulty is None:
        difficulty = choose_difficulty(surface, pygame, fonts)
        if difficulty is None:
            pygame.quit()
            return
    sounds = SoundBank(pygame)
    game = HunterGame(difficulty)
    clock = pygame.time.Clock()
    running = True
    while running:
        now = pygame.time.get_ticks() / 1000
        if not game.started_at:
            game.started_at = now
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_p:
                    game.paused = not game.paused
                elif event.key == pygame.K_r:
                    game.restart()
                    game.started_at = now
                elif event.key == pygame.K_SPACE:
                    game.dash(now)
                elif event.key in (pygame.K_UP, pygame.K_w):
                    game.move_player((-1, 0), now)
                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    game.move_player((1, 0), now)
                elif event.key in (pygame.K_LEFT, pygame.K_a):
                    game.move_player((0, -1), now)
                elif event.key in (pygame.K_RIGHT, pygame.K_d):
                    game.move_player((0, 1), now)
        if not game.game_over and not game.won and int(now * 4) != int((now - 1 / 60) * 4):
            game.update_ai(now)
        sounds.play_events(game.events)
        draw_game(surface, game, fonts, now)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


def smoke_test() -> None:
    game = HunterGame("hard", seed=7)
    path = astar(game.grid, game.enemies[0], game.player)

    assert path and path[0] == game.enemies[0] and path[-1] == game.player
    assert all(game.grid[row][col] == 0 for row, col in path)
    print(f"Smoke test passed: {len(path) - 1} steps")


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Maze Hunter")
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard", "extreme"],
        default=None,
        help="Skip the difficulty selection screen",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run the A* smoke test instead of starting the game",
    )

    args = parser.parse_args()

    if args.test:
        smoke_test()
    else:
        run_game(args.difficulty)


if __name__ == "__main__":
    main()