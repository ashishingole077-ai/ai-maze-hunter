# AI Maze Hunter

A small Pygame game where you collect every coin in a generated maze while one or more enemies use A* pathfinding to hunt you down.

## Run

```powershell
py -m pip install -r requirements.txt
py app.py
```

Launching without an option opens the start menu. Choose Start Game, How to Play, Leaderboard, or Exit. The difficulty picker supports Easy, Medium, Hard, and Extreme.

Choose a mode directly with `py app.py --difficulty easy`, `medium`, `hard`, or `extreme`. Easy gives you 5 health, 8 coins, and a slower hunter. Medium gives 3 health and balanced rules. Hard gives 2 health, 12 coins, a faster hunter, and a second enemy. Extreme gives 1 health, 14 coins, three fast hunters, and predictive A* targeting.

During a run, the board visualizes the first hunter's latest search: blue tiles are open nodes, red tiles are explored/closed nodes, and the gold line is the selected route. The HUD reports `G(n)`, `H(n)`, and `F(n) = G(n) + H(n)`.

The leaderboard screen reads the persistent top-five scores from `leaderboard.json`. The smoke test checks the generated maze and shortest path without opening a window:

```powershell
py app.py --test
```

## Controls

- `WASD` or arrow keys: move one tile
- `Space`: dash two tiles in your last movement direction; cooldown is 2.5 seconds
- `P`: pause or resume the run
- `R`: generate a new maze
- `Esc`: quit

The game generates sound effects at runtime for coins, dashes, hits, wins, and losses, so no audio files are needed.

Collecting a coin adds 100 points. Clear the board to finish the run; finishing faster gives a time bonus. Dashing can cross open corridors quickly, but enemies keep moving with A* while you play. Hard mode adds a second A* enemy. Top scores are stored locally in `leaderboard.json` after a win.