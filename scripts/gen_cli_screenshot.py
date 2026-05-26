"""Generate terminal-style CLI screenshot for README."""
import textwrap
from PIL import Image, ImageDraw, ImageFont
import subprocess, sys, os

WIDTH = 820
HEIGHT = 600
BG = "#1e1e2e"
FG = "#cdd6f4"
GREEN = "#a6e3a1"
BLUE = "#89b4fa"
YELLOW = "#f9e2af"
GRAY = "#585b70"
PADDING = 28
LINE_H = 22
FONT_SIZE = 13

OUT = os.path.join(os.path.dirname(__file__), "..", "screenshots", "cli.png")

try:
    font = ImageFont.truetype("CascadiaCode.ttf", FONT_SIZE)
except OSError:
    try:
        font = ImageFont.truetype("consola.ttf", FONT_SIZE)
    except OSError:
        font = ImageFont.load_default()

img = Image.new("RGB", (WIDTH, HEIGHT), BG)
draw = ImageDraw.Draw(img)

y = PADDING
x = PADDING + 4

draw.text((x, y), "trading-agent --help", fill=GREEN, font=font)
y += LINE_H + 6

lines = [
    (FG, ""),
    (FG, " Usage: trading-agent [OPTIONS] COMMAND [ARGS]..."),
    (FG, ""),
    (GRAY, "╭─ Options ─────────────────────────────────────────────────────╮"),
    (FG, "│ --install-completion    Install completion for current shell.  │"),
    (FG, "│ --show-completion       Show completion for current shell.     │"),
    (FG, "│ --help                  Show this message and exit.           │"),
    (GRAY, "╰────────────────────────────────────────────────────────────────╯"),
    (FG, ""),
    (GRAY, "╭─ Commands ───────────────────────────────────────────────────╮"),
    (YELLOW, "│ backtest    Run and compare backtests                          │"),
    (FG, "│   run           Single backtest with LLM agent                  │"),
    (FG, "│   compare       Side-by-side model comparison                   │"),
    (FG, "│   compare-risk  Risk-averse vs risk-taker                       │"),
    (FG, ""),
    (YELLOW, "│ cgan        Train CGAN world agent and simulate LOB markets    │"),
    (FG, "│   train        Train Generator + Discriminator on LOB data      │"),
    (FG, "│   generate     Generate action sequences from trained model     │"),
    (FG, "│   simulate     Run interactive LOB simulation                   │"),
    (FG, ""),
    (YELLOW, "│ pinn        Train PINN and generate market data                │"),
    (FG, "│   train        Train MarketPINN with Black-Scholes PDE          │"),
    (FG, "│   generate     Generate synthetic price paths                   │"),
    (FG, ""),
    (YELLOW, "│ serve       Start web dashboard (FastAPI + HTMX)               │"),
    (GRAY, "╰────────────────────────────────────────────────────────────────╯"),
]

for color, text in lines:
    draw.text((x, y), text, fill=color, font=font)
    y += LINE_H

img.save(OUT)
print(f"Saved CLI screenshot to {OUT}")
