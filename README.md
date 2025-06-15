# 🧩 Board Puzzle Solver (BPSolver)

This project provides a framework for solving board-based puzzles, where the goal is to place objects on a grid according to certain rules. Examples include chess-based puzzles like Queens or constraint puzzles like Tango.

It includes:

- A general-purpose puzzle solver (`BPSolver`)
- An abstract puzzle interface (`BoardPuzzle`)
- Puzzle-specific implementations (`Queens`, `Tango`)
- Tools for extracting board data from images (`Grid`)
- A Jupyter notebook demonstrating the system
- A Telegram bot for solving puzzles directly from your phone

---

## 🎯 Project Goals

The main goal of this project is to offer a reusable and extensible base for solving any kind of board puzzle. If a puzzle can be described as placing items on a grid under a set of rules, it can likely be solved using this system.

Examples:

- Place 8 queens on a board without attacking each other.
- Fill a grid with symbols so that constraints between adjacent cells are met.

By defining your own class that implements the `BoardPuzzle` interface, you can plug it into the `BPSolver` and get a working solution engine with minimal additional logic.

---

## 🧠 Core Components

### `BoardPuzzle`

An abstract interface that defines the required methods for puzzle solving. Any puzzle must implement this interface to be solvable with `BPSolver`.

### `BPSolver`

The engine that solves any `BoardPuzzle`. It uses a mix of logical deductions and backtracking with configurable depth, time, and move limits.

> **Strategy:**
>
> 1. Deduce all mandatory moves.
> 2. If stuck, make an educated guess and recurse.
> 3. Backtrack when a dead-end is reached.

### `Grid`

Reads and analyzes puzzle images to extract the board layout.

- Uses OpenCV for image preprocessing and line detection.
- Clustering methods like KMeans and DBSCAN help identify repeated patterns (e.g., symbols).

### `Queens` / `Tango`

Concrete implementations of `BoardPuzzle`.

- `Queens`: Places N queens without threats.
- `Tango`: Solves grid-based symbol puzzles with constraint lines between cells.

---

## 🤖 Telegram Bot

Send a picture of a puzzle, and the bot will attempt to solve it and send back the result.

### Example:

- You take a screenshot of a Tango puzzle.
- Send it to the bot.
- Receive the solved board image back.

> Built with `python-telegram-bot`, OpenCV, and the BPSolver engine.

> 🛡️ **Token security note:** Be sure to load your bot token from a `.env` file (already set up in the code via `dotenv`) so it is not exposed in public repositories.

---

## 📓 Jupyter Notebook

The repository includes a Jupyter notebook that demonstrates the full workflow, from loading an image to solving the puzzle.

### 🧪 How to Open and Use the Notebook

To run it:

1. Make sure you have **Jupyter** installed. If not, you can install it with:

   ```bash
   pip install notebook
   ```

2. Launch the notebook server from the project root:

   ```bash
   jupyter notebook
   ```

3. Open the file named:

   ```
   notebook.ipynb
   ```

4. Run the cells step-by-step to:

   - Load board images
   - Detect the grid
   - Initialize puzzle objects
   - Visualize solutions (including batch-solving multiple boards)

You can also modify the notebook to test new puzzles or experiment with puzzle-solving strategies.

---

## 📁 Repository Structure

```
BPSolver/
├── BPSolver.py          # Solver engine
├── BoardPuzzle.py       # Abstract puzzle interface
├── queens.py            # Queens puzzle implementation
├── tango.py             # Tango puzzle implementation
├── grids.py             # Grid detection and analysis
├── bot.py               # Telegram bot logic
├── notebook.ipynb       # Interactive exploration notebook
├── templates_eq.png     # Template for '=' sign (Tango)
├── templates_x.png      # Template for 'x' sign (Tango)
└── .env                 # Bot token (not tracked by git)
```

---

## ✨ Author

Developed by **Gustavo Escandarani**

📎 Full source code available at: [github.com/GusEscanda/BPSolver](https://github.com/GusEscanda/BPSolver)

Feel free to explore the code, open issues, or suggest puzzles to implement!
