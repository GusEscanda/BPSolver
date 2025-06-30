import os
import numpy as np
from datetime import datetime, timedelta
import cv2
from io import BytesIO
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes

from grids import Grid, draw_grid
from queens import Queens
from tango import Tango
from BPSolver import BPSolver

# --- Load environment variables ---
if os.getenv("RAILWAY_ENVIRONMENT") is None:
    print('Loading .env')
    from dotenv import load_dotenv
    load_dotenv()
else:
    print('Using Railway variables')

TOKEN = os.getenv("BOT_TOKEN")
OWNER_CHAT_ID = int(os.getenv("OWNER_CHAT_ID"))

# --- Puzzle stats ---
STATS = {
    "received": 0,
    "solved": 0
}

# --- Puzzle solver logic ---
def try_solutions(image):
    solved, messages, result_image = False, [], None

    # Try to detect the puzzle grid in the image
    grid = Grid(image=image)
    grid.preprocess_image(resize_width=500)
    grid.find_grid(
        min_line_length=[250, 300, 350, 400],
        max_line_gap=[8, 12],
        max_groupping_dist=3,
        min_valid_n=5,
        max_valid_n=15
    )

    if grid.n == 0:
        messages.append("Couldn't detect the board grid in the image.")
        return solved, messages, result_image
    else:
        result_image = draw_grid(grid.image, grid.x_axis, grid.y_axis)

    # Try solving as Queens puzzle
    puzzle = Queens(grid)
    solver = BPSolver(puzzle, 1000, timedelta(seconds=120), 20)

    if puzzle.msg != 'OK':
        messages.append(f"Queens: {puzzle.msg}")
    else:
        msg = solver.solve()
        solved = solver.solved
        if solved:
            messages = ["Queens puzzle detected — here's the solution!"]
            result_image = puzzle.draw_solution(grid.image)
            return solved, messages, result_image
        messages.append(f"Queens: {msg}")

    # Try solving as Tango puzzle
    puzzle = Tango(
        grid,
        eq_filename='templates_eq.png',
        x_filename='templates_x.png'
    )
    solver = BPSolver(puzzle, 1000, timedelta(seconds=120), 20)

    if puzzle.msg != 'OK':
        messages.append(f"Tango: {puzzle.msg}")
    else:
        msg = solver.solve()
        solved = solver.solved
        if solved:
            messages = ["Tango puzzle detected — here's the solution!"]
            result_image = puzzle.draw_solution(grid.image)
            return solved, messages, result_image
        messages.append(f"Tango: {msg}")

    return solved, messages, result_image

# --- Handlers ---

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    STATS["received"] += 1

    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    bio = BytesIO()
    await file.download_to_memory(out=bio)
    bio.seek(0)

    file_bytes = np.asarray(bytearray(bio.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    try:
        solved, messages, result_image = try_solutions(img)
        for msg in messages:
            print(msg)
            await update.message.reply_text(msg)

        if result_image is not None:
            _, buffer = cv2.imencode(".jpg", result_image)
            output_io = BytesIO(buffer.tobytes())
            output_io.seek(0)
            await update.message.reply_photo(photo=output_io)

        if solved:
            STATS["solved"] += 1
            await update.message.reply_text(
                "Check out the code on GitHub:\n"
                "https://github.com/GusEscanda/BPSolver\n\n"
                "Includes solvers, the bot, and a Jupyter notebook to test everything. Enjoy!\n\n"
                "— Gustavo Escandarani"
            )
        else:
            # Send original image to owner
            await context.bot.send_message(chat_id=OWNER_CHAT_ID, text="Unsolved puzzle received.")
            _, buffer = cv2.imencode(".jpg", img)
            output_io = BytesIO(buffer.tobytes())
            output_io.seek(0)
            await context.bot.send_photo(chat_id=OWNER_CHAT_ID, photo=output_io)

    except Exception as e:
        await update.message.reply_text(f"An error occurred while processing the image: {e}")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Send me a screenshot of an unsolved Queens or Tango puzzle.\n"
        "I'll try to detect the grid and, if I can, I'll solve it and send the solution back to you!"
    )

async def handle_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != OWNER_CHAT_ID:
        await update.message.reply_text("Sorry, this command is not available.")
        return

    await update.message.reply_text(
        f"Puzzles received: {STATS['received']}\n"
        f"Puzzles solved: {STATS['solved']}"
    )

# --- Bot setup ---
if __name__ == '__main__':
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))
    app.add_handler(MessageHandler(filters.TEXT & filters.Regex("^/stats$"), handle_stats))
    app.add_handler(MessageHandler(filters.TEXT, handle_text))
    app.run_polling()
