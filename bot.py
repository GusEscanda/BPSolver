import os
import numpy as np
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import cv2
from io import BytesIO
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes, CommandHandler

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
STATS = dict()

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
        return solved, messages, result_image, 'no_grid_detected'
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
            return solved, messages, result_image, 'solved_queens'
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
            return solved, messages, result_image, 'solved_tango'
        messages.append(f"Tango: {msg}")

    return solved, messages, result_image, 'puzzle_unsolved'

# --- Stats ---
def update_stats(user, stat):
    username = user.username or f"{user.first_name} {user.last_name or ''}".strip()
    today = datetime.strftime(datetime.now(ZoneInfo("America/Argentina/Buenos_Aires")), '%Y-%m-%d')
    STATS.setdefault(today, {}).setdefault(username, {}).setdefault(stat, 0)
    STATS[today][username][stat] += 1

# --- Handlers ---

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    update_stats(update.effective_user, 'image')

    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    bio = BytesIO()
    await file.download_to_memory(out=bio)
    bio.seek(0)

    file_bytes = np.asarray(bytearray(bio.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    try:
        solved, messages, result_image, stat = try_solutions(img)
        update_stats(update.effective_user, stat)
        for msg in messages:
            print(msg)
            await update.message.reply_text(msg)

        if result_image is not None:
            _, buffer = cv2.imencode(".jpg", result_image)
            output_io = BytesIO(buffer.tobytes())
            output_io.seek(0)
            await update.message.reply_photo(photo=output_io)

        if solved:
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
    update_stats(update.effective_user, 'text')

async def handle_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != OWNER_CHAT_ID:
        await update.message.reply_text("Sorry, this command is not available.")
        return

    pretty_stats = json.dumps(STATS, indent=4, ensure_ascii=False)
    
    if len(pretty_stats) < 3800:
        await update.message.reply_text(f"📊 Estadísticas:\n\n{pretty_stats}")
    else:
        await update.message.reply_text("📊 Estadísticas enviadas como archivo.")
        await update.message.reply_document(
            document=BytesIO(pretty_stats.encode()),
            filename="stats.json"
        )

async def handle_clearstats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != OWNER_CHAT_ID:
        await update.message.reply_text("Sorry, this command is not available.")
        return

    try:
        if not context.args or len(context.args) != 1:
            cutoff = datetime.strftime(datetime.now(ZoneInfo("America/Argentina/Buenos_Aires")), '%Y-%m-%d')
        else:
            cutoff = datetime.strptime(context.args[0], "%Y-%m-%d").date()
        deleted = []
        for day in STATS:
            if datetime.strptime(day, "%Y-%m-%d").date() <= cutoff:
                del STATS[day]
                deleted.append(day)
        await update.message.reply_text(f"Deleted stats for dates: {', '.join(deleted) or 'none'}")
    except Exception as e:
        await update.message.reply_text(f"Invalid date format or error: {e}")


# --- Bot setup ---
if __name__ == '__main__':
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("stats", handle_stats))
    app.add_handler(CommandHandler("clearstats", handle_clearstats))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))
    app.run_polling()
