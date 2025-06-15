import os
from dotenv import load_dotenv
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

load_dotenv()
TOKEN = os.getenv("BOT_TOKEN")

def try_solutions(image):
    messages, result_image = ([], None)

    # Find the puzzle's grid
    grid = Grid(image=image)
    grid.preprocess_image(
        resize_width=500,
        threshold_values=[70, 100, 130, 160]
    )
    grid.find_grid(
        min_line_length=[300, 350, 400], 
        max_line_gap=[8, 12], 
        max_groupping_dist=3,
        min_valid_n=5,
        max_valid_n=15
    )
    if grid.n == 0:
        messages.append("Could't find any board")
        return (messages, result_image)
    else:
        result_image = draw_grid(grid.image, grid.x_axis, grid.y_axis)

    # Try a Queens puzzle

    puzzle = Queens(grid)
    solver = BPSolver(puzzle, 1000, timedelta(seconds=120), 20)

    if puzzle.msg != 'OK':
        messages.append(f'Queens: {puzzle.msg}')
    else:
        msg = solver.solve()
        if solver.solved:
            messages = ["Queens board found, here's your solution!!"]
            result_image = puzzle.draw_solution(grid.image)
            return (messages, result_image)
        messages.append(f'Queens: {msg}')

    # Try a Tango puzzle

    puzzle = Tango(
        grid,
        eq_filename='templates_eq.png',
        x_filename='templates_x.png'
    )
    solver = BPSolver(puzzle, 1000, timedelta(seconds=120), 20)

    if puzzle.msg != 'OK':
        messages.append(f'Tango: {puzzle.msg}')
    else:
        msg = solver.solve()
        if solver.solved:
            messages = ["Tango board found, here's your solution!!"]
            result_image = puzzle.draw_solution(grid.image)
            return (messages, result_image)    
        messages.append(f'Tango: {msg}')

    return (messages, result_image)


async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    bio = BytesIO()
    await file.download_to_memory(out=bio)
    bio.seek(0)

    # Load image with OpenCV, BytesIO
    file_bytes = np.asarray(bytearray(bio.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    try:
        messages, result_image = try_solutions(img)
        for msg in messages:
            print(msg)
            await update.message.reply_text(msg)
        if result_image is not None:
            _, buffer = cv2.imencode(".jpg", result_image)
            output_io = BytesIO(buffer.tobytes())
            output_io.seek(0)
            await update.message.reply_photo(photo=output_io)
        
    except Exception as e:
        await update.message.reply_text(f"Error processing the image: {e}")

if __name__ == '__main__':
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))
    app.run_polling()
