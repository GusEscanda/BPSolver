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
        grid = Grid(image=img)
        threshold_values = [70, 100, 130, 160]
        min_line_length = [300, 350, 400]
        max_line_gap = [8, 12]
        max_groupping_dist = 3

        solved = False
        if not solved and grid.image is not None:
            print('Trying to find a Queens puzzle')
            await update.message.reply_text('Trying to find a Queens puzzle')
            puzzle = Queens(
                grid, 
                resize_width=500, 
                threshold_value= threshold_values,
                min_line_length= min_line_length, 
                max_line_gap= max_line_gap,
                max_groupping_dist= max_groupping_dist,
                min_valid_n= 6, max_valid_n= 16
            )
            print(puzzle.msg)
            await update.message.reply_text(puzzle.msg)
            solver = BPSolver(puzzle, 1000, timedelta(seconds=120), 20)
            msg = solver.solve()
            print(msg)
            await update.message.reply_text(msg)
            solved = solver.solved
        if not solved and grid.image is not None:
            print('Trying to find a Tango puzzle')
            await update.message.reply_text('Trying to find a Tango puzzle')
            puzzle = Tango(
                grid, 
                resize_width=500, 
                threshold_value= threshold_values,
                min_line_length= min_line_length, 
                max_line_gap= max_line_gap,
                max_groupping_dist= max_groupping_dist,
                eq_filename= 'templates_eq.png',
                x_filename='templates_x.png'
            )
            print(puzzle.msg)
            await update.message.reply_text(puzzle.msg)
            solver = BPSolver(puzzle, 1000, timedelta(seconds=120), 20)
            msg = solver.solve()
            print(msg)
            await update.message.reply_text(msg)
            solved = solver.solved

        if grid.image is not None:
            if solved:
                result_img = puzzle.draw_solution(grid.image)
            elif grid.n > 0:
                result_img = draw_grid(grid.image, grid.x_axis, grid.y_axis)
            else:
                result_img = grid.image

            # Send result with Telegram
            _, buffer = cv2.imencode(".jpg", result_img)
            output_io = BytesIO(buffer.tobytes())
            output_io.seek(0)

            await update.message.reply_photo(photo=output_io)

    except Exception as e:
        await update.message.reply_text(f"Error processing the image: {e}")

if __name__ == '__main__':
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))
    app.run_polling()
