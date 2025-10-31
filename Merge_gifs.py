#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stitch_gifs.py

- Looks in a specified run folder (e.g., results/gifs/<RUN_FOLDER_NAME>).
- Finds all individual GIF files (like pendulum_th0_...).
- Sorts them by name.
- Stitches them together, one after another, into a single GIF.
- Saves the combined GIF as '_stitched_sweep.gif' in the same folder.
"""

# =================== USER SETTINGS ===================
# *** This MUST match the folder name from your other scripts ***
RUN_FOLDER_NAME = "saved_model_for_figure"

# --- Output settings ---
# This is the name of the final combined GIF
OUTPUT_GIF_NAME = "_stitched_sweep.gif"
# Desired FPS for the *final* stitched GIF
FPS = 20
# =====================================================

import os
import glob
from PIL import Image

# ---------- Resolve paths ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GIF_DIR = "results/gifs"
INPUT_DIR = os.path.join(BASE_DIR, GIF_DIR, RUN_FOLDER_NAME)
OUTPUT_PATH = os.path.join(INPUT_DIR, OUTPUT_GIF_NAME)

if not os.path.exists(INPUT_DIR):
    raise FileNotFoundError(f"Input directory not found. Did you run create_gifs.py?\nChecked: {INPUT_DIR}")

# ---------- 1. Find all individual GIFs ----------
# Find all .gif files
gif_pattern = os.path.join(INPUT_DIR, "pendulum_th0_*.gif")
gif_files = glob.glob(gif_pattern)

# Sort them alphabetically (which works for your naming scheme)
gif_files.sort()

if not gif_files:
    print(f"Error: No GIF files found in {INPUT_DIR} matching the pattern 'pendulum_th0_*.gif'")
    exit()

print(f"Found {len(gif_files)} GIFs to stitch:")
for f in gif_files:
    print(f"  - {os.path.basename(f)}")

# ---------- 2. Extract all frames ----------
all_frames = []
for gif_path in gif_files:
    print(f"Processing {os.path.basename(gif_path)}...")
    img = Image.open(gif_path)

    # Iterate over each frame in the GIF
    frame_index = 0
    while True:
        try:
            img.seek(frame_index)
            # Need to copy the frame, otherwise we just get a pointer
            all_frames.append(img.copy())
            frame_index += 1
        except EOFError:
            # Reached the end of this GIF
            break

# ---------- 3. Save the combined GIF ----------
if not all_frames:
    print("Error: No frames were extracted. Cannot create combined GIF.")
else:
    print(f"\nTotal frames extracted: {len(all_frames)}")
    # Calculate duration in milliseconds from FPS
    duration_ms = 1000 / FPS

    # Take the first frame and tell it to save all others after it
    all_frames[0].save(
        OUTPUT_PATH,
        save_all=True,
        append_images=all_frames[1:],  # List of all other frames
        duration=duration_ms,
        loop=0  # 0 means loop indefinitely
    )
    print(f"\nSuccessfully stitched all GIFs into:\n{OUTPUT_PATH}")