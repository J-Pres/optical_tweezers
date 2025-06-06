#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 13 18:26:29 2025

@author: jessica
"""
import cv2
import os

# --- Set path to your .avi video
video_path = "/Users/jessica/Downloads/NMTT/CHEM324_DATA/bug/Tepache_1x_glucose/85mA_20250512_tepache_1x_glucose_8.avi"

# --- Set the output directory where you want to save the frame images
output_dir = "/Users/jessica/Downloads/NMTT/CHEM324_DATA/bug/Tepache_1x_glucose"

# --- Create the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# --- Set the output image name and frame number
output_image = os.path.join(output_dir, "bug_frame_10.png")  # Combine directory and image filename
frame_number = 10  # Change to any frame you want

# --- Open the video
cap = cv2.VideoCapture(video_path)

# --- Go to the desired frame
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
ret, frame = cap.read()

if ret:
    cv2.imwrite(output_image, frame)
    print(f"Frame {frame_number} saved as {output_image}")
else:
    print("Failed to read the frame.")

cap.release()







#----- Task 2 of bacteria_analysis: identify escape threshold from bead data -----
import os
import numpy as np
import pandas as pd

folder_path2 = "/Users/jessica/Downloads/NMTT/CHEM324_DATA/bug/Tepache_1x_glucose/TrackingOutput_85mA_3um/"


# --- Column names in your Excel files ---
msd_col = 'MSD (μm²)'
time_col = 'Time (s)'


bead_msd_values = []
all_bead_slopes = []

# --- Loop through files and collect MSD and slope data ---
for fname in os.listdir(folder_path2):
    if fname.endswith('.xlsx'):
        file_path = os.path.join(folder_path2, fname)
        df = pd.read_excel(file_path)

        # Drop NaNs and extract MSD and time
        df = df[[msd_col, time_col]].dropna()
        msd = df[msd_col].values
        time = df[time_col].values

        # Collect MSD values
        bead_msd_values.extend(msd)

        # Compute slope: ΔMSD/Δt
        slope = np.gradient(msd, time)
        all_bead_slopes.extend(slope)

# --- Convert to arrays ---
bead_msd_values = np.array(bead_msd_values)
all_bead_slopes = np.array(all_bead_slopes)

# --- Calculate thresholds ---
msd_threshold_95 = np.percentile(bead_msd_values, 99.99)
slope_threshold_99 = np.percentile(all_bead_slopes, 99.99)

# --- Output results ---
print("\n✅ Threshold Summary from Bead Data:")
print(f"MSD threshold (95th percentile):      {msd_threshold_95:.3f} μm²")
print(f"Slope threshold (99th percentile):    {slope_threshold_99:.3f} μm²/s")

