#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 20:35:11 2025

@author: jessica
"""

import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#------------ HEATMAP --------------------------
# ----------------- Setup -----------------
output_dir = "/Users/jessica/Downloads/NMTT/CHEM324_DATA/Final/bacteria_MSD_plots/heatmaps"
os.makedirs(output_dir, exist_ok=True)

data_folders = {
    "1x_glucose": "/Users/jessica/Downloads/NMTT/CHEM324_DATA/Final/bacteria_MSD_data/1x_glucose",
    "1x_glucose2": "/Users/jessica/Downloads/NMTT/CHEM324_DATA/Final/bacteria_MSD_data/1x_glucose2",
    "0.1x_glucose": "/Users/jessica/Downloads/NMTT/CHEM324_DATA/Final/bacteria_MSD_data/0.1x_glucose"
}

# Define heat map binning
grid_size = 0.2  # μm per bin
x_range = (-5, 5)  # μm
y_range = (-5, 5)  # μm
x_bins = int((x_range[1] - x_range[0]) / grid_size)
y_bins = int((y_range[1] - y_range[0]) / grid_size)

# ----------------- Heat Map Plotting -----------------
def plot_heatmap(data, time_data, title, filename):
    heatmap, xedges, yedges = np.histogram2d(
        data["x (μm)"], data["y (μm)"],
        bins=[x_bins, y_bins],
        range=[x_range, y_range]
    )
    heatmap = heatmap.T  # Flip to match plot orientation
    norm = plt.Normalize(vmin=np.min(time_data), vmax=np.max(time_data))
    plt.figure(figsize=(6, 5))
    sns.heatmap(heatmap, cmap="magma", cbar=True)
    plt.title(title)
    plt.xlabel("x (μm)")
    plt.ylabel("y (μm)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

# ----------------- Main Loop -----------------
for condition, folder in data_folders.items():
    combined_positions = pd.DataFrame(columns=["x (μm)", "y (μm)"])
    combined_time_data = []

    for file in glob.glob(os.path.join(folder, "*.xlsx")):
        try:
            df = pd.read_excel(file, engine="openpyxl")
            if "Δx (μm)" not in df.columns or "Δy (μm)" not in df.columns:
                print(f"⚠️ Skipping {file}: Δx or Δy not found.")
                continue

            # Reconstruct trajectory from deltas
            x = np.cumsum(df["Δx (μm)"].values)
            y = np.cumsum(df["Δy (μm)"].values)
            time_data = df["Time (s)"].values

            pos_df = pd.DataFrame({"x (μm)": x, "y (μm)": y})

            # Plot individual heat map
            base_name = os.path.splitext(os.path.basename(file))[0]
            plot_heatmap(pos_df, time_data, f"Position Heatmap - {base_name}", f"{condition}_{base_name}_heatmap.png")

            # Append to combined
            combined_positions = pd.concat([combined_positions, pos_df])
            combined_time_data.append(time_data)

        except Exception as e:
            print(f"❌ Error in {file}: {e}")

    # Plot combined heat map
    if not combined_positions.empty and combined_time_data:
        try:
            all_time_data = np.concatenate(combined_time_data)
            plot_heatmap(combined_positions, all_time_data, f"Combined Heatmap - {condition}", f"{condition}_combined_heatmap.png")
        except Exception as e:
            print(f"❌ Error generating combined heatmap for {condition}: {e}")

print("✅ Heat map generation complete.")


####################################################################
# ------------- combined MSD for each bacteria condition ----------
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------- Main Loop -----------------
msd_data_all_bacteria = []  # To store MSDs from all bacteria

# Loop over all data files in the folder (assuming CSVs or XLSXs)
for file in glob.glob(os.path.join("/Users/jessica/Downloads/NMTT/CHEM324_DATA/Final/bacteria_MSD_data/1x_glucose2", "*.xlsx")):  
    try:
        # Read data
        df = pd.read_excel(file, engine="openpyxl")
        
        # Normalize column names (strip extra spaces and make lowercase)
        df.columns = [col.strip().lower() for col in df.columns]
        
        # Check if required columns are in the DataFrame (case-insensitive and stripped spaces)
        if "msd (μm²)" not in df.columns or "time (s)" not in df.columns:
            print(f"⚠️ Skipping {file}: Required columns not found.")
            continue
        
        # Get the MSD and Time data
        msd = df["msd (μm²)"].values
        time = df["time (s)"].values
        
        # Append the MSD data for this bacteria to the list
        msd_data_all_bacteria.append(msd)
        
    except Exception as e:
        print(f"❌ Error reading {file}: {e}")

# ----------------- Combine MSDs from all bacteria -----------------
if msd_data_all_bacteria:
    max_length = max(len(msd) for msd in msd_data_all_bacteria)
    
    # Pad shorter MSDs with NaN to make them the same length
    msd_data_padded = [
        np.pad(msd, (0, max_length - len(msd)), constant_values=np.nan) for msd in msd_data_all_bacteria
    ]
    
    # Convert to numpy array for easy processing
    msd_data_combined = np.array(msd_data_padded)

    # Calculate the mean MSD over all bacteria
    mean_msd = np.nanmean(msd_data_combined, axis=0)
    
    # Create time points for plotting (assumes all time vectors are the same)
    time_points = time[:len(mean_msd)]  # Use the time vector from any file, assuming it's consistent
    
    
    # ----------------- Trim to 375 frames -----------------
    trim_length = 375
    if len(mean_msd) > trim_length:
        mean_msd = mean_msd[:trim_length]
        # Trim the time points to match the length of the MSD
        time_points = time[:trim_length]
    else:
        # If data has less than 375 frames, extend it with NaN (or another padding value)
        mean_msd = np.pad(mean_msd, (0, trim_length - len(mean_msd)), constant_values=np.nan)
        time_points = np.pad(time[:len(mean_msd)], (0, trim_length - len(time_points)), constant_values=np.nan)


    # ----------------- Plot Combined MSD -----------------
    plt.figure(figsize=(8, 6))
    plt.plot(time_points, mean_msd, label="Mean MSD (μm²)", color="b")
    plt.xlabel("Time (s)")
    plt.ylabel("Mean Squared Displacement (µm²)")
    plt.title("Combined MSD Plot - 1x Bacteria 4 hr off heat")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("combined_msd_plot.png")
    plt.show()

else:
    print("⚠️ No valid MSD data found.")

print("✅ Combined MSD plot generation complete.")


# ============ overlaying =====================

# Conditions and their directories
conditions = {
    "1x glucose": "/Users/jessica/Downloads/NMTT/CHEM324_DATA/Final/bacteria_MSD_data/1x_glucose",
    "0.1x glucose": "/Users/jessica/Downloads/NMTT/CHEM324_DATA/Final/bacteria_MSD_data/0.1x_glucose",
    "1x glucose2": "/Users/jessica/Downloads/NMTT/CHEM324_DATA/Final/bacteria_MSD_data/1x_glucose2"
}

# ----------------- Main Loop -----------------
msd_data_all_conditions = {
    "1x glucose": [],
    "0.1x glucose": [],
    "1x glucose2": []
}

# Loop over all conditions and their data files
for condition, folder in conditions.items():
    for file in glob.glob(os.path.join(folder, "*.xlsx")):  
        try:
            # Read data
            df = pd.read_excel(file, engine="openpyxl")
            
            # Normalize column names (strip extra spaces and make lowercase)
            df.columns = [col.strip().lower() for col in df.columns]
            
            # Check if required columns are in the DataFrame (case-insensitive and stripped spaces)
            if "msd (μm²)" not in df.columns or "time (s)" not in df.columns:
                print(f"⚠️ Skipping {file}: Required columns not found.")
                continue
            
            # Get the MSD and Time data
            msd = df["msd (μm²)"].values
            time = df["time (s)"].values
            
            # Append the MSD data for this condition
            msd_data_all_conditions[condition].append((msd, time))
        
        except Exception as e:
            print(f"❌ Error reading {file}: {e}")

# ----------------- Plot Combined MSD for All Conditions -----------------
plt.figure(figsize=(8, 6))

# Loop over each condition's MSD data
for condition, data_list in msd_data_all_conditions.items():
    msd_data_all_bacteria = [data[0] for data in data_list]
    
    # Find the max length of all MSD data for this condition
    max_length = max(len(msd) for msd in msd_data_all_bacteria)
    
    # Pad shorter MSDs with NaN to make them the same length
    msd_data_padded = [
        np.pad(msd, (0, max_length - len(msd)), constant_values=np.nan) for msd in msd_data_all_bacteria
    ]
    
    # Convert to numpy array for easy processing
    msd_data_combined = np.array(msd_data_padded)

    # Calculate the mean MSD over all bacteria for this condition
    mean_msd = np.nanmean(msd_data_combined, axis=0)
    
    # Trim both mean_msd and time_points to 375 frames
    trim_length = 375
    if len(mean_msd) > trim_length:
        mean_msd = mean_msd[:trim_length]
        time_points = data_list[0][1][:trim_length]  # Use the first time vector for the condition
    else:
        mean_msd = np.pad(mean_msd, (0, trim_length - len(mean_msd)), constant_values=np.nan)
        time_points = np.pad(data_list[0][1][:len(mean_msd)], (0, trim_length - len(time_points)), constant_values=np.nan)

    # Plot the mean MSD for this condition
    plt.plot(time_points, mean_msd, label=condition)

# ----------------- Customize and Show the Plot -----------------
plt.xlabel("Time (s)")
plt.ylabel("Mean Squared Displacement (µm²)")
plt.title("Overlay of Combined MSD for All Bacteria Conditions")
plt.grid(True)
plt.legend()  # Show the legend to label each condition
plt.tight_layout()
plt.savefig("overlay_msd_plot.png")  # Save the plot as a PNG file
plt.show()

print("✅ Overlay MSD plot generation complete.")
#=========== Laser intensity vs Force with R² =======
import matplotlib.pyplot as plt
import numpy as np

# Data
current_mA = np.array([65, 75, 85])  # Laser current in milliamps
force_pN = np.array([0.0400, 0.0607, 0.187])  # Force in picoNewtons

# Fit a linear trendline: Force = a * Current + b
coeffs = np.polyfit(current_mA, force_pN, 1)  # Linear fit
trendline = np.poly1d(coeffs)
fit_y = trendline(current_mA)

# Calculate R²
ss_res = np.sum((force_pN - fit_y) ** 2)
ss_tot = np.sum((force_pN - np.mean(force_pN)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

# Create the plot
plt.figure(figsize=(8, 6))
plt.scatter(current_mA, force_pN, color='b', label="Measured Data")
plt.plot(current_mA, fit_y, color='r', linestyle='--',
         label=f"Linear Fit: F = {coeffs[0]:.3e}·I + {coeffs[1]:.3e}\n$R^2$ = {r_squared:.3f}")

# Labeling
plt.xlabel("Current (mA)")
plt.ylabel("Force (pN)")
plt.title("Plot of Laser Intensity vs Force (by Hooke's Law)")
plt.legend()

# Grid and layout
plt.grid(False)
plt.tight_layout()

# Save and display
plt.savefig("laser_intensity_vs_force_with_fit_and_r2.png")
plt.show()

print("✅ Plot with linear trendline and R² value generated and saved.")



#=========== Combine MSD examples ================
import pandas as pd
import matplotlib.pyplot as plt

# Parameters
msd_threshold = 0.490       # μm²
slope_threshold = 3.434     # μm²/s

# File paths — replace with your actual file paths
file_paths = [
     "/Users/jessica/Downloads/NMTT/CHEM324_DATA/Final/bacteria_MSD_data/1x_glucose/85mA_20250512_tepache_1x_glucose_8_tracking_output.xlsx",
    "/Users/jessica/Downloads/NMTT/CHEM324_DATA/Final/bacteria_MSD_data/0.1x_glucose/85mA_20250512_tepache_0.1x_glucose_3_tracking_output.xlsx",
    "/Users/jessica/Downloads/NMTT/CHEM324_DATA/Brownian/bacteria/85mA_20250512_tepache_1x_glucose_8_tracking_output.xlsx"
]

# Colors and labels for plot
colors = ['orange', 'green', 'red']
labels = ['1x glucose', '0.1x glucose', 'Untrapped (1x glucose']

# Plotting
plt.figure(figsize=(10, 6))

for i, file_path in enumerate(file_paths):
    # Load data
    df = pd.read_excel(file_path)

    # Ensure correct column names
    time = df['Time (s)']
    msd = df['MSD (μm²)']

    # Compute slope
    slopes = msd.diff() / time.diff()

    # Determine escape index
    escape_index = ((msd >= msd_threshold) & (slopes >= slope_threshold)).idxmax()

    # Escape time and MSD
    escape_time = time.iloc[escape_index]
    escape_msd = msd.iloc[escape_index]

    # Plot MSD vs Time
    plt.plot(time, msd, label=f"{labels[i]} (Escape at {escape_time:.2f}s)", color=colors[i])
    plt.plot(escape_time, escape_msd, 'o', color=colors[i], markersize=10)

# Labels and formatting
plt.title("MSD vs Time with Escape Points")
plt.xlabel("Time (s)")
plt.ylabel("MSD (μm²)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

