#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 16 18:59:04 2025

@author: jessica
"""

import pandas as pd
import numpy as np
import glob
import os
import re
import matplotlib.pyplot as plt
from scipy.stats import linregress

# -------------------- Constants --------------------
kB = 1.38064852e-23  # Boltzmann constant (J/K)
T = 298              # Room temperature (K)
r = 1.5e-6           # Bead radius (3 μm diameter)
scale = 1e-6         # μm to m
frame_interval = 0.06666  # s per frame
n_eff = 0.003201205     # Constant effective viscosity (Pa·s)

# -------------------- Setup Output Folder --------------------
os.makedirs("plots", exist_ok=True)

# -------------------- Helper Functions --------------------

def extract_laser_current(file_name):
    """Extract laser current (e.g., 65 from '3um_65mA_10000_250428_0_0_tracking_output.xlsx')"""
    match = re.search(r'(\d+)mA', file_name)
    return int(match.group(1)) if match else None

def fit_msd_slope(msd, frame_interval, file_id):
    """Fit linear region of MSD curve and plot it."""
    n_points = min(len(msd), 300)  # Adjust this if needed
    times = np.arange(n_points) * frame_interval
    msd_fit = msd[:n_points]

    # Linear regression
    slope, intercept, _, _, _ = linregress(times, msd_fit)
    m = slope  # μm²/s

    # Plot
    plt.figure()
    plt.plot(np.arange(len(msd)) * frame_interval, msd, label="MSD")
    plt.plot(times, slope * times + intercept, label=f"Linear fit (m={m:.3f} μm²/s)", linestyle='--')
    plt.xlabel("Time (s)")
    plt.ylabel("MSD (μm²)")
    plt.title(f"MSD vs Time: {file_id}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/MSD_fit_{file_id}.png")
    plt.close()

    return m

def calculate_stokes_force(eta, velocity_m_s, r):
    return 6 * np.pi * eta * r * velocity_m_s  # in Newtons

def calculate_trap_stiffness(msd_plateau_um2):
    msd_m2 = msd_plateau_um2 * (scale ** 2)
    k = (kB * T) / msd_m2  # in N/m
    return k

# -------------------- Main File Processor --------------------

def process_file(file_path):
    df = pd.read_excel(file_path, engine='openpyxl')
    file_name = os.path.basename(file_path)
    laser_current = extract_laser_current(file_name)

    # Extract MSD and velocity
    msd = df["MSD (μm²)"].values
    velocity_um_s = df["Instantaneous Velocity (μm/s)"].values

    # Fit MSD slope (for plotting only)
    msd_slope = fit_msd_slope(msd, frame_interval, os.path.splitext(file_name)[0])

    # Max velocity
    v_max_um_s = np.max(velocity_um_s)
    v_max_m_s = v_max_um_s * scale

    # Max holding force using constant η_eff
    F_max_N = calculate_stokes_force(n_eff, v_max_m_s, r)

    # MSD plateau and trap stiffness
    msd_plateau_um2 = np.mean(msd[-100:])  # use last 100 frames
    k_trap = calculate_trap_stiffness(msd_plateau_um2)

    # Displacement
    dx = df["Δx (μm)"].values
    dy = df["Δy (μm)"].values
    displacement_um = np.sqrt(dx**2 + dy**2)
    max_disp_um = np.max(displacement_um)
    max_disp_m = max_disp_um * scale
    F_Hooke = (max_disp_m * k_trap) # in Newtons


    return {
        "file": file_name,
        "Laser Current (mA)": laser_current,
        "MSD_slope (μm²/s)": msd_slope,
        "η_eff (Pa·s)": n_eff,
        "v_max (μm/s)": v_max_um_s,
        "F_max_Stokes (pN)": F_max_N * 1e12,
        "MSD_plateau (μm²)": msd_plateau_um2,
        "k_trap (N/m)": k_trap,
        "Max displacement (μm)": max_disp_um,
        "Max displacement (m)": max_disp_m,
        "F_Hooke (pN)": F_Hooke * 1e12,
    
    }

# -------------------- Batch Process All Files --------------------

results = []
for file in glob.glob("/Users/jessica/Downloads/NMTT/CHEM324_DATA/holding_force/*.xlsx"):
    result = process_file(file)
    results.append(result)

results_df = pd.DataFrame(results)
results_df.to_excel("ttrap_results_combined.xlsx", index=False)

# -------------------- Group by Laser Current --------------------

grouped = results_df.groupby("Laser Current (mA)").mean(numeric_only=True)
grouped.to_excel("ttrap_results_grouped_by_current.xlsx")
print("✅ Processing complete. Individual and grouped results saved.")


# -------------------- Create Histograms of Displacement --------------------

import seaborn as sns
from collections import defaultdict

# Collect all displacements grouped by laser current
displacements_by_current = defaultdict(list)

for file in glob.glob("/Users/jessica/Downloads/NMTT/CHEM324_DATA/holding_force/*.xlsx"):
    df = pd.read_excel(file, engine="openpyxl")
    file_name = os.path.basename(file)
    current = extract_laser_current(file_name)

    if "Δx (μm)" in df.columns and "Δy (μm)" in df.columns:
        dx = df["Δx (μm)"].values
        dy = df["Δy (μm)"].values
        displacement = np.sqrt(dx**2 + dy**2)
        displacements_by_current[current].extend(displacement)
    else:
        print(f"⚠️ Skipping file {file_name}: Δx or Δy columns missing.")

# Create one histogram per current
for current, displacements in displacements_by_current.items():
    plt.figure(figsize=(6, 4))
    sns.histplot(displacements, bins=20, kde=True, color="steelblue")
    plt.xlabel("Displacement (μm)")
    plt.ylabel("Frequency")
    plt.title(f"Displacement Histogram - {current} mA")
    plt.tight_layout()
    plt.savefig(f"plots/Displacement_Histogram_{current}mA.png")
    plt.close()

print("📊 Displacement histograms saved in 'plots' folder.")

# -------------------- Create GREEN Histograms of Displacement --------------------

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import glob
from collections import defaultdict
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap, Normalize
import os

# Define and create the output directory
output_dir = "/Users/jessica/Downloads/NMTT/CHEM324_DATA/Final/bead_standards_plots"
os.makedirs(output_dir, exist_ok=True)

# Function to extract laser current from filename
def extract_laser_current(file_name):
    import re
    match = re.search(r'(\d+)mA', file_name)
    return int(match.group(1)) if match else None

# Initialize displacement storage
displacements_by_current = defaultdict(list)

# Load data
for file in glob.glob("/Users/jessica/Downloads/NMTT/CHEM324_DATA/holding_force/*.xlsx"):
    df = pd.read_excel(file, engine="openpyxl")
    file_name = os.path.basename(file)
    current = extract_laser_current(file_name)

    if current is None:
        print(f"⚠️ Could not extract current from {file_name}")
        continue

    if "Δx (μm)" in df.columns and "Δy (μm)" in df.columns:
        dx = df["Δx (μm)"].values
        dy = df["Δy (μm)"].values
        displacement = np.sqrt(dx**2 + dy**2)
        displacements_by_current[current].extend(displacement)
    else:
        print(f"⚠️ Skipping file {file_name}: Δx or Δy columns missing.")

# Custom green colormap
green_colors = [  
    "#f1f8f5",  # very pale mint (almost white sage)
    "#b8d8d1",  # cool eucalyptus green
    "#98c9b8",  # minty sage
    "#80b2a3",  # fresh soft teal-green
    "#679f8a",  # bright cool mossy green
    "#4b7f65"   # deep sage-blue
    ]   

green_cmap = LinearSegmentedColormap.from_list("green_cmap", green_colors)

# Loop over each current and generate histogram
for current, displacements in displacements_by_current.items():
    plt.figure(figsize=(6, 4))

    # Compute histogram
    counts, bins = np.histogram(displacements, bins=20)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    # Normalize and assign colors
    norm = Normalize(vmin=counts.min(), vmax=counts.max())
    colors = green_cmap(norm(counts))

    # Plot each bar manually
    for i in range(len(counts)):
        plt.bar(bin_centers[i], counts[i], width=bins[1] - bins[0], color=colors[i], align='center')

    # Labels and title
    plt.xlabel("Displacement (μm)")
    plt.ylabel("Frequency")
    plt.title(f"Displacement Histogram - {current} mA")
    plt.tight_layout()

    # Save plot
    filename = f"Displacement_Histogram_{current}mA.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

print(f"📊 Displacement histograms saved in: {output_dir}")


#hist analysis
# -------------------- Displacement Statistics Summary --------------------

from scipy.stats import skew, kurtosis

print("\n📈 Summary Statistics for Bead Displacement by Laser Current:")

summary_data = []
for current, displacements in sorted(displacements_by_current.items()):
    displacements = np.array(displacements)
    mean_val = np.mean(displacements)
    std_val = np.std(displacements)
    cv_val = std_val / mean_val if mean_val != 0 else float('nan')
    skew_val = skew(displacements)
    kurt_val = kurtosis(displacements)  # excess kurtosis (normal = 0)

    print(f"\n🔹 {current} mA:")
    print(f"   Mean (μ)       = {mean_val:.4f} µm")
    print(f"   Std Dev (σ)    = {std_val:.4f} µm")
    print(f"   CV             = {cv_val:.4f}")
    print(f"   Skewness       = {skew_val:.4f}")
    print(f"   Kurtosis       = {kurt_val:.4f}")

    summary_data.append([current, mean_val, std_val, cv_val, skew_val, kurt_val])

# Optional: Save stats to CSV
import pandas as pd
stats_df = pd.DataFrame(summary_data, columns=[
    "Current (mA)", "Mean (µm)", "Std Dev (µm)", "CV", "Skewness", "Kurtosis"
])
stats_df.to_csv("plots/Displacement_Stats_Summary.csv", index=False)
print("\n📄 Summary statistics saved to 'Displacement_Stats_Summary.csv'")




import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import re

# Ensure output folder exists
os.makedirs("plots", exist_ok=True)

# Extract current from filename
def extract_laser_current(filename):
    match = re.search(r"_(\d{2,3})mA", filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Laser current not found in filename: {filename}")

# Collect Δx and Δy for each laser current
xy_by_current = defaultdict(lambda: {"dx": [], "dy": []})

# Process data files
for file in glob.glob("/Users/jessica/Downloads/NMTT/CHEM324_DATA/holding_force/*.xlsx"):
    df = pd.read_excel(file, engine="openpyxl")
    file_name = os.path.basename(file)
    try:
        current = extract_laser_current(file_name)
    except ValueError as e:
        print(e)
        continue

    if "Δx (μm)" in df.columns and "Δy (μm)" in df.columns:
        dx = df["Δx (μm)"].values
        dy = df["Δy (μm)"].values
        xy_by_current[current]["dx"].extend(dx)
        xy_by_current[current]["dy"].extend(dy)
    else:
        print(f"⚠️ Skipping file {file_name}: Δx or Δy columns missing.")

# Create 2D scatter plots
for current, coords in xy_by_current.items():
    dx = coords["dx"]
    dy = coords["dy"]
    
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=dx, y=dy, alpha=0.5, s=10, color="darkorange", edgecolor=None)
    plt.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    plt.axvline(0, color='grey', linestyle='--', linewidth=0.8)
    plt.xlabel("Δx (μm)")
    plt.ylabel("Δy (μm)")
    plt.title(f"2D Displacement Plot (Δx vs Δy) - {current} mA")
    plt.axis("equal")  # Ensures circular traps look circular
    plt.tight_layout()
    plt.savefig(f"plots/Scatter_Dx_Dy_{current}mA.png")
    plt.close()

print("📍 2D Δx vs Δy scatter plots saved in 'plots' folder.")


