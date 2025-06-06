#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 12:18:44 2025

@author: jessica
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------- Task 1: Create plots of bacteria MDS vs time ---------

# ----------- Constants -----------
frame_interval = 0.06666  # seconds per frame
folder_path = "/Users/jessica/Downloads/NMTT/CHEM324_DATA/Brownian/bacteria/"
plot_folder = os.path.join(folder_path, 'brownian_msd_plots')
os.makedirs(plot_folder, exist_ok=True)

# ----------- Plot Each MSD File -----------
for file in glob.glob(os.path.join(folder_path, "*.xlsx")):
    try:
        df = pd.read_excel(file, engine='openpyxl')
        if "MSD (μm²)" not in df.columns:
            print(f"⚠️ Skipping {file}: 'MSD (μm²)' column not found.")
            continue

        msd = df["MSD (μm²)"].values
        times = np.arange(len(msd)) * frame_interval

        plt.figure()
        plt.plot(times, msd, label="MSD", color='blue')
        plt.xlabel("Time (s)")
        plt.ylabel("MSD (μm²)")
        plt.title(f"MSD Curve: {os.path.basename(file)}")
        plt.grid(True)
        plt.tight_layout()

        output_path = os.path.join(plot_folder, f"MSD_{os.path.splitext(os.path.basename(file))[0]}.png")
        plt.savefig(output_path)
        plt.close()
        print(f"✅ Saved plot: {output_path}")

    except Exception as e:
        print(f"❌ Error reading {file}: {e}")


    
# -------- Task 2: Produce Escape Data -------
# --- CONFIGURABLE SETTINGS ---
output_plot_folder = os.path.join(folder_path, 'brownian_marked_plots')
excel_summary_path = os.path.join(folder_path, 'brownian_escape_summary.xlsx')

k = 4.55113862919928E-07  # Trap stiffness in N/m from 85μm bead data

# Thresholds for escape detection based on bead behaviour (see Extra.py)
msd_threshold = 0.490       # μm²
slope_threshold = 3.434     # μm²/s

# Create output folder for plots
os.makedirs(output_plot_folder, exist_ok=True)

# Store results
escape_data = []

# --- Process each Excel file in the folder ---
for filename in os.listdir(folder_path):
    if filename.endswith('.xlsx'):
        file_path = os.path.join(folder_path, filename)
        try:
            df = pd.read_excel(file_path)

            time = df['Time (s)'].values
            msd = df['MSD (μm²)'].values
            x_disp = df['Δx (μm)'].values  # in micrometers
            y_disp = df['Δy (μm)'].values  # in micrometers

            # Compute MSD slope
            msd_derivative = np.gradient(msd, time)

            # Detect escape frame
            escape_index = None
            for i in range(1, len(msd)):
                if msd[i] > msd_threshold and msd_derivative[i] > slope_threshold:
                    escape_index = i
                    break

            escape_force = None
            net_displacement_um = None

            # Plot
            plt.figure(figsize=(8, 5))
            plt.plot(time, msd, label='MSD', color='blue')

            if escape_index is not None:
                escape_time = time[escape_index]
                plt.axvline(escape_time, color='red', linestyle='--', label=f'Escape @ {escape_time:.2f}s')
                plt.scatter(escape_time, msd[escape_index], color='red')
                escape_status = 'ESCAPED'

                # --- Calculate escape displacement and force ---
                x = x_disp[escape_index]
                y = y_disp[escape_index]
                net_displacement_um = np.sqrt(x**2 + y**2)         # in micrometers
                net_displacement_m = net_displacement_um * 1e-6    # convert to meters
                escape_force = (k * net_displacement_m) * 1e12           # in picoNewtons

            else:
                escape_time = None
                escape_status = 'NO ESCAPE'

            plt.xlabel('Time (s)')
            plt.ylabel('MSD (μm²)')
            plt.title(f'MSD: {filename}')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            # Save plot
            plot_file = os.path.join(output_plot_folder, f"{os.path.splitext(filename)[0]}_msd_plot.png")
            plt.savefig(plot_file)
            plt.close()

            # Save escape info
            escape_data.append({
                'Filename': filename,
                'Escape Detected': escape_status,
                'Escape Frame': escape_index if escape_index is not None else 'N/A',
                'Escape Time (s)': escape_time if escape_time is not None else 'N/A',
                'Escape Displacement (μm)': net_displacement_um if net_displacement_um is not None else 'N/A',
                'Escape Force (pN)': escape_force if escape_force is not None else 'N/A'
            })

            print(f"Processed {filename}: {escape_status}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

# --- Save summary CSV ---
summary_df = pd.DataFrame(escape_data)
excel_summary_path = os.path.join(folder_path, 'escape_summary.xlsx')

try:
    summary_df = pd.DataFrame(escape_data)
    summary_df.to_excel(excel_summary_path, index=False, engine='openpyxl')
    print(f"\nExcel summary saved to: {excel_summary_path}")
except Exception as e:
    print(f"Failed to save Excel summary: {e}")
    
    
    
    
#statistical Analysis
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, kruskal, mannwhitneyu, levene

# --- Load the data from Excel files ---
folder_path = "/Users/jessica/Downloads/NMTT/CHEM324_DATA/Final/Analysis/"
files = {
    "0.1x": "0.1x_escape_summary.xlsx",
    "1x": "1x_escape_summary.xlsx",
    "1x_later": "1x_later_escape_summary.xlsx"
}

data = {}
for label, file in files.items():
    df = pd.read_excel(os.path.join(folder_path, file))
    df["Condition"] = label
    data[label] = df

# --- Combine for easier plotting ---
combined_df = pd.concat(data.values(), ignore_index=True)

# --- Plot: Boxplot of Escape Forces ---
plt.figure(figsize=(8, 6))
sns.boxplot(x="Condition", y="Escape Force (pN)", data=combined_df, palette="Set2")
plt.title("Escape Force by Glucose Condition")
plt.tight_layout()
plt.savefig("escape_force_boxplot.png")
plt.show()

# --- Shapiro-Wilk Test (Normality) ---
print("\n🔍 Shapiro-Wilk Normality Test:")
for label, df in data.items():
    stat, p = shapiro(df["Escape Force (pN)"])
    print(f"{label}: p = {p:.4f} {'(normal)' if p > 0.05 else '❌ not normal'}")

# --- Levene’s Test (Homogeneity of variances) ---
levene_stat, levene_p = levene(
    data["0.1x"]["Escape Force (pN)"],
    data["1x"]["Escape Force (pN)"],
    data["1x_later"]["Escape Force (pN)"]
)
print(f"\n📏 Levene’s test for equal variances: p = {levene_p:.4f}")

# --- Kruskal-Wallis Test (Non-parametric ANOVA) ---
kruskal_stat, kruskal_p = kruskal(
    data["0.1x"]["Escape Force (pN)"],
    data["1x"]["Escape Force (pN)"],
    data["1x_later"]["Escape Force (pN)"]
)
print(f"📊 Kruskal-Wallis test: p = {kruskal_p:.4f}")

# --- Pairwise Mann-Whitney U Tests ---
print("\n🧪 Pairwise Mann–Whitney U Tests:")
groups = list(data.keys())
for i in range(len(groups)):
    for j in range(i+1, len(groups)):
        g1, g2 = groups[i], groups[j]
        stat, p = mannwhitneyu(
            data[g1]["Escape Force (pN)"], 
            data[g2]["Escape Force (pN)"], 
            alternative='two-sided'
        )
        print(f"{g1} vs {g2}: p = {p:.4f}")
        
        
import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load the Excel files ---
folder_path = "/Users/jessica/Downloads/NMTT/CHEM324_DATA/Final/Analysis/"
file_01x = pd.read_excel(os.path.join(folder_path, "0.1x_escape_summary.xlsx"))
file_1x = pd.read_excel(os.path.join(folder_path, "1x_escape_summary.xlsx"))

# --- Extract and clean the "Escape Force" values ---
escape_01x = file_01x["Escape Force (pN)"].dropna().values
escape_1x = file_1x["Escape Force (pN)"].dropna().values

# --- Parametric ANOVA (F-test for two groups) ---
anova_stat, anova_p = stats.f_oneway(escape_01x, escape_1x)
print(f"🔬 ANOVA p-value (0.1x vs 1x): {anova_p:.4f}")

# --- Permutation Test: difference in means ---
observed_diff = np.mean(escape_1x) - np.mean(escape_01x)
combined = np.concatenate([escape_1x, escape_01x])
n_permutations = 10000
perm_diffs = []

for _ in range(n_permutations):
    np.random.shuffle(combined)
    perm_1x = combined[:len(escape_1x)]
    perm_01x = combined[len(escape_1x):]
    perm_diffs.append(np.mean(perm_1x) - np.mean(perm_01x))

perm_diffs = np.array(perm_diffs)
p_perm = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
print(f"🧪 Permutation test p-value: {p_perm:.4f}")
print(f"Observed mean difference: {observed_diff:.4f} pN")

# --- Plot the permutation test distribution ---
plt.figure(figsize=(8, 5))
sns.histplot(perm_diffs, kde=True, bins=50, color="skyblue")
plt.axvline(observed_diff, color='red', linestyle='--', label=f'Observed Δ = {observed_diff:.4f}')
plt.axvline(-observed_diff, color='red', linestyle='--')
plt.title("Permutation Test: Difference in Means (1x vs 0.1x)")
plt.xlabel("Mean Difference (pN)")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig("permutation_test_force.png")
plt.show()
