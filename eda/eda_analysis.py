"""
Comprehensive Exploratory Data Analysis (EDA) for EEG Motor Imagery Data
Industry-Standard Analysis Pipeline

This script generates 15+ visualization categories covering:
1. Data Overview & Class Distribution
2. Signal Characteristics
3. Frequency Domain (PSD)
4. Band Power Analysis
5. Channel-wise Analysis
6. Topographic Maps (Enhanced)
7. Motor Cortex Focus Topography
8. Temporal Dynamics (ERP)
9. Subject-wise Analysis
10. Correlation & Separability (Enhanced)
11. Class-wise Band Power Heatmap
12. Time-Frequency by Channel
13. Subject Performance Variability
14. Outlier Detection
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from scipy import signal
from scipy.interpolate import griddata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings

warnings.filterwarnings("ignore")

# Configuration
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 10

# Paths
DATA_DIR = "/home/heyatoy/Chimera/EEGEncoder/src/data/BCICIV_2a_gdf"
OUTPUT_DIR = "/home/heyatoy/Chimera/EEGEncoder/eda/figures"
SUBJECTS = ["A01", "A02", "A03", "A04", "A05", "A06", "A07", "A08", "A09"]
CLASS_NAMES = ["Left Hand", "Right Hand", "Feet", "Tongue"]
COLORS = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]

# Full 25-channel 10-20 system positions (x, y) - normalized to unit circle
CHANNEL_POSITIONS = {
    "Fp1": (-0.294, 0.897),
    "Fp2": (0.294, 0.897),
    "F7": (-0.679, 0.561),
    "F3": (-0.334, 0.466),
    "Fz": (0.0, 0.421),
    "F4": (0.334, 0.466),
    "F8": (0.679, 0.561),
    "T7": (-0.892, 0.092),
    "C3": (-0.392, -0.025),
    "Cz": (0.0, -0.025),
    "C4": (0.392, -0.025),
    "T8": (0.892, 0.092),
    "P7": (-0.679, -0.438),
    "P3": (-0.334, -0.523),
    "Pz": (0.0, -0.572),
    "P4": (0.334, -0.523),
    "P8": (0.679, -0.438),
    "O1": (-0.294, -0.897),
    "Oz": (0.0, -0.947),
    "O2": (0.294, -0.897),
    "F1": (-0.167, 0.625),
    "F2": (0.167, 0.625),
    "P1": (-0.167, -0.625),
    "P2": (0.167, -0.625),
    "POz": (0.0, -0.760),
}

CHANNEL_NAMES_25 = [
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "T7",
    "C3",
    "Cz",
    "C4",
    "T8",
    "P7",
    "P3",
    "Pz",
    "P4",
    "P8",
    "O1",
    "Oz",
    "O2",
    "F1",
    "F2",
    "P1",
    "P2",
    "POz",
]

CHANNEL_NAMES_22 = CHANNEL_NAMES_25[:22]

print("=" * 60)
print("COMPREHENSIVE EEG MOTOR IMAGERY EDA")
print("=" * 60)

# Import data
import sys

sys.path.insert(0, "/home/heyatoy/Chimera/EEGEncoder")
from src.preprocessing.motor_imagery_pipeline import MotorImageryPreprocessor

# Load all data
print("\n[1/10] Loading data...")
preprocessor = MotorImageryPreprocessor(data_dir=DATA_DIR, filter_alpha_beta=False)

all_data, all_labels = {}, {}
for subj in SUBJECTS:
    X, y, _ = preprocessor.load_and_preprocess(subj)
    all_data[subj] = X
    all_labels[subj] = y

X_all = np.concatenate([all_data[s] for s in SUBJECTS], axis=0)
y_all = np.concatenate([all_labels[s] for s in SUBJECTS], axis=0)
print(
    f"  Loaded: {X_all.shape[0]} trials, {X_all.shape[1]} channels, {X_all.shape[2]} timepoints"
)

# ============================================================================
# 1. CLASS DISTRIBUTION
# ============================================================================
print("\n[2/10] Class distribution analysis...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Overall distribution
class_counts = np.bincount(y_all)
axes[0].bar(CLASS_NAMES, class_counts, color=COLORS)
axes[0].set_title("Overall Class Distribution")
axes[0].set_ylabel("Number of Trials")
for i, v in enumerate(class_counts):
    axes[0].text(i, v + 10, str(v), ha="center")

# Per-subject
x = np.arange(len(SUBJECTS))
w = 0.2
for i, (name, color) in enumerate(zip(CLASS_NAMES, COLORS)):
    counts = [np.sum(all_labels[s] == i) for s in SUBJECTS]
    axes[1].bar(x + i * w, counts, w, label=name, color=color)
axes[1].set_title("Class Distribution by Subject")
axes[1].set_xticks(x + 1.5 * w)
axes[1].set_xticklabels(SUBJECTS)
axes[1].legend(fontsize=8)

# Pie chart
balance_ratio = min(class_counts) / max(class_counts)
axes[2].pie(
    class_counts, labels=CLASS_NAMES, autopct="%1.1f%%", colors=COLORS, startangle=90
)
axes[2].set_title(f"Class Proportions\nBalance: {balance_ratio:.2f}")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_class_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: 01_class_distribution.png")

# ============================================================================
# 2. SIGNAL CHARACTERISTICS
# ============================================================================
print("\n[3/10] Signal characteristics analysis...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Amplitude histogram
flat = X_all.flatten()
axes[0, 0].hist(flat, bins=100, color="steelblue", edgecolor="black", alpha=0.7)
axes[0, 0].axvline(
    np.mean(flat), color="red", linestyle="--", label=f"Mean: {np.mean(flat):.3f}"
)
axes[0, 0].axvline(
    np.median(flat),
    color="orange",
    linestyle="--",
    label=f"Median: {np.median(flat):.3f}",
)
axes[0, 0].set_title("Amplitude Distribution")
axes[0, 0].set_xlabel("Amplitude (z-scored)")
axes[0, 0].legend()

# Channel means
ch_means = np.mean(X_all, axis=(0, 2))
axes[0, 1].bar(range(len(ch_means)), ch_means, color="teal")
axes[0, 1].set_title("Mean Amplitude by Channel")
axes[0, 1].set_xlabel("Channel Index")

# Channel stds
ch_stds = np.std(X_all, axis=(0, 2))
axes[0, 2].bar(range(len(ch_stds)), ch_stds, color="coral")
axes[0, 2].set_title("Std Dev by Channel")
axes[0, 2].set_xlabel("Channel Index")

# Sample trials
time = np.arange(X_all.shape[2]) / 250
for idx, (name, color) in enumerate(zip(CLASS_NAMES, COLORS)):
    trial_idx = np.where(y_all == idx)[0][0]
    axes[1, 0].plot(time, X_all[trial_idx, 0, :], label=name, color=color, alpha=0.8)
axes[1, 0].set_title("Sample Trials by Class (Ch 0)")
axes[1, 0].set_xlabel("Time (s)")
axes[1, 0].legend()

# Variance over time
var_time = np.var(X_all, axis=(0, 1))
axes[1, 1].plot(time, var_time, color="purple")
axes[1, 1].set_title("Variance Over Time")
axes[1, 1].set_xlabel("Time (s)")

# Mean signal
mean_time = np.mean(X_all, axis=(0, 1))
axes[1, 2].plot(time, mean_time, color="darkgreen")
axes[1, 2].fill_between(
    time,
    mean_time - np.std(X_all[:, 0, :]),
    mean_time + np.std(X_all[:, 0, :]),
    alpha=0.3,
)
axes[1, 2].set_title("Mean Signal Over Time")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_signal_characteristics.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: 02_signal_characteristics.png")

# ============================================================================
# 3. FREQUENCY DOMAIN - PSD
# ============================================================================
print("\n[4/10] Frequency domain analysis...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))


def compute_psd(data, fs=250):
    psds = []
    for trial in data:
        for ch in range(trial.shape[0]):
            f, p = signal.welch(trial[ch], fs=fs, nperseg=256)
            psds.append(p)
    return f, np.array(psds)


psds_by_class = {}
for cls in range(4):
    cls_data = X_all[y_all == cls]
    f, psd = compute_psd(cls_data)
    psds_by_class[cls] = psd

# PSD by class
for name, color, cls in zip(CLASS_NAMES, COLORS, range(4)):
    axes[0, 0].semilogy(
        f, np.mean(psds_by_class[cls], axis=0), label=name, color=color, linewidth=2
    )
axes[0, 0].set_title("Power Spectral Density by Class")
axes[0, 0].set_xlabel("Frequency (Hz)")
axes[0, 0].set_ylabel("PSD")
axes[0, 0].legend()
axes[0, 0].set_xlim([0, 50])
axes[0, 0].axvspan(8, 13, alpha=0.2, color="yellow", label="Alpha")
axes[0, 0].axvspan(13, 30, alpha=0.2, color="green", label="Beta")

# Band powers
bands = {
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta": (13, 30),
    "Gamma": (30, 45),
}
band_powers = {b: [] for b in bands}
for cls in range(4):
    for band, (lo, hi) in bands.items():
        idx = np.where((f >= lo) & (f < hi))[0]
        power = np.mean(psds_by_class[cls][:, idx], axis=1)
        band_powers[band].append(power)

bp_data = []
bp_labels = []
for band_name in bands.keys():
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        bp_data.append(band_powers[band_name][cls_idx])
        bp_labels.append(f"{band_name[:3]}\n{cls_name[:3]}")

axes[0, 1].boxplot(bp_data, showfliers=False)
axes[0, 1].set_xticklabels(bp_labels, rotation=45, ha="right", fontsize=7)
axes[0, 1].set_title("Band Power by Class")

# Alpha power
alpha_pow = {c: [] for c in range(4)}
for cls in range(4):
    for trial in X_all[y_all == cls]:
        for ch in range(trial.shape[0]):
            f, p = signal.welch(trial[ch], fs=250, nperseg=256)
            idx = np.where((f >= 8) & (f <= 13))[0]
            alpha_pow[cls].append(np.mean(p[idx]))

axes[1, 0].boxplot([alpha_pow[c] for c in range(4)], labels=CLASS_NAMES)
axes[1, 0].set_title("Alpha Power (8-13 Hz) by Class")

# Beta power
beta_pow = {c: [] for c in range(4)}
for cls in range(4):
    for trial in X_all[y_all == cls]:
        for ch in range(trial.shape[0]):
            f, p = signal.welch(trial[ch], fs=250, nperseg=256)
            idx = np.where((f >= 13) & (f <= 30))[0]
            beta_pow[cls].append(np.mean(p[idx]))

axes[1, 1].boxplot([beta_pow[c] for c in range(4)], labels=CLASS_NAMES)
axes[1, 1].set_title("Beta Power (13-30 Hz) by Class")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_frequency_psd.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: 03_frequency_psd.png")

# ============================================================================
# 4. CHANNEL-WISE ANALYSIS
# ============================================================================
print("\n[5/10] Channel-wise analysis...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

n_ch = min(20, X_all.shape[1])  # Use 20 to match available positions
channel_names = [
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "T7",
    "C3",
    "Cz",
    "C4",
    "T8",
    "P7",
    "P3",
    "Pz",
    "P4",
    "P8",
    "O1",
    "Oz",
    "O2",
][::2][:n_ch]

# Channel activity
ch_act = np.mean(np.abs(X_all[:, :n_ch, :]), axis=(0, 2))
axes[0, 0].barh(range(n_ch), ch_act, color="steelblue")
axes[0, 0].set_title("Mean Channel Activity")
axes[0, 0].set_xlabel("Mean |Amplitude|")

# Channel variability
ch_var = np.std(X_all[:, :n_ch, :], axis=(0, 2))
axes[0, 1].barh(range(n_ch), ch_var, color="coral")
axes[0, 1].set_title("Channel Variability")

# Discriminativity (F-score)
f_scores = []
for ch in range(n_ch):
    class_means = [np.mean(X_all[y_all == c, ch, :]) for c in range(4)]
    overall = np.mean(X_all[:, ch, :])
    between = sum(
        [np.sum(y_all == c) * (m - overall) ** 2 for c, m in enumerate(class_means)]
    )
    within = np.var(X_all[:, ch, :])
    f_scores.append(between / (within + 1e-10))

axes[1, 0].barh(range(n_ch), f_scores, color="green")
axes[1, 0].set_title("Class Discriminativity (F-score)")

# Top 5 channels
top5 = np.argsort(f_scores)[-5:][::-1]
top5_data = []
for ch in top5:
    top5_data.append([np.mean(X_all[y_all == c, ch, :]) for c in range(4)])

x = np.arange(4)
w = 0.15
for i, ch in enumerate(top5):
    axes[1, 1].bar(x + i * w, top5_data[i], w, label=f"Ch {ch}")
axes[1, 1].set_xticks(x + 2 * w)
axes[1, 1].set_xticklabels(CLASS_NAMES)
axes[1, 1].set_title("Top 5 Discriminative Channels")
axes[1, 1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_channel_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: 04_channel_analysis.png")

# ============================================================================
# 5. TOPOGRAPHIC MAPS - ENHANCED WITH LABELS AND HEAD OUTLINE
# ============================================================================
print("\n[6/10] Topographic maps (enhanced)...")
n_ch = X_all.shape[1]
ch_names = CHANNEL_NAMES_25[:n_ch]
pos = np.array([CHANNEL_POSITIONS[ch] for ch in ch_names])


def plot_topomap(
    ax, data, title, cmap="RdBu_r", vmin=None, vmax=None, show_labels=True
):
    head_circle = Circle((0, 0), 0.95, fill=False, color="black", linewidth=2)
    ax.add_patch(head_circle)
    nose = Polygon(
        [(0, 0.95), (-0.05, 0.85), (0.05, 0.85)], closed=True, fill=True, color="black"
    )
    ax.add_patch(nose)
    ear_l = Circle((-0.95, 0), 0.05, fill=True, color="black")
    ear_r = Circle((0.95, 0), 0.05, fill=True, color="black")
    ax.add_patch(ear_l)
    ax.add_patch(ear_r)

    if vmin is None:
        vmin = np.nanmin(data)
        vmax = np.nanmax(data)

    grid_x, grid_y = np.mgrid[-1:1:100j, -1:1:100j]
    try:
        grid_z = griddata(
            pos, data, (grid_x, grid_y), method="cubic", fill_value=np.nan
        )
        im = ax.contourf(
            grid_x,
            grid_y,
            grid_z,
            levels=20,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=0.8,
        )
    except:
        im = ax.tricontourf(
            pos[:, 0], pos[:, 1], data, levels=20, cmap=cmap, vmin=vmin, vmax=vmax
        )

    scatter = ax.scatter(
        pos[:, 0],
        pos[:, 1],
        c=data,
        cmap=cmap,
        s=100,
        edgecolors="black",
        linewidths=1,
        vmin=vmin,
        vmax=vmax,
        zorder=5,
    )

    if show_labels:
        for i, name in enumerate(ch_names):
            ax.annotate(
                name,
                (pos[i, 0], pos[i, 1]),
                fontsize=7,
                ha="center",
                va="center",
                fontweight="bold",
                zorder=10,
            )

    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold")

    return scatter, vmin, vmax


fig, axes = plt.subplots(2, 4, figsize=(16, 8))
all_data_maps = []
for cls in range(4):
    data = np.mean(X_all[y_all == cls, :n_ch, :], axis=(0, 2))
    all_data_maps.append(data)

vmin_all = min(np.nanmin(d) for d in all_data_maps)
vmax_all = max(np.nanmax(d) for d in all_data_maps)

for cls in range(4):
    ax = axes[0, cls]
    data = np.mean(X_all[y_all == cls, :n_ch, :], axis=(0, 2))
    scatter, _, _ = plot_topomap(
        ax, data, f"{CLASS_NAMES[cls]}", vmin=vmin_all, vmax=vmax_all
    )

for ax in axes[0, :]:
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)

diff = np.mean(X_all[y_all == 0, :n_ch, :], axis=(0, 2)) - np.mean(
    X_all[y_all == 1, :n_ch, :], axis=(0, 2)
)
scatter, _, _ = plot_topomap(
    axes[1, 0],
    diff,
    "Left - Right Hand",
    cmap="RdBu_r",
    vmin=-np.nanmax(np.abs(diff)),
    vmax=np.nanmax(np.abs(diff)),
)
plt.colorbar(scatter, ax=axes[1, 0], shrink=0.6)

diff_feet = np.mean(X_all[y_all == 2, :n_ch, :], axis=(0, 2)) - np.mean(
    X_all[y_all == 3, :n_ch, :], axis=(0, 2)
)
scatter, _, _ = plot_topomap(
    axes[1, 1],
    diff_feet,
    "Feet - Tongue",
    cmap="RdBu_r",
    vmin=-np.nanmax(np.abs(diff_feet)),
    vmax=np.nanmax(np.abs(diff_feet)),
)
plt.colorbar(scatter, ax=axes[1, 1], shrink=0.6)

axes[1, 2].axis("off")
axes[1, 3].axis("off")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_topographic_maps.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: 05_topographic_maps.png")

# ============================================================================
# 5b. MOTOR CORTEX FOCUS - C3, Cz, C4
# ============================================================================
print("\n[6b/10] Motor cortex focus topography...")
n_ch_full = X_all.shape[1]
motor_ch_names = ["C3", "Cz", "C4"]
motor_pos = np.array([CHANNEL_POSITIONS[ch] for ch in motor_ch_names])
ch_names_full = CHANNEL_NAMES_25[:n_ch_full]
motor_ch_indices = [list(ch_names_full).index(ch) for ch in motor_ch_names]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

all_motor_data = []
for cls in range(4):
    data = np.mean(X_all[y_all == cls, :, :], axis=(0, 2))  # Use all 25 channels
    motor_vals = data[motor_ch_indices]
    all_motor_data.append(motor_vals)

vmin_m = min(np.nanmin(d) for d in all_motor_data)
vmax_m = max(np.nanmax(d) for d in all_motor_data)

for cls in range(4):
    data = np.mean(X_all[y_all == cls, :, :], axis=(0, 2))
    motor_vals = data[motor_ch_indices].flatten()

    head_circle = Circle((0, 0), 0.95, fill=False, color="black", linewidth=2)
    axes[cls].add_patch(head_circle)
    nose = Polygon(
        [(0, 0.95), (-0.05, 0.85), (0.05, 0.85)], closed=True, fill=True, color="black"
    )
    axes[cls].add_patch(nose)

    scatter = axes[cls].scatter(
        motor_pos[:, 0],
        motor_pos[:, 1],
        c=motor_vals,
        cmap="RdBu_r",
        s=300,
        edgecolors="black",
        linewidths=2,
        vmin=vmin_m,
        vmax=vmax_m,
        zorder=5,
    )

    for i, name in enumerate(motor_ch_names):
        axes[cls].annotate(
            name,
            (motor_pos[i, 0], motor_pos[i, 1]),
            fontsize=14,
            ha="center",
            va="center",
            fontweight="bold",
            color="white",
            zorder=10,
        )

    axes[cls].set_xlim([-1.1, 1.1])
    axes[cls].set_ylim([-1.1, 1.1])
    axes[cls].set_aspect("equal")
    axes[cls].axis("off")
    axes[cls].set_title(CLASS_NAMES[cls], fontsize=12, fontweight="bold")

plt.colorbar(scatter, ax=axes, shrink=0.5, label="Mean Activity")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05b_motor_cortex_focus.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: 05b_motor_cortex_focus.png")

# ============================================================================
# 6. TEMPORAL DYNAMICS (ERP)
# ============================================================================
print("\n[7/10] Temporal dynamics analysis...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ERP by class at Cz (channel 9)
erp_by_class = {}
for cls in range(4):
    erp_by_class[cls] = np.mean(X_all[y_all == cls], axis=0)

for name, color, cls in zip(CLASS_NAMES, COLORS, range(4)):
    axes[0, 0].plot(time, erp_by_class[cls][9, :], label=name, color=color, linewidth=2)
axes[0, 0].set_title("ERP at Cz Channel")
axes[0, 0].set_xlabel("Time (s)")
axes[0, 0].legend()
axes[0, 0].axvline(0, color="black", linestyle="--", alpha=0.5)

# Motor cortex average
motor_chs = [8, 9, 10]  # C3, Cz, C4
for name, color, cls in zip(CLASS_NAMES, COLORS, range(4)):
    erp_motor = np.mean([erp_by_class[cls][ch, :] for ch in motor_chs], axis=0)
    axes[0, 1].plot(time, erp_motor, label=name, color=color, linewidth=2)
axes[0, 1].set_title("Motor Cortex ERP (C3,Cz,C4)")
axes[0, 1].set_xlabel("Time (s)")
axes[0, 1].legend()
axes[0, 1].axvline(0, color="black", linestyle="--", alpha=0.5)

# Time-frequency for one class
trial = X_all[y_all == 0][0]
tfr = []
for ch in range(min(5, X_all.shape[1])):
    f, t, sxx = signal.spectrogram(trial[ch], fs=250, nperseg=64)
    tfr.append(sxx)
tfr = np.mean(tfr, axis=0)
im = axes[1, 0].pcolormesh(
    t, f[:30], np.log10(tfr[:30, :] + 1e-10), cmap="viridis", shading="gouraud"
)
axes[1, 0].set_title("Time-Frequency (Left Hand)")
axes[1, 0].set_xlabel("Time (s)")
axes[1, 0].set_ylabel("Frequency (Hz)")
plt.colorbar(im, ax=axes[1, 0])

# Trial variability - variance per trial (averaged across channels and time)
trials = X_all[y_all == 0][:50]
trial_var = np.var(trials, axis=(1, 2))  # Variance per trial (50,)
axes[1, 1].plot(
    range(len(trial_var)), trial_var, marker="o", markersize=3, color="steelblue"
)
axes[1, 1].set_title("Trial-to-Trial Variability (Left Hand)")
axes[1, 1].set_xlabel("Trial Number")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_temporal_dynamics.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: 06_temporal_dynamics.png")

# ============================================================================
# 7. SUBJECT ANALYSIS
# ============================================================================
print("\n[8/10] Subject analysis...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Trials per subject
trials_per_subj = [all_data[s].shape[0] for s in SUBJECTS]
axes[0, 0].bar(SUBJECTS, trials_per_subj, color="teal")
axes[0, 0].set_title("Trials per Subject")

# Mean amplitude by subject
subj_means = [np.mean(np.abs(all_data[s])) for s in SUBJECTS]
axes[0, 1].bar(SUBJECTS, subj_means, color="coral")
axes[0, 1].set_title("Mean |Amplitude| by Subject")

# Variance by subject
subj_vars = [np.var(all_data[s]) for s in SUBJECTS]
axes[0, 2].bar(SUBJECTS, subj_vars, color="purple")
axes[0, 2].set_title("Variance by Subject")

# Subject similarity - compare subjects based on mean activity patterns
subj_pats = []
for s in SUBJECTS:
    subj_pats.append(np.mean(all_data[s], axis=0).flatten())  # (25, 1126) -> flatten
subj_pats = np.array(subj_pats)  # (9, 28226)
# Use PCA to reduce dimensionality before correlation
from sklearn.decomposition import PCA

pca_subj = PCA(n_components=min(5, len(SUBJECTS)))
subj_pats_pca = pca_subj.fit_transform(subj_pats)
corr = np.corrcoef(subj_pats_pca)
im = axes[1, 0].imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
axes[1, 0].set_xticks(range(len(SUBJECTS)))
axes[1, 0].set_yticks(range(len(SUBJECTS)))
axes[1, 0].set_xticklabels(SUBJECTS, rotation=45)
axes[1, 0].set_yticklabels(SUBJECTS)
axes[1, 0].set_title("Subject Similarity")
plt.colorbar(im, ax=axes[1, 0])

# Class distribution by subject
bottom = np.zeros(len(SUBJECTS))
for cls in range(4):
    cls_cnt = [np.sum(all_labels[s] == cls) for s in SUBJECTS]
    axes[1, 1].bar(
        SUBJECTS, cls_cnt, bottom=bottom, label=CLASS_NAMES[cls], color=COLORS[cls]
    )
    bottom += cls_cnt
axes[1, 1].set_title("Class Distribution by Subject")
axes[1, 1].legend(fontsize=8)

# Inter-subject variability over time - variance across subjects averaged across channels
subj_mean_trial = np.array(
    [np.mean(all_data[s], axis=0) for s in SUBJECTS]
)  # (9, 25, 1126)
time_var = np.var(subj_mean_trial, axis=0).mean(
    axis=0
)  # variance across subjects, mean across channels -> (1126,)
axes[1, 2].plot(time, time_var, color="darkgreen")
axes[1, 2].set_title("Inter-Subject Variability Over Time")
axes[1, 2].set_xlabel("Time (s)")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_subject_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: 07_subject_analysis.png")

# ============================================================================
# 8. CORRELATION & SEPARABILITY - ENHANCED WITH COLORED LABELS
# ============================================================================
print("\n[9/10] Correlation and separability analysis (enhanced)...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Inter-channel correlation with colored tick labels
all_corrs = []
for s in SUBJECTS:
    c = np.corrcoef(all_data[s].mean(axis=2).T)
    all_corrs.append(c)
avg_corr = np.mean(all_corrs, axis=0)
im1 = axes[0].imshow(avg_corr[:n_ch, :n_ch], cmap="coolwarm", vmin=-1, vmax=1)
axes[0].set_title("Inter-Channel Correlation", fontweight="bold")

# Determine left/right hemisphere for each channel
left_channels = {"Fp1", "F7", "F3", "F1", "T7", "C3", "P7", "P3", "P1", "O1"}
tick_labels = ch_names[:n_ch]
colors = ["#e74c3c" if ch in left_channels else "#3498db" for ch in tick_labels]

for tick, color in zip(axes[0].get_xticklabels(), colors):
    tick.set_color(color)
    tick.set_fontweight("bold")
for tick, color in zip(axes[0].get_yticklabels(), colors):
    tick.set_color(color)
    tick.set_fontweight("bold")

axes[0].set_xticks(range(n_ch))
axes[0].set_yticks(range(n_ch))
axes[0].set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
axes[0].set_yticklabels(tick_labels, fontsize=8)
plt.colorbar(im1, ax=axes[0], label="Correlation")

# Add legend for hemisphere colors
from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor="#e74c3c", label="Left Hemisphere"),
    Patch(facecolor="#3498db", label="Right Hemisphere"),
]
axes[0].legend(handles=legend_elements, loc="upper right", fontsize=8)

# Prepare data for dimensionality reduction
X_flat = X_all[:, :n_ch, :].reshape(X_all.shape[0], -1)

# PCA - use distinct colors for each class
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_flat[:1000])
colors_array = np.array(COLORS)[y_all[:1000]]
scatter = axes[1].scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=colors_array,
    alpha=0.6,
    edgecolors="white",
    linewidths=0.5,
)
axes[1].set_title("PCA Projection", fontweight="bold")
axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")

# Add legend for PCA
legend_elements1 = [Patch(facecolor=c, label=n) for c, n in zip(COLORS, CLASS_NAMES)]
axes[1].legend(handles=legend_elements1, loc="best", fontsize=8)

# t-SNE - use distinct colors for each class
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_flat[:2000])
colors_array_tsne = np.array(COLORS)[y_all[:2000]]
scatter = axes[2].scatter(
    X_tsne[:, 0],
    X_tsne[:, 1],
    c=colors_array_tsne,
    alpha=0.6,
    edgecolors="white",
    linewidths=0.5,
)
axes[2].set_title("t-SNE Projection", fontweight="bold")

# Add legend for t-SNE
legend_elements2 = [Patch(facecolor=c, label=n) for c, n in zip(COLORS, CLASS_NAMES)]
axes[2].legend(handles=legend_elements2, loc="best", fontsize=8)

plt.tight_layout()
plt.savefig(
    f"{OUTPUT_DIR}/08_correlation_separability.png", dpi=150, bbox_inches="tight"
)
plt.close()
print(f"  Saved: 08_correlation_separability.png")

# ============================================================================
# 9. CLASS-WISE BAND POWER HEATMAP
# ============================================================================
print("\n[10/10] Class-wise band power heatmap...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Compute band powers for each channel and class
bands = {"Alpha": (8, 13), "Beta": (13, 30)}
band_power_data = {band: np.zeros((4, n_ch)) for band in bands}

for cls in range(4):
    cls_data = X_all[y_all == cls]
    n_trials = cls_data.shape[0]
    for ch in range(n_ch):
        for trial in cls_data:
            f, p = signal.welch(trial[ch], fs=250, nperseg=256)
            for band_name, (lo, hi) in bands.items():
                idx = np.where((f >= lo) & (f < hi))[0]
                band_power_data[band_name][cls, ch] += np.mean(p[idx])

    for band_name in bands:
        band_power_data[band_name][cls, :] /= n_trials

# Alpha power heatmap
im1 = axes[0].imshow(band_power_data["Alpha"], cmap="viridis", aspect="auto")
axes[0].set_title("Alpha Power (8-13 Hz) by Class and Channel", fontweight="bold")
axes[0].set_xlabel("Channel")
axes[0].set_ylabel("Class")
axes[0].set_yticks(range(4))
axes[0].set_yticklabels(CLASS_NAMES)
axes[0].set_xticks(range(n_ch))
axes[0].set_xticklabels(ch_names, rotation=45, ha="right", fontsize=8)
plt.colorbar(im1, ax=axes[0], label="Power")

# Beta power heatmap
im2 = axes[1].imshow(band_power_data["Beta"], cmap="viridis", aspect="auto")
axes[1].set_title("Beta Power (13-30 Hz) by Class and Channel", fontweight="bold")
axes[1].set_xlabel("Channel")
axes[1].set_ylabel("Class")
axes[1].set_yticks(range(4))
axes[1].set_yticklabels(CLASS_NAMES)
axes[1].set_xticks(range(n_ch))
axes[1].set_xticklabels(ch_names, rotation=45, ha="right", fontsize=8)
plt.colorbar(im2, ax=axes[1], label="Power")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/09_band_power_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: 09_band_power_heatmap.png")

# ============================================================================
# 10. TIME-FREQUENCY BY CHANNEL (MOTOR CHANNELS)
# ============================================================================
print("\n[11/10] Time-frequency analysis for motor channels...")
motor_channels = ["C3", "Cz", "C4"]
ch_names_full = CHANNEL_NAMES_25[: X_all.shape[1]]
motor_idx = [list(ch_names_full).index(ch) for ch in motor_channels]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (ch_name, ch_idx) in enumerate(zip(motor_channels, motor_idx)):
    trials = X_all[y_all == 0, ch_idx, :]

    f, t, sxx = signal.spectrogram(trials[0], fs=250, nperseg=64, noverlap=32)
    n_freq = min(30, len(f))
    n_time = len(t)
    mean_tfr = np.zeros((n_freq, n_time))

    count = 0
    for trial in trials[:30]:
        f, t, sxx = signal.spectrogram(trial, fs=250, nperseg=64, noverlap=32)
        if sxx.shape[1] == n_time:
            mean_tfr += np.log10(sxx[:n_freq, :] + 1e-10)
            count += 1
    if count > 0:
        mean_tfr /= count

    im = axes[idx].pcolormesh(
        t, f[:n_freq], mean_tfr, cmap="viridis", shading="gouraud"
    )
    axes[idx].set_title(f"Time-Frequency: {ch_name} (Left Hand)", fontweight="bold")
    axes[idx].set_xlabel("Time (s)")
    axes[idx].set_ylabel("Frequency (Hz)")
    axes[idx].set_ylim([0, 30])
    plt.colorbar(im, ax=axes[idx], label="Log Power")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/10_timefreq_motor.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: 10_timefreq_motor.png")

# ============================================================================
# 11. SUBJECT PERFORMANCE VARIABILITY
# ============================================================================
print("\n[12/10] Subject performance variability...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Compute per-subject class statistics
subj_class_stats = {s: {} for s in SUBJECTS}
for s in SUBJECTS:
    for cls in range(4):
        cls_data = all_data[s][all_labels[s] == cls]
        subj_class_stats[s][cls] = {
            "mean": np.mean(cls_data),
            "std": np.std(cls_data),
            "trials": cls_data.shape[0],
        }

# Per-subject class mean activity
ax = axes[0, 0]
for cls in range(4):
    means = [subj_class_stats[s][cls]["mean"] for s in SUBJECTS]
    ax.plot(
        SUBJECTS,
        means,
        marker="o",
        label=CLASS_NAMES[cls],
        color=COLORS[cls],
        linewidth=2,
    )
ax.set_title("Mean Activity by Subject and Class", fontweight="bold")
ax.set_xlabel("Subject")
ax.set_ylabel("Mean Amplitude")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Per-subject class variability
ax = axes[0, 1]
for cls in range(4):
    stds = [subj_class_stats[s][cls]["std"] for s in SUBJECTS]
    ax.bar(
        np.arange(len(SUBJECTS)) + cls * 0.2,
        stds,
        0.2,
        label=CLASS_NAMES[cls],
        color=COLORS[cls],
    )
ax.set_xticks(np.arange(len(SUBJECTS)) + 0.3)
ax.set_xticklabels(SUBJECTS)
ax.set_title("Signal Variability by Subject and Class", fontweight="bold")
ax.set_xlabel("Subject")
ax.set_ylabel("Std Dev")
ax.legend(fontsize=8)

# Subject class balance
ax = axes[1, 0]
class_balance = []
for s in SUBJECTS:
    counts = [subj_class_stats[s][cls]["trials"] for cls in range(4)]
    balance = min(counts) / max(counts) if max(counts) > 0 else 0
    class_balance.append(balance)
ax.bar(SUBJECTS, class_balance, color="teal")
ax.set_title("Class Balance per Subject", fontweight="bold")
ax.set_xlabel("Subject")
ax.set_ylabel("Balance Ratio (min/max)")
ax.set_ylim([0, 1])

# Inter-subject variability heatmap
ax = axes[1, 1]
subj_means_matrix = np.zeros((len(SUBJECTS), n_ch))
for i, s in enumerate(SUBJECTS):
    subj_means_matrix[i, :] = np.mean(all_data[s], axis=(0, 2))

im = ax.imshow(subj_means_matrix, cmap="RdBu_r", aspect="auto")
ax.set_yticks(range(len(SUBJECTS)))
ax.set_yticklabels(SUBJECTS)
ax.set_xticks(range(n_ch))
ax.set_xticklabels(ch_names, rotation=45, ha="right", fontsize=8)
ax.set_title("Mean Activity by Subject and Channel", fontweight="bold")
ax.set_xlabel("Channel")
ax.set_ylabel("Subject")
plt.colorbar(im, ax=ax, label="Mean Activity")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/11_subject_variability.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: 11_subject_variability.png")

# ============================================================================
# 9. OUTLIER DETECTION
# ============================================================================
print("\n[10/10] Outlier detection...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Z-score scatter - use distinct colors for each class
trial_means = np.mean(X_all, axis=(1, 2))
trial_stds = np.std(X_all, axis=(1, 2))
z_means = (trial_means - np.mean(trial_means)) / np.std(trial_means)
z_stds = (trial_stds - np.mean(trial_stds)) / np.std(trial_stds)

colors_zscore = np.array(COLORS)[y_all]
axes[0, 0].scatter(
    z_means, z_stds, c=colors_zscore, alpha=0.6, edgecolors="white", linewidths=0.3
)
for v in [3, -3]:
    axes[0, 0].axhline(v, color="red", linestyle="--", alpha=0.5)
    axes[0, 0].axvline(v, color="red", linestyle="--", alpha=0.5)
axes[0, 0].set_title("Trial Z-Score Distribution")
axes[0, 0].set_xlabel("Z-score of Mean")
axes[0, 0].set_ylabel("Z-score of Std")

# Add legend for z-score plot
legend_elements_z = [Patch(facecolor=c, label=n) for c, n in zip(COLORS, CLASS_NAMES)]
axes[0, 0].legend(handles=legend_elements_z, loc="upper right", fontsize=8)

# Outlier pie
outliers = (np.abs(z_means) > 3) | (np.abs(z_stds) > 3)
outlier_cnt = np.sum(outliers)
axes[0, 1].pie(
    [outlier_cnt, len(y_all) - outlier_cnt],
    labels=["Outliers", "Normal"],
    autopct="%1.1f%%",
    colors=["red", "green"],
)
axes[0, 1].set_title(f"Outlier Detection ({outlier_cnt} trials)")

# Channel outliers
ch_outliers = []
for ch in range(n_ch):
    ch_d = X_all[:, ch, :].flatten()
    z_ch = (ch_d - np.mean(ch_d)) / np.std(ch_d)
    ch_outliers.append(np.sum(np.abs(z_ch) > 3))
axes[1, 0].bar(range(n_ch), ch_outliers, color="coral")
axes[1, 0].set_title("Outliers by Channel")

# Subject outlier rates
subj_out_rates = []
for s in SUBJECTS:
    s_data = all_data[s]
    s_mean, s_std = np.mean(s_data), np.std(s_data)
    out = np.sum(np.abs(s_data - s_mean) > 3 * s_std) / s_data.size
    subj_out_rates.append(out * 100)
axes[1, 1].bar(SUBJECTS, subj_out_rates, color="purple")
axes[1, 1].set_title("Outlier Rate by Subject")
axes[1, 1].set_ylabel("Outlier Rate (%)")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/12_outlier_detection.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: 12_outlier_detection.png")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n[FINAL] Generating summary report...")

summary = f"""
EEG MOTOR IMAGERY DATA SUMMARY
{"=" * 50}

DATASET OVERVIEW
{"-" * 30}
Total Trials: {X_all.shape[0]}
Channels: {X_all.shape[1]}
Time Points: {X_all.shape[2]}
Sampling Rate: 250 Hz
Duration: {X_all.shape[2] / 250:.2f} seconds
Classes: 4
Subjects: 9

SIGNAL STATISTICS
{"-" * 30}
Mean: {np.mean(X_all):.4f}
Std: {np.std(X_all):.4f}
Min: {np.min(X_all):.4f}
Max: {np.max(X_all):.4f}

CLASS DISTRIBUTION
{"-" * 30}
"""
for name, cnt in zip(CLASS_NAMES, class_counts):
    summary += f"{name}: {cnt} ({cnt / len(y_all) * 100:.1f}%)\n"

summary += f"""
SUBJECT STATISTICS
{"-" * 30}
"""
for s in SUBJECTS:
    summary += f"{s}: {all_data[s].shape[0]} trials, mean={np.mean(all_data[s]):.4f}\n"

summary += f"""
{"=" * 50}
Generated by Comprehensive EDA Pipeline
{"=" * 50}
"""

with open(f"{OUTPUT_DIR}/summary_statistics.txt", "w") as f:
    f.write(summary)

print(f"  Saved: summary_statistics.txt")

# ============================================================================
# COMPLETE
# ============================================================================
print("\n" + "=" * 60)
print("EDA COMPLETE!")
print("=" * 60)
print(f"\nGenerated files in: {OUTPUT_DIR}/")
print("  01_class_distribution.png")
print("  02_signal_characteristics.png")
print("  03_frequency_psd.png")
print("  04_channel_analysis.png")
print("  05_topographic_maps.png")
print("  05b_motor_cortex_focus.png")
print("  06_temporal_dynamics.png")
print("  07_subject_analysis.png")
print("  08_correlation_separability.png")
print("  09_band_power_heatmap.png")
print("  10_timefreq_motor.png")
print("  11_subject_variability.png")
print("  12_outlier_detection.png")
print("  summary_statistics.txt")
print("=" * 60)
