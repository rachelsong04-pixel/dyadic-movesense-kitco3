"""
Stomp Visualizer — Two Participants, Two Stomps Each
-----------------------------------------------------
Finds the top 2 acceleration spikes per participant (start + end stomps)
and plots them side-by-side.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.signal import find_peaks

# ── CONFIG ────────────────────────────────────────────────────────────────────
P1_FILE = "/Volumes/Klimt/Studies/Collaborative Kitchen Cognition (KITCO)/KITCO Study C/Food Preparation Study/4. Raw Data/Participants 61-64/KITCO3Feb25no1 (MCL)/2026_02_25-22_07_30/242930002204/Acc-2026_02_25-22_07_30.csv"
P2_FILE = "/Volumes/Klimt/Studies/Collaborative Kitchen Cognition (KITCO)/KITCO Study C/Food Preparation Study/4. Raw Data/Participants 61-64/KITCO3Feb25no1 (MCL)/2026_02_25-22_07_30/242930002205/Acc-2026_02_25-22_07_30.csv"

P1_LABEL = "242930002204"
P2_LABEL = "242930002205"

TIME_COL = "Timestamp"
X_COL    = "AccX"
Y_COL    = "AccY"
Z_COL    = "AccZ"

STOMP_THRESHOLD_SD = 6   # SD above mean to count as a stomp
MIN_SEPARATION_S   = 200  # minimum seconds between the two stomps
# ──────────────────────────────────────────────────────────────────────────────


def load_acc(filepath):
    df = pd.read_csv(filepath, skiprows=1)
    print(f"\n📂 Loaded: {filepath}")
    print(f"   Columns : {list(df.columns)}")
    print(f"   Shape   : {df.shape}")
    return df


def compute_magnitude(df, x, y, z):
    df = df.copy()
    df["magnitude"] = np.sqrt(df[x]**2 + df[y]**2 + df[z]**2)
    return df


def estimate_sample_rate(times):
    """Estimate samples per second, handling both second and millisecond timestamps."""
    duration = times[-1] - times[0]
    n = len(times)
    rate = n / duration  # samples per time-unit

    # If timestamps are in milliseconds, rate will be tiny (e.g. 0.013)
    # If timestamps are in seconds, rate will be ~13 Hz
    # Normalize to samples/second
    if rate < 1:
        # timestamps are likely in milliseconds
        rate = rate * 1000
        print(f"   ℹ️  Timestamps appear to be in milliseconds (rate ≈ {rate:.1f} Hz)")
    else:
        print(f"   ℹ️  Timestamps appear to be in seconds (rate ≈ {rate:.1f} Hz)")
    return rate


def find_two_stomps(df, time_col, mag_col="magnitude"):
    """
    Find the two largest spikes at least MIN_SEPARATION_S apart.
    Returns list of (df_index, timestamp) sorted chronologically.
    """
    mean = df[mag_col].mean()
    sd   = df[mag_col].std()
    threshold = mean + STOMP_THRESHOLD_SD * sd

    times = df[time_col].values
    sample_rate = estimate_sample_rate(times)
    min_sep_samples = max(1, int(MIN_SEPARATION_S * sample_rate))

    print(f"   ℹ️  Min separation: {MIN_SEPARATION_S}s = {min_sep_samples} samples")

    peak_indices, _ = find_peaks(df[mag_col].values,
                                 height=threshold,
                                 distance=min_sep_samples)

    if len(peak_indices) == 0:
        print(f"   ⚠️  No stomps found at {STOMP_THRESHOLD_SD} SD — try lowering STOMP_THRESHOLD_SD")
        return []

    # Take top 2 by magnitude, re-sort chronologically
    peak_mags = df[mag_col].values[peak_indices]
    top2_order = np.argsort(peak_mags)[::-1][:2]
    top2_idx = np.sort(peak_indices[top2_order])

    results = []
    for rank, idx in enumerate(top2_idx):
        ts = df[time_col].iloc[idx]
        label = "START" if rank == 0 else "END"
        print(f"   🦶 Stomp {label}: timestamp={ts:.4f}, magnitude={df[mag_col].iloc[idx]:.3f}")
        results.append((idx, ts))

    return results


def plot_participant(ax_list, df, t, x, y, z, label, color):
    if "magnitude" not in df.columns:
        df = compute_magnitude(df, x, y, z)
    stomps = find_two_stomps(df, t)

    stomp_colors = ["#FF4C4C", "#FFB347"]  # red = start, orange = end
    stomp_labels = ["Stomp START", "Stomp END"]

    axes_data = [(x, "X"), (y, "Y"), (z, "Z"), ("magnitude", "Magnitude")]
    alphas    = [0.55, 0.55, 0.55, 1.0]

    for ax, (col, name), alpha in zip(ax_list, axes_data, alphas):
        lw = 0.8 if name != "Magnitude" else 1.5
        ax.plot(df[t], df[col], color=color, alpha=alpha, linewidth=lw)

        for i, (stomp_idx, stomp_time) in enumerate(stomps):
            sc = stomp_colors[i]
            sl = stomp_labels[i]
            ax.axvline(stomp_time, color=sc, linewidth=1.8, linestyle="--", label=sl)
            ax.scatter(stomp_time, df[col].iloc[stomp_idx], color=sc, zorder=5, s=60)

        ax.set_ylabel(name, fontsize=8, color="lightgray")
        ax.tick_params(colors="gray", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        ax.set_facecolor("#0d0d0d")

    ax_list[0].set_title(label, fontsize=11, color="white", pad=6)
    ax_list[-1].legend(fontsize=7, facecolor="#222", labelcolor="white",
                       loc="upper right", framealpha=0.7)


def main():
    df1 = load_acc(P1_FILE)
    df2 = load_acc(P2_FILE)

    fig = plt.figure(figsize=(16, 10), facecolor="#111111")
    fig.suptitle("Accelerometer — Start & End Stomp Detection", fontsize=14,
                 color="white", fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(4, 2, hspace=0.45, wspace=0.3,
                           left=0.07, right=0.97, top=0.94, bottom=0.06)

    axes_p1 = [fig.add_subplot(gs[r, 0]) for r in range(4)]
    axes_p2 = [fig.add_subplot(gs[r, 1]) for r in range(4)]

    df1 = compute_magnitude(df1, X_COL, Y_COL, Z_COL)
    stomps1 = find_two_stomps(df1, TIME_COL)
    save_annotations(df1, TIME_COL, stomps1, P1_LABEL)
    plot_participant(axes_p1, df1, TIME_COL, X_COL, Y_COL, Z_COL, P1_LABEL, "#4FC3F7")
    df2 = compute_magnitude(df2, X_COL, Y_COL, Z_COL)
    stomps2 = find_two_stomps(df2, TIME_COL)
    save_annotations(df2, TIME_COL, stomps2, P2_LABEL)
    plot_participant(axes_p2, df2, TIME_COL, X_COL, Y_COL, Z_COL, P2_LABEL, "#81C784")

    for ax in [axes_p1[-1], axes_p2[-1]]:
        ax.set_xlabel("Timestamp (s)", fontsize=8, color="lightgray")

    plt.savefig("stomp_visualization.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("\n✅ Saved → stomp_visualization.png")
    plt.show()


if __name__ == "__main__":
    main()


def save_annotations(df, time_col, stomps, device_id):
    """Save annotations CSV matching the standard format."""
    recording_start = df[time_col].iloc[0]
    recording_stop  = df[time_col].iloc[-1]

    rows = [{"Timestamp": recording_start, "Type": "RecordingStart",       "Content": ""}]

    if len(stomps) >= 1:
        rows.append({"Timestamp": stomps[0][1], "Type": "VideoAnnotationStart", "Content": ""})
    if len(stomps) >= 2:
        rows.append({"Timestamp": stomps[1][1], "Type": "VideoAnnotationStop",  "Content": ""})

    rows.append({"Timestamp": recording_stop, "Type": "RecordingStop", "Content": ""})

    out_df = pd.DataFrame(rows, columns=["Timestamp", "Type", "Content"])
    filename = f"Annotations-{device_id}.csv"
    out_df.to_csv(filename, index=False)
    print(f"   💾 Saved annotations → {filename}")
    print(out_df.to_string(index=False))