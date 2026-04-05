# ecg_analyser.py
# ECG Signal Analyser — Filter, R-peak Detection, HR & HRV
# Author: Vibha (Nova) | NMIT Bengaluru | ECE
#
# Pipeline:
#   Noisy ECG → Bandpass Filter → R-peak Detection
#               → Heart Rate → HRV → Full Plot

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import butter, filtfilt, find_peaks
from ecg_generator import generate_ecg_signal, add_artifact


# ──────────────────────────────────────────────────────────
# STEP 1: BANDPASS FILTER
# ──────────────────────────────────────────────────────────
# A real ECG signal sits between 0.5 Hz and 40 Hz.
# Anything below 0.5 Hz = baseline wander (breathing noise)
# Anything above 40 Hz  = muscle/powerline noise
#
# We use a Butterworth bandpass filter — maximally flat in
# the passband (no ripple). This is the standard in ECG
# processing (AHA guidelines recommend 0.5–40 Hz).
# ──────────────────────────────────────────────────────────

def bandpass_filter(signal, sample_rate, low_hz=0.5, high_hz=40.0, order=4):
    """
    Apply a Butterworth bandpass filter to the ECG signal.

    Parameters
    ----------
    signal      : raw noisy ECG (numpy array)
    sample_rate : Hz
    low_hz      : lower cutoff (removes baseline wander)
    high_hz     : upper cutoff (removes muscle/powerline noise)
    order       : filter order (higher = sharper cutoff, more phase shift)

    How it works:
    -------------
    butter() designs the filter coefficients (b, a)
    filtfilt() applies it FORWARD then BACKWARD — this gives
    zero phase distortion (no time shift), which is critical
    for accurate R-peak timing.
    """
    nyquist = sample_rate / 2.0   # Nyquist theorem: max detectable freq

    low  = low_hz  / nyquist      # Normalise to [0, 1]
    high = high_hz / nyquist

    # Design Butterworth bandpass filter
    b, a = butter(order, [low, high], btype='band')

    # Zero-phase filtering (forward + backward pass)
    filtered = filtfilt(b, a, signal)

    return filtered


# ──────────────────────────────────────────────────────────
# STEP 2: R-PEAK DETECTION
# ──────────────────────────────────────────────────────────
# The R-peak is the tallest spike in each heartbeat.
# Detecting it accurately gives us:
#   - When each beat occurred (R-peak timestamps)
#   - RR intervals (time between consecutive beats)
#   - Heart rate (60 / mean RR interval)
#   - HRV (variability in RR intervals)
#
# We use scipy's find_peaks() with two constraints:
#   height    = minimum amplitude (ignores small noise spikes)
#   distance  = minimum samples between peaks (avoids double-detection)
# ──────────────────────────────────────────────────────────

def detect_r_peaks(filtered_ecg, sample_rate, min_rr_sec=0.3):
    """
    Detect R-peaks in the filtered ECG signal.

    min_rr_sec: minimum time between two heartbeats (0.3s = 200 BPM max)
    This prevents double-detection of the same peak.

    Returns
    -------
    r_peaks : indices of R-peak locations in the signal array
    """
    # Minimum distance between peaks in samples
    min_distance = int(min_rr_sec * sample_rate)

    # Height threshold = 50% of signal max
    # (R-peaks are always the tallest feature)
    height_threshold = 0.5 * np.max(filtered_ecg)

    r_peaks, properties = find_peaks(
        filtered_ecg,
        height=height_threshold,
        distance=min_distance
    )

    return r_peaks


# ──────────────────────────────────────────────────────────
# STEP 3: HEART RATE & HRV
# ──────────────────────────────────────────────────────────

def compute_heart_rate(r_peaks, sample_rate):
    """
    Calculate heart rate from R-peak locations.

    RR interval = time between consecutive R-peaks (in seconds)
    Heart Rate  = 60 / mean(RR interval)  [BPM]
    """
    if len(r_peaks) < 2:
        return None, None

    # RR intervals in seconds
    rr_intervals_samples = np.diff(r_peaks)           # in samples
    rr_intervals_sec     = rr_intervals_samples / sample_rate  # convert to seconds

    mean_rr   = np.mean(rr_intervals_sec)
    heart_rate = 60.0 / mean_rr                       # BPM

    return heart_rate, rr_intervals_sec


def compute_hrv(rr_intervals_sec):
    """
    Heart Rate Variability (HRV) metrics.

    HRV measures how much the time between heartbeats varies.
    Higher HRV = healthier autonomic nervous system.

    SDNN  : Standard Deviation of NN intervals
            → overall HRV, long-term variability

    RMSSD : Root Mean Square of Successive Differences
            → short-term HRV, parasympathetic (vagal) activity
            → most commonly used clinical metric

    pNN50 : % of successive RR pairs differing by > 50ms
            → another parasympathetic marker
    """
    if rr_intervals_sec is None or len(rr_intervals_sec) < 2:
        return {}

    rr_ms = rr_intervals_sec * 1000   # convert to milliseconds

    sdnn  = np.std(rr_ms)
    rmssd = np.sqrt(np.mean(np.diff(rr_ms) ** 2))
    pnn50 = np.sum(np.abs(np.diff(rr_ms)) > 50) / len(rr_ms) * 100

    return {
        "SDNN (ms)"  : round(sdnn,  2),
        "RMSSD (ms)" : round(rmssd, 2),
        "pNN50 (%)"  : round(pnn50, 2),
    }


# ──────────────────────────────────────────────────────────
# STEP 4: FULL ANALYSIS PLOT
# ──────────────────────────────────────────────────────────

def plot_ecg_analysis(t, ecg_noisy, ecg_filtered, r_peaks, rr_intervals_sec,
                      heart_rate, hrv_metrics, sample_rate):
    """
    4-panel plot:
    1. Noisy raw ECG
    2. Filtered ECG with R-peaks marked
    3. RR interval tachogram (beat-to-beat variation)
    4. HRV metrics summary
    """
    fig = plt.figure(figsize=(14, 12))
    fig.patch.set_facecolor('#0e0e12')
    gs = gridspec.GridSpec(4, 1, hspace=0.6)

    # ── Panel 1: Raw noisy ECG ──
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor('#16161e')
    ax1.plot(t, ecg_noisy, color='#f87171', linewidth=0.8, alpha=0.9)
    ax1.set_title('Raw ECG Signal (with noise + baseline wander)',
                  color='white', fontsize=11, fontweight='bold', pad=6)
    ax1.set_ylabel('Amplitude (mV)', color='#888', fontsize=9)
    _style_ax(ax1)

    # ── Panel 2: Filtered ECG + R-peaks ──
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor('#16161e')
    ax2.plot(t, ecg_filtered, color='#4ade80', linewidth=1.0)
    ax2.plot(t[r_peaks], ecg_filtered[r_peaks],
             'v', color='#fbbf24', markersize=8, label=f'R-peaks ({len(r_peaks)} detected)')
    ax2.set_title(f'Filtered ECG — Bandpass 0.5–40 Hz | {len(r_peaks)} R-peaks detected',
                  color='white', fontsize=11, fontweight='bold', pad=6)
    ax2.set_ylabel('Amplitude (mV)', color='#888', fontsize=9)
    ax2.legend(facecolor='#1e1e2a', edgecolor='#2a2a3a',
               labelcolor='white', fontsize=9, loc='upper right')
    _style_ax(ax2)

    # ── Panel 3: RR Interval Tachogram ──
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor('#16161e')

    if rr_intervals_sec is not None:
        beat_numbers = np.arange(1, len(rr_intervals_sec) + 1)
        rr_ms = rr_intervals_sec * 1000

        ax3.plot(beat_numbers, rr_ms, 'o-',
                 color='#a78bfa', linewidth=1.5, markersize=5)
        ax3.axhline(np.mean(rr_ms), color='#fbbf24',
                    linewidth=1.2, linestyle='--',
                    label=f'Mean RR = {np.mean(rr_ms):.1f} ms')
        ax3.fill_between(beat_numbers, rr_ms,
                         np.mean(rr_ms), alpha=0.15, color='#a78bfa')

    ax3.set_title('RR Interval Tachogram (Beat-to-Beat Variation)',
                  color='white', fontsize=11, fontweight='bold', pad=6)
    ax3.set_xlabel('Beat Number', color='#888', fontsize=9)
    ax3.set_ylabel('RR Interval (ms)', color='#888', fontsize=9)
    ax3.legend(facecolor='#1e1e2a', edgecolor='#2a2a3a',
               labelcolor='white', fontsize=9)
    _style_ax(ax3)

    # ── Panel 4: Metrics Summary ──
    ax4 = fig.add_subplot(gs[3])
    ax4.set_facecolor('#0e0e12')
    ax4.axis('off')

    # Build summary text
    summary = (
        f"  ❤️  Heart Rate : {heart_rate:.1f} BPM   "
        f"({'Normal' if 60 <= heart_rate <= 100 else 'Abnormal — consult a doctor'})\n\n"
        f"  📊  HRV Metrics (higher = healthier autonomic function):\n"
        f"      SDNN  = {hrv_metrics.get('SDNN (ms)', 'N/A')} ms   "
        f"(normal: 50–100 ms)\n"
        f"      RMSSD = {hrv_metrics.get('RMSSD (ms)', 'N/A')} ms   "
        f"(parasympathetic activity)\n"
        f"      pNN50 = {hrv_metrics.get('pNN50 (%)', 'N/A')} %    "
        f"(% beats differing > 50 ms)\n\n"
        f"  🔬  Filter : 4th-order Butterworth bandpass (0.5–40 Hz, zero-phase)\n"
        f"  📡  Peaks  : {len(r_peaks)} R-peaks detected | "
        f"Sample rate: {sample_rate} Hz"
    )

    ax4.text(0.02, 0.5, summary,
             color='#94a3b8', fontsize=10, va='center',
             fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=1.0',
                       facecolor='#1e1e2a',
                       edgecolor='#2a2a3a'))

    fig.suptitle('ECG Signal Analyser — Filter · R-peak Detection · HRV',
                 color='white', fontsize=14, fontweight='bold', y=0.98)

    plt.savefig('ecg_analysis.png', dpi=150,
                bbox_inches='tight', facecolor='#0e0e12')
    print("✅  Plot saved → ecg_analysis.png")
    plt.show()


def _style_ax(ax):
    """Shared dark-theme styling for axes."""
    ax.tick_params(colors='#555')
    ax.set_xlim(ax.get_xlim())
    for spine in ax.spines.values():
        spine.set_color('#2a2a3a')
    ax.grid(True, color='#1e1e2a', linewidth=0.6)


# ──────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────

def main():
    SAMPLE_RATE  = 360    # Hz — MIT-BIH standard
    DURATION_SEC = 10     # seconds of ECG to analyse
    HEART_RATE   = 72     # BPM for synthetic signal

    print("=" * 52)
    print("  ECG Signal Analyser")
    print("  Filter · R-peak Detection · HR · HRV")
    print("=" * 52)

    # ── 1. Generate synthetic ECG ──
    print("\n[1/4] Generating synthetic ECG signal...")
    t, ecg_clean, ecg_noisy = generate_ecg_signal(
        duration_sec=DURATION_SEC,
        heart_rate_bpm=HEART_RATE,
        sample_rate=SAMPLE_RATE,
        noise_level=0.06
    )
    # Add baseline wander artifact (breathing noise)
    ecg_noisy = add_artifact(ecg_noisy, SAMPLE_RATE, artifact_type='baseline')
    print(f"   → {len(t)} samples | {DURATION_SEC}s | {SAMPLE_RATE} Hz")

    # ── 2. Apply bandpass filter ──
    print("\n[2/4] Applying Butterworth bandpass filter (0.5–40 Hz)...")
    ecg_filtered = bandpass_filter(ecg_noisy, SAMPLE_RATE,
                                   low_hz=0.5, high_hz=40.0, order=4)
    print("   → Zero-phase filtering applied (filtfilt)")

    # ── 3. Detect R-peaks ──
    print("\n[3/4] Detecting R-peaks...")
    r_peaks = detect_r_peaks(ecg_filtered, SAMPLE_RATE)
    print(f"   → {len(r_peaks)} R-peaks detected")

    # ── 4. Heart rate & HRV ──
    print("\n[4/4] Computing Heart Rate & HRV metrics...")
    heart_rate, rr_intervals = compute_heart_rate(r_peaks, SAMPLE_RATE)
    hrv_metrics = compute_hrv(rr_intervals)

    print(f"\n{'─'*40}")
    print(f"  Heart Rate : {heart_rate:.1f} BPM")
    for metric, value in hrv_metrics.items():
        print(f"  {metric:<14}: {value}")
    print(f"{'─'*40}")

    # ── 5. Plot ──
    print("\n  Generating plots...")
    plot_ecg_analysis(
        t, ecg_noisy, ecg_filtered,
        r_peaks, rr_intervals,
        heart_rate, hrv_metrics,
        SAMPLE_RATE
    )


if __name__ == "__main__":
    main()
