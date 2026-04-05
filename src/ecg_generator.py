# ecg_generator.py
# Synthetic ECG Signal Generator
# Generates a realistic ECG waveform using Gaussian curves
# modelling P, Q, R, S, T waves — based on published ECG models
# Author: Vibha (Nova) | NMIT Bengaluru | ECE

import numpy as np


# ──────────────────────────────────────────────────────────
# WHY SYNTHETIC ECG?
# A real ECG waveform is made of 5 waves: P, Q, R, S, T
# Each wave can be approximated by a Gaussian curve with
# known amplitude, width, and position relative to the
# R-peak (the tallest spike — the "heartbeat" you see)
#
# This approach is from McSharry et al. (2003), widely used
# in ECG simulation research.
# ──────────────────────────────────────────────────────────

# Each tuple: (amplitude, centre_offset_from_R, width_in_radians)
# P wave → small positive bump before R
# Q wave → small negative dip before R
# R wave → tall positive spike (the main peak)
# S wave → small negative dip after R
# T wave → medium positive bump after R

ECG_WAVE_PARAMS = [
    # (amplitude,  theta_i,      b_i)
    (  0.20,      -1.05,        0.09),   # P wave
    ( -0.25,      -0.25,        0.05),   # Q wave
    (  1.60,       0.00,        0.10),   # R wave  ← the main spike
    ( -0.40,       0.30,        0.06),   # S wave
    (  0.40,       1.20,        0.14),   # T wave
]


def generate_ecg_beat(duration_rad=2 * np.pi, num_points=500):
    """
    Generate one ECG beat (one cardiac cycle) as a sum of Gaussians.

    Each wave component:
      z_i(t) = A_i × exp( -(θ - θ_i)² / (2 × b_i²) )

    Returns time array and one beat of ECG signal.
    """
    theta = np.linspace(-np.pi, np.pi, num_points)
    beat = np.zeros(num_points)

    for amplitude, theta_i, b_i in ECG_WAVE_PARAMS:
        beat += amplitude * np.exp(
            -((theta - theta_i) ** 2) / (2 * b_i ** 2)
        )

    return theta, beat


def generate_ecg_signal(
    duration_sec=10,
    heart_rate_bpm=72,
    sample_rate=360,
    noise_level=0.05,
    seed=42
):
    """
    Generate a full multi-beat ECG signal.

    Parameters
    ----------
    duration_sec  : total signal duration in seconds
    heart_rate_bpm: beats per minute (controls R-peak spacing)
    sample_rate   : samples per second (Hz) — 360 Hz = MIT-BIH standard
    noise_level   : standard deviation of added Gaussian noise
    seed          : for reproducibility

    Returns
    -------
    t      : time axis (numpy array)
    ecg    : clean ECG signal
    ecg_noisy : ECG + Gaussian noise (what a real sensor gives you)
    """
    np.random.seed(seed)

    total_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, total_samples)

    # Samples per beat based on heart rate
    samples_per_beat = int(sample_rate * 60 / heart_rate_bpm)

    # Generate one template beat at the right resolution
    _, beat_template = generate_ecg_beat(num_points=samples_per_beat)

    # Tile the beat to fill the full duration
    num_beats = int(np.ceil(total_samples / samples_per_beat))
    ecg_tiled = np.tile(beat_template, num_beats)
    ecg = ecg_tiled[:total_samples]

    # Add realistic Gaussian noise (simulates electrode + muscle noise)
    noise = np.random.normal(0, noise_level, total_samples)
    ecg_noisy = ecg + noise

    return t, ecg, ecg_noisy


def add_artifact(ecg_noisy, sample_rate, artifact_type='baseline'):
    """
    Add realistic ECG artifacts for demonstration.

    baseline  : slow baseline wander (breathing artifact ~0.15 Hz)
    powerline : 50 Hz powerline interference (common in India!)
    """
    t = np.linspace(0, len(ecg_noisy) / sample_rate, len(ecg_noisy))

    if artifact_type == 'baseline':
        # Breathing causes ~0.15–0.3 Hz baseline wander
        wander = 0.15 * np.sin(2 * np.pi * 0.2 * t)
        return ecg_noisy + wander

    elif artifact_type == 'powerline':
        # 50 Hz mains interference (India uses 50 Hz)
        interference = 0.08 * np.sin(2 * np.pi * 50 * t)
        return ecg_noisy + interference

    return ecg_noisy


if __name__ == "__main__":
    # Quick test
    t, ecg_clean, ecg_noisy = generate_ecg_signal(duration_sec=5)
    print(f"✅ ECG generated: {len(t)} samples over 5 seconds")
    print(f"   Sample rate : 360 Hz (MIT-BIH standard)")
    print(f"   Heart rate  : 72 BPM")
    print(f"   Signal range: {ecg_noisy.min():.3f} to {ecg_noisy.max():.3f} mV")
