import os
import json
import wave
import zipfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(BASE_DIR, exist_ok=True)

for sub in ["audio/clean", "audio/mixed", "audio/separated_ica", "audio/separated_pca", "audio/shazam_library", "src", "results", "notebooks"]:
    os.makedirs(os.path.join(BASE_DIR, sub), exist_ok=True)

# ---------------------------------------------------------
# 1. Generate Realistic Clean Audio Sources (Speech, Music, Ambient)
# ---------------------------------------------------------
FS = 16000  # 16 kHz sampling rate
DURATION = 5.0  # 5 seconds
N_SAMPLES = int(FS * DURATION)
t = np.linspace(0, DURATION, N_SAMPLES, endpoint=False)

def save_wav(filepath, signal, fs=FS):
    sig = np.asarray(signal, dtype=np.float64)
    max_val = np.max(np.abs(sig))
    if max_val > 0:
        sig = sig / max_val * 0.92
    pcm = np.int16(np.clip(sig * 32767, -32768, 32767))
    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(pcm.tobytes())

# Source 1: Speech-like Signal (syllabic amplitude envelope + harmonic formants + glottal pulses)
syllable_freqs = [3.2, 5.1, 2.3]
envelope = np.zeros_like(t)
for sf in syllable_freqs:
    envelope += np.maximum(0, np.sin(2 * np.pi * sf * t + 0.5)) ** 2
# Add pause gaps like natural speech cadences
pause_mask = (np.sin(2 * np.pi * 0.85 * t) > -0.45).astype(float)
# Smooth pause mask
kernel = np.hanning(400) / np.sum(np.hanning(400))
pause_mask = np.convolve(pause_mask, kernel, mode="same")
f0 = 125 + 18 * np.sin(2 * np.pi * 1.4 * t)  # Pitch contour (male vocal range ~125 Hz)
phase = 2 * np.pi * np.cumsum(f0) / FS
glottal = (
    1.0 * np.sin(phase)
    + 0.65 * np.sin(2 * phase + 0.2)
    + 0.45 * np.sin(3 * phase + 0.4)
    + 0.30 * np.sin(5 * phase + 0.1)  # Formant F1 (~625 Hz)
    + 0.20 * np.sin(9 * phase + 0.7)  # Formant F2 (~1125 Hz)
)
s1_speech = envelope * pause_mask * glottal
s1_speech = (s1_speech - np.mean(s1_speech)) / np.std(s1_speech)

# Source 2: Music Signal (Arpeggiated synth melody + chord progression with pluck decay)
notes_hz = [261.63, 329.63, 392.00, 523.25, 392.00, 329.63, 293.66, 349.23, 440.00, 587.33]  # C major -> D minor
s2_music = np.zeros_like(t)
step_samples = N_SAMPLES // len(notes_hz)
for idx, freq in enumerate(notes_hz):
    start = idx * step_samples
    end = start + step_samples if idx < len(notes_hz) - 1 else N_SAMPLES
    t_local = np.arange(end - start) / FS
    pluck_env = np.exp(-3.8 * t_local) * (1 - np.exp(-80 * t_local))
    # Sawtooth-like rich harmonic synth tone
    tone = (
        np.sin(2 * np.pi * freq * t_local)
        + 0.5 * np.sin(2 * np.pi * 2 * freq * t_local)
        + 0.25 * np.sin(2 * np.pi * 3 * freq * t_local)
        + 0.3 * np.sin(2 * np.pi * (freq * 0.5) * t_local)  # Sub-bass octave
    )
    s2_music[start:end] = pluck_env * tone
s2_music = (s2_music - np.mean(s2_music)) / np.std(s2_music)

# Source 3: Ambient Signal (Super-Gaussian impulsive rain/crackle texture + wind swell)
# Laplacian/impulsive distribution + slow FM wind shimmer (distinctly non-Gaussian!)
u = np.random.uniform(-0.5, 0.5, size=N_SAMPLES)
laplacian_rain = -np.sign(u) * np.log(1 - 2 * np.abs(u))
# Band-shape the rain crackle + wind swell
wind_swell = (0.4 + 0.6 * np.sin(2 * np.pi * 0.45 * t) ** 2) * np.sign(np.sin(2 * np.pi * 430 * t + 3.0 * np.sin(2 * np.pi * 2.5 * t)))
s3_ambient = 0.65 * laplacian_rain * (0.5 + 0.5 * np.cos(2 * np.pi * 1.1 * t)) + 0.35 * wind_swell
s3_ambient = (s3_ambient - np.mean(s3_ambient)) / np.std(s3_ambient)

S_true = np.vstack([s1_speech, s2_music, s3_ambient])  # Shape: (3, N_SAMPLES)
source_names = ["source_1_speech", "source_2_music", "source_3_ambient"]
for i, name in enumerate(source_names):
    save_wav(os.path.join(BASE_DIR, "audio/clean", f"{name}.wav"), S_true[i])

# ---------------------------------------------------------
# 2. Mix Signals with Custom Mixing Matrix A
# ---------------------------------------------------------
A_mix = np.array([
    [0.82, 0.45, 0.31],
    [0.35, 0.88, 0.42],
    [0.29, 0.38, 0.85]
], dtype=np.float64)

X_mixed = A_mix @ S_true  # Shape: (3, N_SAMPLES)
for i in range(3):
    save_wav(os.path.join(BASE_DIR, "audio/mixed", f"mixed_mic_{i+1}.wav"), X_mixed[i])

# ---------------------------------------------------------
# 3. Core Algorithms From Scratch: Centering, PCA Whitening, FastICA
# ---------------------------------------------------------
def center_signals(X):
    """Subtract sample mean from each channel (row)."""
    mean = np.mean(X, axis=1, keepdims=True)
    return X - mean, mean

def pca_whiten(X_centered, eps=1e-12):
    """
    Implement PCA whitening from scratch using sample covariance
    and eigendecomposition:
        Cov(X) = (1 / T) * X @ X.T = E @ D @ E.T
        W_white = D^(-1/2) @ E.T
        Z = W_white @ X_centered  => Cov(Z) = I
    Returns Z_whitened, W_white, D_dewhite, eigenvalues, eigenvectors, X_pca
    """
    n_samples = X_centered.shape[1]
    cov = (X_centered @ X_centered.T) / float(n_samples)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Sort descending by eigenvalue magnitude
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    inv_sqrt_eig = np.diag(1.0 / np.sqrt(np.maximum(eigenvalues, eps)))
    sqrt_eig = np.diag(np.sqrt(np.maximum(eigenvalues, eps)))

    W_white = inv_sqrt_eig @ eigenvectors.T
    D_dewhite = eigenvectors @ sqrt_eig
    Z_white = W_white @ X_centered

    # Pure PCA decorrelated projections (standardized to unit variance)
    X_pca = Z_white.copy()
    return Z_white, W_white, D_dewhite, eigenvalues, eigenvectors, X_pca

def _sym_decorrelate(W):
    """
    Symmetric orthogonalization: W <- (W @ W.T)^(-1/2) @ W
    Implemented via eigendecomposition of W @ W.T.
    """
    K = W @ W.T
    eigvals, eigvecs = np.linalg.eigh(K)
    inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(eigvals, 1e-12)))
    return (eigvecs @ inv_sqrt @ eigvecs.T) @ W

def fast_ica_from_scratch(Z_white, fun="logcosh", alpha=1.0, max_iter=500, tol=1e-7, seed=42):
    """
    Symmetric FastICA implemented from scratch on whitened data Z_white (N x T).
    Maximizes non-Gaussianity (negentropy approximation) via fixed-point iteration.
    """
    rng = np.random.default_rng(seed)
    n_comp, n_samples = Z_white.shape
    W = rng.normal(size=(n_comp, n_comp))
    W = _sym_decorrelate(W)

    convergence_history = []
    for it in range(1, max_iter + 1):
        WX = W @ Z_white  # (n_comp, n_samples)
        if fun == "logcosh":
            g = np.tanh(alpha * WX)
            g_prime = alpha * (1.0 - g ** 2)
        elif fun == "exp":
            exp_term = np.exp(-(WX ** 2) / 2.0)
            g = WX * exp_term
            g_prime = (1.0 - WX ** 2) * exp_term
        elif fun == "cube":
            g = WX ** 3
            g_prime = 3.0 * (WX ** 2)
        else:
            raise ValueError(f"Unsupported contrast function: {fun}")

        # Fixed-point update across all components simultaneously
        W_new = (g @ Z_white.T) / float(n_samples) - np.mean(g_prime, axis=1, keepdims=True) * W
        W_new = _sym_decorrelate(W_new)

        # Check convergence: max(1 - |dot(w_i_new, w_i_old)|)
        lim = np.max(np.abs(np.abs(np.sum(W_new * W, axis=1)) - 1.0))
        convergence_history.append(float(lim))
        W = W_new
        if lim < tol:
            break

    S_est = W @ Z_white
    return S_est, W, convergence_history

# ---------------------------------------------------------
# 4. Evaluation Metrics (Accounting for Permutation & Scale/Sign Ambiguity)
# ---------------------------------------------------------
from itertools import permutations

def align_and_evaluate(S_true, S_est):
    """
    Finds the optimal permutation and sign/scale alignment between S_est and S_true.
    Computes:
      - Abs Pearson Correlation |r|
      - Signal-to-Distortion Ratio (SDR in dB)
      - Signal-to-Interference Ratio (SIR in dB)
    """
    n_comp = S_true.shape[0]
    # Normalize rows to zero mean, unit variance
    S_t = (S_true - np.mean(S_true, axis=1, keepdims=True)) / np.std(S_true, axis=1, keepdims=True)
    S_e = (S_est - np.mean(S_est, axis=1, keepdims=True)) / np.std(S_est, axis=1, keepdims=True)

    corr_matrix = (S_t @ S_e.T) / float(S_true.shape[1])

    # Search over all N! permutations to maximize sum of |corr|
    best_perm = None
    best_score = -1.0
    for perm in permutations(range(n_comp)):
        score = sum(abs(corr_matrix[i, perm[i]]) for i in range(n_comp))
        if score > best_score:
            best_score = score
            best_perm = perm

    S_aligned = np.zeros_like(S_t)
    metrics_per_source = []
    for i in range(n_comp):
        j = best_perm[i]
        r = corr_matrix[i, j]
        sign = 1.0 if r >= 0 else -1.0
        s_hat = sign * S_e[j]
        # Least-squares optimal scaling factor alpha = <s_true, s_hat> / <s_hat, s_hat>
        scale = np.dot(S_t[i], s_hat) / (np.dot(s_hat, s_hat) + 1e-12)
        s_target = scale * s_hat
        e_total = S_t[i] - s_target
        sdr_db = 10.0 * np.log10((np.sum(s_target ** 2) + 1e-12) / (np.sum(e_total ** 2) + 1e-12))

        # Excess kurtosis
        kurt_true = float(np.mean(S_t[i] ** 4) - 3.0)
        kurt_est = float(np.mean(s_hat ** 4) - 3.0)

        S_aligned[i] = s_hat
        metrics_per_source.append({
            "source_index": i + 1,
            "source_name": source_names[i],
            "matched_component": int(j),
            "sign_flip": int(sign),
            "abs_correlation": float(abs(r)),
            "sdr_db": float(sdr_db),
            "kurtosis_true": kurt_true,
            "kurtosis_recovered": kurt_est
        })

    return S_aligned, metrics_per_source, corr_matrix, best_perm

# Run pipeline on X_mixed
X_centered, X_mean = center_signals(X_mixed)
Z_white, W_white, D_dewhite, eigvals, eigvecs, S_pca_raw = pca_whiten(X_centered)
S_ica_raw, W_ica, conv_hist = fast_ica_from_scratch(Z_white, fun="logcosh", max_iter=500, tol=1e-8)

S_ica_aligned, ica_metrics, corr_mat_ica, perm_ica = align_and_evaluate(S_true, S_ica_raw)
S_pca_aligned, pca_metrics, corr_mat_pca, perm_pca = align_and_evaluate(S_true, S_pca_raw)
_, mixed_metrics, _, _ = align_and_evaluate(S_true, X_mixed)

for i, name in enumerate(source_names):
    save_wav(os.path.join(BASE_DIR, "audio/separated_ica", f"recovered_{i+1}_{name.split('_')[-1]}.wav"), S_ica_aligned[i])
    save_wav(os.path.join(BASE_DIR, "audio/separated_pca", f"pca_{i+1}_{name.split('_')[-1]}.wav"), S_pca_aligned[i])

# ---------------------------------------------------------
# 5. Bonus Experiment: SNR Robustness Sweep (0 dB to 40 dB)
# ---------------------------------------------------------
snr_levels_db = [0, 5, 10, 15, 20, 25, 30, 35, 40]
snr_results = []
sig_power = np.mean(X_mixed ** 2)
for snr_db in snr_levels_db:
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise = np.random.normal(0, np.sqrt(noise_power), size=X_mixed.shape)
    X_noisy = X_mixed + noise
    Xc_n, _ = center_signals(X_noisy)
    Zw_n, _, _, _, _, Spca_n = pca_whiten(Xc_n)
    Sica_n, _, _ = fast_ica_from_scratch(Zw_n, fun="logcosh", seed=42)
    _, m_ica, _, _ = align_and_evaluate(S_true, Sica_n)
    _, m_pca, _, _ = align_and_evaluate(S_true, Spca_n)
    snr_results.append({
        "snr_db": snr_db,
        "ica_mean_corr": float(np.mean([m["abs_correlation"] for m in m_ica])),
        "ica_mean_sdr_db": float(np.mean([m["sdr_db"] for m in m_ica])),
        "pca_mean_corr": float(np.mean([m["abs_correlation"] for m in m_pca])),
        "pca_mean_sdr_db": float(np.mean([m["sdr_db"] for m in m_pca]))
    })

# ---------------------------------------------------------
# 6. Stretch Goal: Mini-Shazam Audio Fingerprinting on 8-Song Library
# ---------------------------------------------------------
def compute_stft_mag(sig, n_fft=512, hop=256):
    window = np.hanning(n_fft)
    n_frames = (len(sig) - n_fft) // hop + 1
    spec = np.empty((n_fft // 2 + 1, n_frames), dtype=np.float64)
    for f_idx in range(n_frames):
        frame = sig[f_idx * hop : f_idx * hop + n_fft] * window
        spec[:, f_idx] = np.abs(np.fft.rfft(frame))
    return spec

def extract_constellation_hashes(sig, n_fft=512, hop=256, fan_out=5):
    """Extract local STFT spectral peaks and pair them into (f1, f2, dt) hashes."""
    spec = compute_stft_mag(sig, n_fft=n_fft, hop=hop)
    # Divide frequency bins into 6 logarithmic/sub-bands and pick top peak per frame if above threshold
    bands = [(2, 12), (12, 28), (28, 55), (55, 100), (100, 170), (170, 250)]
    peaks = []  # (time_idx, freq_bin)
    med_threshold = np.percentile(spec, 82)
    for t_idx in range(0, spec.shape[1], 2):  # Step every 2 frames for compact constellation
        for b_low, b_high in bands:
            slice_band = spec[b_low:b_high, t_idx]
            max_local = np.argmax(slice_band)
            val = slice_band[max_local]
            if val > med_threshold:
                peaks.append((t_idx, b_low + int(max_local)))
    hashes = {}
    for i in range(len(peaks)):
        t1, f1 = peaks[i]
        for j in range(1, min(fan_out + 1, len(peaks) - i)):
            t2, f2 = peaks[i + j]
            dt = t2 - t1
            if 1 <= dt <= 32:
                h = (f1, f2, dt)
                hashes.setdefault(h, []).append(t1)
    return hashes, len(peaks)

# Build a 6-song synthetic library + our 3 clean sources (total 8 library tracks)
library_tracks = {
    "Track_01_Speech_Podcast": s1_speech,
    "Track_02_C_Major_Synth_Arp": s2_music,
    "Track_03_Ambient_Rainstorm": s3_ambient,
}
extra_melodies = [
    ("Track_04_Cyberpunk_Bassline", [110.0, 130.81, 146.83, 164.81, 110.0, 196.00, 174.61, 146.83]),
    ("Track_05_Pentatonic_Chime", [523.25, 587.33, 659.25, 783.99, 880.00, 783.99, 659.25, 523.25]),
    ("Track_06_Jazz_Walking_Triad", [196.00, 246.94, 293.66, 349.23, 329.63, 261.63, 220.00, 196.00]),
    ("Track_07_LoFi_Chord_Loop", [220.00, 277.18, 329.63, 440.00, 246.94, 311.13, 369.99, 493.88]),
    ("Track_08_Arcade_8Bit_Lead", [659.25, 493.88, 523.25, 587.33, 523.25, 493.88, 440.00, 440.00]),
]
for t_name, freqs in extra_melodies:
    sig_m = np.zeros_like(t)
    step = N_SAMPLES // len(freqs)
    for k, f_hz in enumerate(freqs):
        st, en = k * step, (k + 1) * step if k < len(freqs) - 1 else N_SAMPLES
        tl = np.arange(en - st) / FS
        sig_m[st:en] = np.exp(-3.0 * tl) * (np.sin(2 * np.pi * f_hz * tl) + 0.4 * np.sin(2 * np.pi * 2 * f_hz * tl))
    sig_m = (sig_m - np.mean(sig_m)) / np.std(sig_m)
    library_tracks[t_name] = sig_m
    save_wav(os.path.join(BASE_DIR, "audio/shazam_library", f"{t_name}.wav"), sig_m)

# Index all 8 tracks
db_index = {}
for name, sig in library_tracks.items():
    db_index[name], _ = extract_constellation_hashes(sig)

def query_mini_shazam(query_sig):
    q_hashes, _ = extract_constellation_hashes(query_sig)
    scores = {}
    for track_name, t_hashes in db_index.items():
        offset_counts = {}
        for h, q_times in q_hashes.items():
            if h in t_hashes:
                for qt in q_times:
                    for tt in t_hashes[h]:
                        diff = tt - qt
                        offset_counts[diff] = offset_counts.get(diff, 0) + 1
        scores[track_name] = max(offset_counts.values()) if offset_counts else 0
    return scores

shazam_ica_music_scores = query_mini_shazam(S_ica_aligned[1])
shazam_ica_speech_scores = query_mini_shazam(S_ica_aligned[0])

# ---------------------------------------------------------
# 7. Generate 4 High-Resolution Visual Plots
# ---------------------------------------------------------
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#334155"

# Plot 1: Time-Domain Waveforms (Original vs Mixed vs PCA vs ICA)
fig, axes = plt.subplots(3, 4, figsize=(16, 8.5), sharex=True)
slice_idx = slice(0, int(1.5 * FS))  # First 1.5s for clear visual detail
t_slice = t[slice_idx]
titles_col = ["Clean Original Sources", "Mixed Microphone Signals", "PCA Whitened (Uncorrelated)", "FastICA Recovered (Independent)"]
colors = ["#2563EB", "#D97706", "#7C3AED", "#059669"]
row_labels = ["Channel 1\n(Speech)", "Channel 2\n(Music)", "Channel 3\n(Ambient)"]

for r in range(3):
    datasets = [S_true[r, slice_idx], X_mixed[r, slice_idx], S_pca_aligned[r, slice_idx], S_ica_aligned[r, slice_idx]]
    for c in range(4):
        ax = axes[r, c]
        ax.plot(t_slice, datasets[c], color=colors[c], lw=0.85, alpha=0.9)
        ax.set_ylim(-3.6, 3.6)
        ax.grid(True, linestyle="--", alpha=0.3)
        if r == 0:
            ax.set_title(titles_col[c], fontsize=11, fontweight="bold", pad=8)
        if c == 0:
            ax.set_ylabel(row_labels[r], fontsize=10, fontweight="bold")
        if r == 2:
            ax.set_xlabel("Time (s)", fontsize=9.5)
        if c == 3:
            corr_v = ica_metrics[r]["abs_correlation"]
            sdr_v = ica_metrics[r]["sdr_db"]
            ax.text(0.97, 0.88, f"|r| = {corr_v:.4f} | SDR = {sdr_v:.1f} dB",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=8.5, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="#ECFDF5", edgecolor="#059669", alpha=0.9))
        elif c == 2:
            corr_p = pca_metrics[r]["abs_correlation"]
            sdr_p = pca_metrics[r]["sdr_db"]
            ax.text(0.97, 0.88, f"|r| = {corr_p:.4f} | SDR = {sdr_p:.1f} dB",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=8.5,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="#F5F3FF", edgecolor="#7C3AED", alpha=0.9))

fig.suptitle("Blind Audio Source Separation: Clean vs. Mixed vs. PCA vs. From-Scratch FastICA", fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(BASE_DIR, "results/waveforms_comparison.png"), dpi=200)
plt.close(fig)

# Plot 2: Joint Distribution Scatter & Correlation Matrices (Why ICA Beats PCA)
fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
sub_idx = np.random.choice(N_SAMPLES, size=3000, replace=False)
scatter_sets = [
    (S_true[0, sub_idx], S_true[1, sub_idx], "1. True Sources (Independent)", "#2563EB"),
    (X_mixed[0, sub_idx], X_mixed[1, sub_idx], "2. Mixed Signals (Correlated)", "#D97706"),
    (S_pca_aligned[0, sub_idx], S_pca_aligned[1, sub_idx], "3. PCA Whitened (Rotated Lattice)", "#7C3AED"),
    (S_ica_aligned[0, sub_idx], S_ica_aligned[1, sub_idx], "4. FastICA Output (Axes Restored)", "#059669"),
]
for idx, (x_pt, y_pt, title_s, col) in enumerate(scatter_sets):
    ax = axes[idx]
    ax.scatter(x_pt, y_pt, s=4, alpha=0.35, color=col)
    ax.set_title(title_s, fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Signal 1 Amplitude", fontsize=9)
    ax.set_ylabel("Signal 2 Amplitude", fontsize=9)
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.grid(True, linestyle="--", alpha=0.3)

fig.suptitle("Geometric Intuition of ICA vs. PCA in Phase Space (Speech vs. Music)", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "results/pca_vs_ica_geometry.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

# Plot 3: SNR Robustness Sweep & FastICA Convergence
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
snrs = [r["snr_db"] for r in snr_results]
ica_sdrs = [r["ica_mean_sdr_db"] for r in snr_results]
pca_sdrs = [r["pca_mean_sdr_db"] for r in snr_results]
ica_corrs = [r["ica_mean_corr"] for r in snr_results]
pca_corrs = [r["pca_mean_corr"] for r in snr_results]

ax1.plot(snrs, ica_sdrs, "o-", color="#059669", lw=2.2, label="FastICA (From Scratch) — Mean SDR (dB)")
ax1.plot(snrs, pca_sdrs, "s--", color="#7C3AED", lw=2.0, label="PCA Baseline — Mean SDR (dB)")
ax1.set_title("Separation Quality (SDR) Across Sensor SNR Levels", fontsize=11.5, fontweight="bold")
ax1.set_xlabel("Input Signal-to-Noise Ratio (dB)", fontsize=10)
ax1.set_ylabel("Mean Recovered SDR (dB)", fontsize=10)
ax1.grid(True, linestyle="--", alpha=0.35)
ax1.legend(loc="upper left", fontsize=9)

ax2.semilogy(range(1, len(conv_hist) + 1), conv_hist, "o-", color="#2563EB", lw=2.2)
ax2.axhline(1e-8, color="#DC2626", linestyle=":", label="Convergence Tolerance (1e-8)")
ax2.set_title(f"FastICA Fixed-Point Convergence ({len(conv_hist)} Iterations)", fontsize=11.5, fontweight="bold")
ax2.set_xlabel("Iteration Number", fontsize=10)
ax2.set_ylabel("Max Weight Change  1 - |w_new · w_old| (log scale)", fontsize=10)
ax2.grid(True, linestyle="--", alpha=0.35)
ax2.legend(loc="upper right", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "results/snr_and_convergence.png"), dpi=200)
plt.close(fig)

# Plot 4: Stretch Goal — Mini-Shazam Constellation Identification After ICA Separation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.6))
tracks_list = list(shazam_ica_music_scores.keys())
short_labels = [t.replace("Track_", "T").replace("_", " ") for t in tracks_list]
music_votes = [shazam_ica_music_scores[k] for k in tracks_list]
speech_votes = [shazam_ica_speech_scores[k] for k in tracks_list]

bars1 = ax1.barh(short_labels, music_votes, color=["#059669" if "C Major" in l else "#94A3B8" for l in short_labels])
ax1.set_title("Mini-Shazam Query: ICA Recovered Channel 2 (Music)", fontsize=11, fontweight="bold")
ax1.set_xlabel("Matching Constellation Hash Votes (Time-Coherent)", fontsize=9.5)
ax1.invert_yaxis()
ax1.grid(True, axis="x", linestyle="--", alpha=0.35)

bars2 = ax2.barh(short_labels, speech_votes, color=["#2563EB" if "Speech" in l else "#94A3B8" for l in short_labels])
ax2.set_title("Mini-Shazam Query: ICA Recovered Channel 1 (Speech)", fontsize=11, fontweight="bold")
ax2.set_xlabel("Matching Constellation Hash Votes (Time-Coherent)", fontsize=9.5)
ax2.invert_yaxis()
ax2.grid(True, axis="x", linestyle="--", alpha=0.35)

plt.suptitle("Stretch Goal: Mini-Shazam Song Identification on ICA-Separated Audio Streams", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "results/mini_shazam_identification.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

# Save summary JSON
summary_data = {
    "mixing_matrix": A_mix.tolist(),
    "eigenvalues_cov": eigvals.tolist(),
    "fastica_iterations": len(conv_hist),
    "mixed_baseline_metrics": mixed_metrics,
    "pca_metrics": pca_metrics,
    "ica_metrics": ica_metrics,
    "snr_sweep": snr_results,
    "mini_shazam_music_query": shazam_ica_music_scores,
    "mini_shazam_speech_query": shazam_ica_speech_scores,
}
with open(os.path.join(BASE_DIR, "results/metrics_summary.json"), "w") as f:
    json.dump(summary_data, f, indent=2)

print("ICA Mean |r|:", np.mean([m["abs_correlation"] for m in ica_metrics]))
print("ICA Mean SDR (dB):", np.mean([m["sdr_db"] for m in ica_metrics]))
print("PCA Mean |r|:", np.mean([m["abs_correlation"] for m in pca_metrics]))
print("PCA Mean SDR (dB):", np.mean([m["sdr_db"] for m in pca_metrics]))
print("FastICA converged in:", len(conv_hist), "iterations")
print("Mini-Shazam top music match:", max(shazam_ica_music_scores.items(), key=lambda x: x[1]))
print("Mini-Shazam top speech match:", max(shazam_ica_speech_scores.items(), key=lambda x: x[1]))
