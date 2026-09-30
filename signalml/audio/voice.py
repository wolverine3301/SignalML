"""Voice-quality measures on a vocal track (Praat via parselmouth).

``harshness`` tells screams and growls from singing. A harsh delivery is loud but
carries no stable pitch: in its loud frames the pitch tracker finds nothing and the
harmonics-to-noise ratio collapses. Measured on a multitrack batch: screamed tracks
showed 41-86% pitchless loud frames with ~0 dB median HNR, clean singing under 20%
pitchless at 13-31 dB. Either signal alone misfires (a few clean tracks read low HNR
yet track pitch fine), so ``likely`` needs both.
"""

from __future__ import annotations

import numpy as np

ANALYSIS_SR = 16000
LOUD_DB = -20.0  # frames within 20 dB of the track's loud level: the sung part
AUDIBLE_DBFS = -60.0
NOISY_HNR_DB = 5.0
PITCHLESS_SHARE = 0.35  # likely harsh: at least this share of loud frames has no pitch...
NOISY_SHARE = 0.60  # ...and at least this share sits under NOISY_HNR_DB


def harshness(y: np.ndarray, sr: int) -> dict | None:
    """``{"pitchless", "noisy", "hnr_median", "likely"}`` over the loud frames of a mono
    signal, or None when the track has too little loud audio to judge (< 1 s)."""
    import librosa
    import parselmouth

    if sr != ANALYSIS_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=ANALYSIS_SR)
    hop = ANALYSIS_SR // 50  # 20 ms, the Praat time step below
    n = len(y) // hop
    if n == 0:
        return None
    rms = np.sqrt(np.mean(y[: n * hop].reshape(n, hop) ** 2, axis=1)) + 1e-10
    dbfs = 20 * np.log10(rms)
    audible = rms[dbfs > AUDIBLE_DBFS]
    if not len(audible):
        return None
    loud = (20 * np.log10(rms / np.percentile(audible, 95)) > LOUD_DB) & (dbfs > AUDIBLE_DBFS)

    snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=ANALYSIS_SR)
    hnr = snd.to_harmonicity_cc(time_step=0.02, minimum_pitch=75).values[0]
    f0 = snd.to_pitch_ac(time_step=0.02, pitch_floor=75, pitch_ceiling=1100) \
        .selected_array["frequency"]
    m = min(len(hnr), len(f0), len(loud))
    mask = loud[:m]
    if mask.sum() < 50:
        return None
    h, f = hnr[:m][mask], f0[:m][mask]
    h = np.where(h < -100, -10.0, h)  # Praat marks unvoiced frames -200: count them as noise
    pitchless, noisy = float(np.mean(f == 0)), float(np.mean(h < NOISY_HNR_DB))
    return {"pitchless": round(pitchless, 3), "noisy": round(noisy, 3),
            "hnr_median": round(float(np.median(h)), 1),
            "likely": pitchless >= PITCHLESS_SHARE and noisy >= NOISY_SHARE}
