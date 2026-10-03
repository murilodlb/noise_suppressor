from typing import Dict, Any, Tuple
import numpy as np

try:
    import pesq as pesq_lib
except ImportError:
    pesq_lib = None

try:
    import pystoi
except ImportError:
    pystoi = None


def compute_snr(reference: np.ndarray, estimate: np.ndarray, eps: float = 1e-8) -> float:
    """Computes standard Signal-to-Noise Ratio (SNR) in dB."""
    min_len = min(len(reference), len(estimate))
    ref = reference[:min_len].astype(np.float64)
    est = estimate[:min_len].astype(np.float64)
    noise = ref - est
    ref_energy = np.sum(ref ** 2)
    noise_energy = np.sum(noise ** 2)
    return float(10.0 * np.log10((ref_energy + eps) / (noise_energy + eps)))


def compute_si_sdr(reference: np.ndarray, estimate: np.ndarray, eps: float = 1e-8) -> float:
    """
    Computes Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) in dB.
    Uses zero-mean signals and orthogonal projection.
    """
    min_len = min(len(reference), len(estimate))
    if min_len == 0:
        return float("nan")

    ref = reference[:min_len].astype(np.float64)
    est = estimate[:min_len].astype(np.float64)

    # Zero-mean
    ref = ref - np.mean(ref)
    est = est - np.mean(est)

    ref_energy = np.sum(ref ** 2)
    if ref_energy <= eps:
        return float("-inf")

    # Scaling factor alpha
    alpha = np.dot(est, ref) / (ref_energy + eps)
    s_target = alpha * ref
    e_noise = est - s_target

    target_energy = np.sum(s_target ** 2)
    noise_energy = np.sum(e_noise ** 2)

    if noise_energy <= eps:
        return 100.0  # Perfect reconstruction

    return float(10.0 * np.log10(target_energy / (noise_energy + eps)))


def compute_segmental_snr(
    reference: np.ndarray,
    estimate: np.ndarray,
    frame_length: int = 512,
    hop_length: int = 256,
    eps: float = 1e-8,
    min_snr: float = -10.0,
    max_snr: float = 35.0,
) -> float:
    """
    Computes Segmental SNR (segSNR) in dB over windowed frames,
    clamped to [min_snr, max_snr] to prevent silent frame distortions.
    """
    min_len = min(len(reference), len(estimate))
    if min_len < frame_length:
        return compute_snr(reference, estimate, eps=eps)

    ref = reference[:min_len].astype(np.float64)
    est = estimate[:min_len].astype(np.float64)

    num_frames = (min_len - frame_length) // hop_length + 1
    frame_snrs = []

    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        frame_ref = ref[start:end]
        frame_est = est[start:end]
        frame_noise = frame_ref - frame_est

        ref_power = np.sum(frame_ref ** 2)
        noise_power = np.sum(frame_noise ** 2)

        if ref_power > eps:
            snr_frame = 10.0 * np.log10(ref_power / (noise_power + eps))
            clamped = float(np.clip(snr_frame, min_snr, max_snr))
            frame_snrs.append(clamped)

    if not frame_snrs:
        return 0.0

    return float(np.mean(frame_snrs))


def compute_pesq(reference: np.ndarray, estimate: np.ndarray, sr: int = 16000) -> float:
    """Computes wideband PESQ (ITU-T P.862.2). Returns float('nan') on error or missing dependency."""
    if pesq_lib is None:
        return float("nan")

    min_len = min(len(reference), len(estimate))
    ref = reference[:min_len].astype(np.float32)
    est = estimate[:min_len].astype(np.float32)

    # Check for silence
    if np.max(np.abs(ref)) < 1e-4 or np.max(np.abs(est)) < 1e-4:
        return float("nan")

    try:
        if sr != 16000:
            import librosa
            ref = librosa.resample(ref, orig_sr=sr, target_sr=16000)
            est = librosa.resample(est, orig_sr=sr, target_sr=16000)
            sr = 16000
        score = pesq_lib.pesq(sr, ref, est, "wb")
        return float(score)
    except Exception:
        return float("nan")


def compute_stoi(reference: np.ndarray, estimate: np.ndarray, sr: int = 16000, extended: bool = False) -> float:
    """Computes STOI or extended STOI (ESTOI). Returns float('nan') on error or missing dependency."""
    if pystoi is None:
        return float("nan")

    min_len = min(len(reference), len(estimate))
    ref = reference[:min_len].astype(np.float64)
    est = estimate[:min_len].astype(np.float64)

    if np.max(np.abs(ref)) < 1e-4 or np.max(np.abs(est)) < 1e-4:
        return float("nan")

    try:
        score = pystoi.stoi(ref, est, sr, extended=extended)
        return float(score)
    except Exception:
        return float("nan")


def compute_mask_threshold(lc_db: float, beta: float = 0.5) -> float:
    """
    Computes the IRM decision threshold corresponding to a local criterion (LC) in dB.
    Formula: threshold = (10^(LC/10) / (1 + 10^(LC/10)))^beta
    """
    snr_lin = 10.0 ** (lc_db / 10.0)
    return float((snr_lin / (1.0 + snr_lin)) ** beta)


def compute_mask_classification_metrics(
    predicted_mask: np.ndarray,
    ideal_mask: np.ndarray,
    threshold: float = 0.70710678,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """
    Computes Precision, Recall, F1, Accuracy, and HIT - FA
    by binarizing predicted and ideal masks at the specified threshold.
    """
    y_true = (ideal_mask >= threshold).astype(bool)
    y_pred = (predicted_mask >= threshold).astype(bool)

    tp = float(np.sum(y_true & y_pred))
    fp = float(np.sum((~y_true) & y_pred))
    fn = float(np.sum(y_true & (~y_pred)))
    tn = float(np.sum((~y_true) & (~y_pred)))

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

    hit_rate = recall
    false_alarm_rate = fp / (fp + tn + eps)
    hit_minus_fa = hit_rate - false_alarm_rate

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "hit_rate": hit_rate,
        "false_alarm_rate": false_alarm_rate,
        "hit_minus_false_alarm": hit_minus_fa,
    }


def compute_mask_regression_metrics(
    predicted_mask: np.ndarray,
    ideal_mask: np.ndarray,
) -> Dict[str, float]:
    """Computes MSE and MAE between predicted mask and target continuous IRM."""
    mse = float(np.mean((predicted_mask - ideal_mask) ** 2))
    mae = float(np.mean(np.abs(predicted_mask - ideal_mask)))
    return {"mse": mse, "mae": mae}


def compute_attenuation_metrics(
    clean_mag: np.ndarray,
    noise_mag: np.ndarray,
    mask: np.ndarray,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """
    Computes Speech Attenuation (SA) and Noise Attenuation (NA) in dB.
    - Speech Attenuation: 10 * log10(||S||^2 / ||M * S||^2), lower is better (less speech loss).
    - Noise Attenuation: 10 * log10(||N||^2 / ||M * N||^2), higher is better (more noise killed).
    """
    clean_energy = float(np.sum(np.square(clean_mag)))
    retained_speech_energy = float(np.sum(np.square(mask * clean_mag)))
    speech_attenuation_db = float(10.0 * np.log10((clean_energy + eps) / (retained_speech_energy + eps)))

    noise_energy = float(np.sum(np.square(noise_mag)))
    passed_noise_energy = float(np.sum(np.square(mask * noise_mag)))
    noise_attenuation_db = float(10.0 * np.log10((noise_energy + eps) / (passed_noise_energy + eps)))

    return {
        "speech_attenuation_db": speech_attenuation_db,
        "noise_attenuation_db": noise_attenuation_db,
    }


def compute_signal_metrics(
    reference: np.ndarray,
    estimate: np.ndarray,
    sr: int = 16000,
) -> Dict[str, float]:
    """Computes all time-domain and perceptual signal quality metrics."""
    return {
        "si_sdr": compute_si_sdr(reference, estimate),
        "snr": compute_snr(reference, estimate),
        "segmental_snr": compute_segmental_snr(reference, estimate),
        "pesq": compute_pesq(reference, estimate, sr=sr),
        "stoi": compute_stoi(reference, estimate, sr=sr, extended=False),
        "estoi": compute_stoi(reference, estimate, sr=sr, extended=True),
    }
