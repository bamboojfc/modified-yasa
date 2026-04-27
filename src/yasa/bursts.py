import numpy as np
from scipy.signal import find_peaks
from tqdm import tqdm

def detect_spindle_bursts(
    envelope: np.ndarray,
    sfreq: float,
    peak_prominence: float = 0.5,
    trough_prominence: float = 0.1,
    min_burst_duration: float = -np.inf,
    max_burst_duration: float = np.inf,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect spindle bursts using envelope peak-trough structure.

    Each burst is defined as: left trough → peak → right trough,
    where the envelope forms one natural "hill". Only peaks with
    sufficient prominence are retained.

    Parameters
    ----------
    envelope : np.ndarray
        sigma envelope (e.g., from Hilbert transform of bandpass-filtered signal).
    sfreq : float
        Sampling frequency in Hz.
    peak_prominence : float
        Minimum prominence of an envelope peak, expressed as a fraction
        of the mean envelope amplitude (e.g., 0.5 = peak must rise at
        least 0.5 * mean_envelope above its surrounding troughs).
    trough_prominence : float
        Minimum prominence of envelope troughs, expressed as a fraction
        of the mean envelope amplitude (e.g., 0.1 = trough must drop at
        least 0.1 * mean_envelope below its surrounding peaks).
        Keep this low (e.g., 0.1) to be inclusive.
    min_burst_duration : float
        Minimum trough-to-trough duration in seconds.
    max_burst_duration : float
        Maximum trough-to-trough duration in seconds.

    Returns
    -------
    burst_starts : np.ndarray
        Sample indices of left troughs (burst onsets).
    burst_ends : np.ndarray
        Sample indices of right troughs (burst offsets).
    peak_indices : np.ndarray
        Sample indices of envelope peaks (one per burst).
    """

    # check if envelope is 1D
    if envelope.ndim != 1:
        raise ValueError(f"Envelope must be a 1D array. Got {envelope.shape} instead.")

    # --- find envelope peaks with prominence criterion ---
    # Prominence is scaled to mean envelope so it adapts across nights/subjects
    peak_prominence = peak_prominence * np.mean(envelope)

    # min_distance: a spindle is at least 0.5s, so peaks can't be closer
    min_distance_samples = int(0.5 * sfreq)

    print(f"Detecting peaks with peak_prominence={peak_prominence:.2f} and min_distance={min_distance_samples} samples ({min_burst_duration:.2f} seconds)")
    peak_idx, _ = find_peaks(
        envelope,
        prominence=peak_prominence,
        distance=min_distance_samples,
    )

    if len(peak_idx) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # --- find troughs (peaks on inverted envelope) ---
    print(f"Detecting troughs with trough_prominence={trough_prominence:.2f}")
    trough_prominence = trough_prominence * np.mean(envelope)
    trough_idx, _ = find_peaks(
        -envelope,
        prominence=trough_prominence,
    )
    trough_idx = np.concatenate([[0], trough_idx, [len(envelope) - 1]])
    
    # --- for each peak, find its boundary
    burst_starts, burst_ends, peak_indices = [], [], []
    print(f"Mapping peaks to troughs and applying duration criteria (min={min_burst_duration:.2f}s, max={max_burst_duration:.2f}s)")
    for _, each_peak_idx in enumerate(peak_idx):
        if each_peak_idx == 0 or each_peak_idx == len(envelope) - 1:
            continue
        
        left_troughs  = trough_idx[trough_idx < each_peak_idx]
        right_troughs = trough_idx[trough_idx > each_peak_idx]

        # nearest trough on each side
        left  = left_troughs[-1]  if len(left_troughs)  > 0 else 0
        right = right_troughs[0]  if len(right_troughs) > 0 else len(envelope) - 1

        dur = (right - left) / sfreq
        if min_burst_duration <= dur <= max_burst_duration:
            burst_starts.append(left)
            burst_ends.append(right)
            peak_indices.append(each_peak_idx)

    # print("peak_indices:", peak_indices)
    # print("peak time (s):", [idx / sfreq for idx in peak_indices])
    return np.array(burst_starts), np.array(burst_ends), np.array(peak_indices)

def mapping_so_bursts(
    so_starts: np.ndarray,
    so_ends: np.ndarray,
    burst_starts: np.ndarray,
    burst_ends: np.ndarray,
    burst_peaks_indices: np.ndarray,
    burst_envelope: np.ndarray,
    burst_overlapping_so_criterion: float,
) -> np.ndarray:
    """
    For each SO, find the bursts that occur within it and return the sample index of the burst's envelope peak.
    
    Parameters
    ----------
    so_start: np.ndarray
        Sample indices of SO onsets.
    so_end: np.ndarray
        Sample indices of SO offsets.
    burst_starts: np.ndarray
        Sample indices of burst onsets (left troughs).
    burst_ends: np.ndarray
        Sample indices of burst offsets (right troughs).
    burst_peaks_indices: np.ndarray
        Sample indices of burst peaks (one per burst).
    burst_envelope: np.ndarray
        Envelope values of the bursts (e.g., from Hilbert transform of bandpass-filtered signal).
    burst_overlapping_so_criterion: float
        Minimum fraction of burst duration that must overlap with SO duration for the burst to be considered as occurring within the SO (e.g., 0.5 = at least 50% of burst duration overlaps with SO).
    
    Returns
    -------
    sigma_peaks_indices: np.ndarray
        Sample indices of burst's envelope peaks (one per SO) or None if no bursts in SO or the peak does not pass criteria.
    """
    
    assert len(so_starts) == len(so_ends), "so_start and so_end must have the same length."
    assert len(burst_starts) == len(burst_ends) == len(burst_peaks_indices), "burst_starts, burst_ends, and burst_peaks_indices must have the same length."
    assert burst_overlapping_so_criterion >= 0 and burst_overlapping_so_criterion <= 1, "burst_overlapping_so_criterion must be between 0 and 1."

    sigma_peaks_indices = np.full_like(so_starts, fill_value=np.nan, dtype=np.float64)

    for i, (so_s, so_e) in tqdm(enumerate(zip(so_starts, so_ends))):
        # Find bursts that overlap with the current SO
        overlapping_bursts = []
        for j in range(0, len(burst_starts)):
            b_s = burst_starts[j]
            b_e = burst_ends[j]
        
            burst_duration = b_e - b_s
            overlap_start = max(so_s, b_s)
            overlap_end = min(so_e, b_e)
            overlap_duration = max(0, overlap_end - overlap_start)
            overlap_percentage = overlap_duration / burst_duration if burst_duration > 0 else 0

            if (
                overlap_percentage >= burst_overlapping_so_criterion and
                burst_peaks_indices[j] <= so_e and burst_peaks_indices[j] >= so_s
            ):
                overlapping_bursts.append({
                    'peak_index': burst_peaks_indices[j], 
                    'overlap_percentage': overlap_percentage,
                    'peak_value': burst_envelope[burst_peaks_indices[j]],    
                })

        if len(overlapping_bursts) > 0:
            # If multiple bursts overlap with the SO, select the one with the most overlap
            best_burst_idx = np.argmax([b['overlap_percentage'] for b in overlapping_bursts])
            sigma_peaks_indices[i] = overlapping_bursts[best_burst_idx]['peak_index']

    return np.array(sigma_peaks_indices)
