import numpy as np
from scipy.signal import find_peaks
from tqdm import tqdm
import matplotlib.pyplot as plt

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
    for _, each_peak_idx in tqdm(enumerate(peak_idx)):
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
            
            # If multiple bursts overlap with the SO, select the one with the highest peak value
            best_burst_idx = np.argmax([b['peak_value'] for b in overlapping_bursts])
            max_peak_value = overlapping_bursts[best_burst_idx]['peak_value']
            if len([b for b in overlapping_bursts if b['peak_value'] == max_peak_value]) > 1:
                txt = f"Warning: Multiple bursts have the same peak value of {max_peak_value:.2f} for SO starting at {so_s}. Selecting the one that is closest to the middle."
                print()
                with open('mapping_so_bursts_warning.txt', 'a') as f:
                    f.write(txt) # TODO remove
                so_midpoint = (so_s + so_e) / 2
                best_burst_idx = np.argmin([abs(b['peak_index'] - so_midpoint) for b in overlapping_bursts if b['peak_value'] == max_peak_value])
            sigma_peaks_indices[i] = overlapping_bursts[best_burst_idx]['peak_index']
            
            """
            # If multiple bursts overlap with the SO, select the one with the highest overlap
            best_burst_idx = np.argmax([b['overlap_percentage'] for b in overlapping_bursts])
            sigma_peaks_indices[i] = overlapping_bursts[best_burst_idx]['peak_index']
            """

    return np.array(sigma_peaks_indices)

def get_all_candidate_bursts_by_mask(
        burst_starts: np.ndarray,
        burst_ends: np.ndarray,
        burst_peaks_indices: np.ndarray,
        idx_included: np.ndarray,
    ):
    """
    Get all candidate bursts by sleep stage.
    
    Parameters
    ----------
    burst_starts: np.ndarray
        Sample indices of burst onsets (left troughs).
    burst_ends: np.ndarray
        Sample indices of burst offsets (right troughs).
    burst_peaks_indices: np.ndarray
        Sample indices of burst peaks (one per burst).
    idx_included: np.ndarray
        Sample indices of the included samples.
    """
    assert burst_starts.shape == burst_ends.shape == burst_peaks_indices.shape, "burst_starts, burst_ends, and burst_peaks_indices must have the same shape."

    # Vectorized check: for each burst, check if ALL samples in [start, end] are included
    keep = np.array([
        idx_included[burst_starts[i]:burst_ends[i] + 1].all()
        for i in range(len(burst_starts))
    ])

    print(f"Filtered bursts: {keep.sum()} out of {len(burst_starts)}")
    return burst_starts[keep], burst_ends[keep], burst_peaks_indices[keep]

def calculate_bursts_median(candidate_sigma_burst_peak_indices: np.ndarray, sigma_envelop: np.ndarray):
    """
    Calculate the median of the candidate sigma bursts' envelope values.
    
    Parameters
    ----------
    candidate_sigma_burst_peak_indices: np.ndarray
        Sample indices of candidate sigma burst peaks.
    sigma_envelop: np.ndarray
        Envelope values of the sigma band (e.g., from Hilbert transform of bandpass-filtered signal).
    
    Returns
    -------
    median_burst_value: float
        Median value of the candidate sigma bursts' envelope values.
    """
    
    if len(candidate_sigma_burst_peak_indices) == 0:
        return np.nan
    
    burst_values = sigma_envelop[candidate_sigma_burst_peak_indices]
    median_burst_value = np.median(burst_values)
    
    return median_burst_value

def calculate_bursts_mad(candidate_sigma_burst_peak_indices: np.ndarray, sigma_envelop: np.ndarray):
    """
    Calculate the median absolute deviation (MAD) of the candidate sigma bursts' envelope values.
    
    Parameters
    ----------
    candidate_sigma_burst_peak_indices: np.ndarray
        Sample indices of candidate sigma burst peaks.
    sigma_envelop: np.ndarray
        Envelope values of the sigma band (e.g., from Hilbert transform of bandpass-filtered signal).
    
    Returns
    -------
    mad_burst_value: float
        Median absolute deviation (MAD) value of the candidate sigma bursts' envelope values.
    """
    
    if len(candidate_sigma_burst_peak_indices) == 0:
        return np.nan
    
    burst_values = sigma_envelop[candidate_sigma_burst_peak_indices]
    mad_burst_value = np.median(np.abs(burst_values - np.median(burst_values)))
    
    return mad_burst_value

def filter_bursts(
        sigma_envelop: np.ndarray,
        burst_starts: np.ndarray,
        burst_ends: np.ndarray,
        burst_peaks_indices: np.ndarray,
        median_sigma_burst_peak: float,
        mad_sigma_burst_peak: float,
        burst_duration_range: tuple[float, float],
        z_peak_threshold: float,
        min_bursts_distance: int,
        fs: float,
    ):
    """
    Filter bursts based on median and MAD of the candidate sigma bursts' envelope values, merge accpeted bursts with minimum_distance gap, and filter again by burst duration.
    
    Parameters
    ----------
    sigma_envelop: np.ndarray
        Envelope values of the sigma band (e.g., from Hilbert transform of bandpass-filtered signal).
    burst_starts: np.ndarray
        Sample indices of burst onsets (left troughs).
    burst_ends: np.ndarray
        Sample indices of burst offsets (right troughs).
    burst_peaks_indices: np.ndarray
        Sample indices of burst peaks (one per burst).
    median_sigma_burst_peak: float
        Median value of the candidate sigma bursts' envelope values.
    mad_sigma_burst_peak: float
        Median absolute deviation (MAD) value of the candidate sigma bursts' envelope values.
    burst_duration_range: tuple[float, float]
        Minimum and maximum burst duration in seconds.
    z_peak_threshold: float
        Z-score threshold for filtering bursts based on their peak envelope values.
    min_bursts_distance: int (ms)
        selected bursts with less than min_bursts_distance ms between them will be merged into one burst.
    fs: float
        Sampling frequency in Hz.
    """
    
    # Calculate z-score threshold
    peaks_z_score = (sigma_envelop[burst_peaks_indices] - median_sigma_burst_peak) / (1.4826 * mad_sigma_burst_peak) # 1.4826 is the standard scaling factor for data that follows a normal distribution
    
    bursts_included = np.where(peaks_z_score >= z_peak_threshold)[0]
    print(f"Bursts included after z-score filtering: {len(bursts_included)} out of {len(burst_peaks_indices)}")
    
    burst_starts_filtered = burst_starts[bursts_included]
    burst_ends_filtered = burst_ends[bursts_included]
    burst_peaks_indices_filtered = burst_peaks_indices[bursts_included]
    
    # Merge bursts that are closer than min_bursts_distance
    min_bursts_distance = int(min_bursts_distance * 0.001 * fs)  # convert ms to samples
    merged_burst_starts = []
    merged_burst_ends = []
    merged_burst_peaks_indices = []
    for i in range(len(burst_starts_filtered)):
        if len(merged_burst_starts) == 0:
            merged_burst_starts.append(burst_starts_filtered[i])
            merged_burst_ends.append(burst_ends_filtered[i])
            merged_burst_peaks_indices.append(burst_peaks_indices_filtered[i])
        else:
            if burst_starts_filtered[i] - merged_burst_ends[-1] <= min_bursts_distance:
                # Merge bursts
                merged_burst_ends[-1] = max(merged_burst_ends[-1], burst_ends_filtered[i])
                if sigma_envelop[burst_peaks_indices_filtered[i]] > sigma_envelop[merged_burst_peaks_indices[-1]]:
                    merged_burst_peaks_indices[-1] = burst_peaks_indices_filtered[i]
            else:
                merged_burst_starts.append(burst_starts_filtered[i])
                merged_burst_ends.append(burst_ends_filtered[i])
                merged_burst_peaks_indices.append(burst_peaks_indices_filtered[i])
    
    merged_burst_starts = np.array(merged_burst_starts)
    merged_burst_ends = np.array(merged_burst_ends)
    merged_burst_peaks_indices = np.array(merged_burst_peaks_indices)
    assert len(merged_burst_starts) == len(merged_burst_ends) == len(merged_burst_peaks_indices), "Merged bursts lists must have the same length."
    print(f"Number of bursts merging: {len(merged_burst_starts)}")
    
    # Filter bursts by duration
    burst_durations = (np.array(merged_burst_ends) - np.array(merged_burst_starts)) / fs
    duration_mask = (burst_durations >= burst_duration_range[0]) & (burst_durations <= burst_duration_range[1])
    burst_starts_final = merged_burst_starts[duration_mask]
    burst_ends_final = merged_burst_ends[duration_mask]
    burst_peaks_indices_final = merged_burst_peaks_indices[duration_mask]
    print(f"Bursts included after duration filtering: {len(burst_starts_final)} out of {len(merged_burst_starts)}")
    assert len(burst_starts_final) == len(burst_ends_final) == len(burst_peaks_indices_final), "Final bursts lists must have the same length."

    """
    _plot_bursts(
        sigma_envelop=sigma_envelop,
        burst_starts=burst_starts,
        burst_ends=burst_ends,
        burst_peaks_indices=burst_peaks_indices,
        burst_starts_filtered=burst_starts_filtered,
        burst_ends_filtered=burst_ends_filtered,
        burst_peaks_indices_fliltered=burst_peaks_indices_filtered,
        merged_burst_starts=merged_burst_starts,
        merged_burst_ends=merged_burst_ends,
        merged_burst_peaks_indices=merged_burst_peaks_indices,
        burst_starts_final=burst_starts_final,
        burst_ends_final=burst_ends_final,
        burst_peaks_indices_final=burst_peaks_indices_final,
        fs=fs,    
    )
    """
    
    return burst_starts_final, burst_ends_final, burst_peaks_indices_final

def _plot_bursts(
        sigma_envelop: np.ndarray,
        burst_starts: np.ndarray,
        burst_ends: np.ndarray,
        burst_peaks_indices: np.ndarray,
        burst_starts_filtered: np.ndarray,
        burst_ends_filtered: np.ndarray,
        burst_peaks_indices_fliltered: np.ndarray,
        merged_burst_starts: np.ndarray,
        merged_burst_ends: np.ndarray,
        merged_burst_peaks_indices: np.ndarray,
        burst_starts_final: np.ndarray,
        burst_ends_final: np.ndarray,
        burst_peaks_indices_final: np.ndarray,
        fs: float,
    ):
    """
    Plot each step result to check if the filtering is working correctly.
    """
    window_size = int(fs * 30)  # 30 seconds window
    
    def _plot_bursts_in_window(ax, idx, window_size, burst_starts, burst_ends, burst_peaks_indices, color='red'):
        ax.plot(np.arange(idx, idx+window_size, dtype=int), sigma_envelop[idx:idx + window_size], color='gray', label='Sigma Envelope')
        _peak_indices = burst_peaks_indices[(burst_peaks_indices >= idx) & (burst_peaks_indices < idx + window_size)]
        ax.scatter(_peak_indices, sigma_envelop[_peak_indices], color=color)
        
        for _sampling_idx in range(idx, idx + window_size):
            if _sampling_idx in burst_starts:
                ax.axvline(x=_sampling_idx, color='black', linestyle='--')
            if _sampling_idx in burst_ends:
                ax.axvline(x=_sampling_idx, color='black', linestyle='-')
        
    
    for idx in range(0, len(sigma_envelop), window_size):
        fig, ax = plt.subplots(4, 1, sharex=True, figsize=(15, 10))
        
        _plot_bursts_in_window(ax[0], idx, window_size, burst_starts, burst_ends, burst_peaks_indices, color='red')
        ax[0].set_title('Original Bursts')
        
        _plot_bursts_in_window(ax[1], idx, window_size, burst_starts_filtered, burst_ends_filtered, burst_peaks_indices_fliltered, color='blue')
        ax[1].set_title('Filtered Bursts (Z-score)')
        
        _plot_bursts_in_window(ax[2], idx, window_size, merged_burst_starts, merged_burst_ends, merged_burst_peaks_indices, color='green')
        ax[2].set_title('Merged Bursts')
        
        _plot_bursts_in_window(ax[3], idx, window_size, burst_starts_final, burst_ends_final, burst_peaks_indices_final, color='purple')
        ax[3].set_title('Final Bursts (Duration Filtered)')
        
        fig.tight_layout()
        plt.show()
        
        if input("Press Enter to continue to the next segment, or type 'q' to quit plotting: ") == 'q':
            break