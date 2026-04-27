import numpy as np

def get_sw_pha_unwrapped(
        sw_pha_unwrapped,
        peak_index,
        sw_target_start,
        sw_target_end,
        sw_expanded_start,
        sw_expanded_end,
        sw_midcrossing,
    ):
    if sw_midcrossing < sw_target_start or sw_midcrossing > sw_target_end:
        raise Exception(f"sw_midcrossing cannot be outside the real target SO. {sw_midcrossing=} {sw_target_start=} {sw_target_end=}")
    
    if peak_index < sw_expanded_start or peak_index > sw_expanded_end:
        raise Exception(f"peak_index (peak of burst) cannot be outside the expanded SO. {peak_index=} {sw_expanded_start=} {sw_expanded_end=}")
    
    phase_at_zc = sw_pha_unwrapped[int(sw_midcrossing)]
    phase_relative = sw_pha_unwrapped - phase_at_zc - np.pi / 2
    
    return phase_relative[peak_index]