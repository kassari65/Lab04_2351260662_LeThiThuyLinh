"""
Module hmm_tts_backend.py
Hiện thực mô hình HMM-based Parametric Speech Synthesis cho kịch bản tiếp thị mời chào khách hàng.
Hỗ trợ cả Giọng Nam và Giọng Nữ.
Dựa trên lý thuyết trong Huang et al. (2001) Ch. 16 và Tokuda et al. (2000).
"""

import numpy as np
import scipy.signal as signal

def normalize_text_simple(text):
    DIGIT = {
        "0": "không", "1": "một", "2": "hai", "3": "ba", "4": "bốn",
        "5": "năm", "6": "sáu", "7": "bảy", "8": "tám", "9": "chín"
    }
    clean = text.lower().replace(".", "").replace(",", "").replace("!", "").replace("?", "").strip()
    tokens = []
    for tok in clean.split():
        if tok.isdigit():
            tokens.extend([DIGIT[c] for c in tok])
        else:
            tokens.append(tok)
    return tokens

# Bảng phân bố HMM cho các từ tiếp thị & số đếm
HMM_UNIT_MODELS = {
    "kính": [
        {"dur": 10, "f0": 0,   "f1": 350, "f2": 2100, "f3": 2800, "voiced": False},
        {"dur": 16, "f0": 135, "f1": 320, "f2": 2250, "f3": 2900, "voiced": True},
        {"dur": 10, "f0": 155, "f1": 380, "f2": 2000, "f3": 2750, "voiced": True},
    ],
    "chào": [
        {"dur": 10, "f0": 0,   "f1": 500, "f2": 1500, "f3": 2550, "voiced": False},
        {"dur": 20, "f0": 132, "f1": 720, "f2": 1150, "f3": 2500, "voiced": True},
        {"dur": 12, "f0": 108, "f1": 550, "f2": 950,  "f3": 2400, "voiced": True},
    ],
    "quý": [
        {"dur": 10, "f0": 126, "f1": 380, "f2": 950,  "f3": 2450, "voiced": True},
        {"dur": 16, "f0": 135, "f1": 350, "f2": 2150, "f3": 2800, "voiced": True},
        {"dur": 10, "f0": 160, "f1": 320, "f2": 2300, "f3": 2900, "voiced": True},
    ],
    "khách": [
        {"dur": 10, "f0": 0,   "f1": 600, "f2": 1400, "f3": 2500, "voiced": False},
        {"dur": 16, "f0": 135, "f1": 720, "f2": 1350, "f3": 2600, "voiced": True},
        {"dur": 10, "f0": 155, "f1": 600, "f2": 1800, "f3": 2700, "voiced": False},
    ],
    "mời": [
        {"dur": 10, "f0": 135, "f1": 350, "f2": 1100, "f3": 2500, "voiced": True},
        {"dur": 18, "f0": 125, "f1": 550, "f2": 1300, "f3": 2550, "voiced": True},
        {"dur": 12, "f0": 105, "f1": 420, "f2": 2000, "f3": 2650, "voiced": True},
    ],
    "đến": [
        {"dur": 8,  "f0": 126, "f1": 400, "f2": 1500, "f3": 2600, "voiced": True},
        {"dur": 18, "f0": 132, "f1": 520, "f2": 1850, "f3": 2650, "voiced": True},
        {"dur": 12, "f0": 158, "f1": 400, "f2": 1600, "f3": 2600, "voiced": True},
    ],
    "ghé": [
        {"dur": 8,  "f0": 125, "f1": 450, "f2": 1600, "f3": 2600, "voiced": True},
        {"dur": 16, "f0": 132, "f1": 520, "f2": 1900, "f3": 2650, "voiced": True},
        {"dur": 10, "f0": 160, "f1": 480, "f2": 1850, "f3": 2600, "voiced": True},
    ],
    "quầy": [
        {"dur": 10, "f0": 124, "f1": 450, "f2": 900,  "f3": 2450, "voiced": True},
        {"dur": 20, "f0": 128, "f1": 560, "f2": 1150, "f3": 2500, "voiced": True},
        {"dur": 12, "f0": 106, "f1": 450, "f2": 2000, "f3": 2680, "voiced": True},
    ],
    "số": [
        {"dur": 12, "f0": 0,   "f1": 450, "f2": 1200, "f3": 2500, "voiced": False},
        {"dur": 16, "f0": 130, "f1": 520, "f2": 950,  "f3": 2500, "voiced": True},
        {"dur": 10, "f0": 160, "f1": 480, "f2": 920,  "f3": 2450, "voiced": True},
    ],
    "ưu": [
        {"dur": 10, "f0": 130, "f1": 380, "f2": 950,  "f3": 2450, "voiced": True},
        {"dur": 18, "f0": 130, "f1": 360, "f2": 950,  "f3": 2450, "voiced": True},
        {"dur": 12, "f0": 126, "f1": 350, "f2": 920,  "f3": 2400, "voiced": True},
    ],
    "đãi": [
        {"dur": 8,  "f0": 125, "f1": 420, "f2": 1200, "f3": 2500, "voiced": True},
        {"dur": 18, "f0": 130, "f1": 720, "f2": 1350, "f3": 2600, "voiced": True},
        {"dur": 12, "f0": 155, "f1": 450, "f2": 2100, "f3": 2750, "voiced": True},
    ],
    "giảm": [
        {"dur": 10, "f0": 0,   "f1": 500, "f2": 1300, "f3": 2500, "voiced": False},
        {"dur": 18, "f0": 120, "f1": 750, "f2": 1250, "f3": 2550, "voiced": True},
        {"dur": 12, "f0": 128, "f1": 450, "f2": 1150, "f3": 2500, "voiced": True},
    ],
    "giá": [
        {"dur": 10, "f0": 0,   "f1": 500, "f2": 1300, "f3": 2500, "voiced": False},
        {"dur": 16, "f0": 132, "f1": 760, "f2": 1250, "f3": 2550, "voiced": True},
        {"dur": 10, "f0": 160, "f1": 720, "f2": 1200, "f3": 2500, "voiced": True},
    ],
    "hôm": [
        {"dur": 10, "f0": 0,   "f1": 500, "f2": 1000, "f3": 2450, "voiced": False},
        {"dur": 18, "f0": 130, "f1": 520, "f2": 920,  "f3": 2450, "voiced": True},
        {"dur": 12, "f0": 126, "f1": 420, "f2": 900,  "f3": 2400, "voiced": True},
    ],
    "nay": [
        {"dur": 10, "f0": 128, "f1": 380, "f2": 1300, "f3": 2550, "voiced": True},
        {"dur": 18, "f0": 130, "f1": 720, "f2": 1350, "f3": 2600, "voiced": True},
        {"dur": 12, "f0": 126, "f1": 450, "f2": 2100, "f3": 2750, "voiced": True},
    ],
    "không": [
        {"dur": 10, "f0": 0,   "f1": 500, "f2": 1000, "f3": 2450, "voiced": False},
        {"dur": 18, "f0": 132, "f1": 520, "f2": 920,  "f3": 2450, "voiced": True},
        {"dur": 12, "f0": 128, "f1": 420, "f2": 900,  "f3": 2400, "voiced": True},
    ],
    "một": [
        {"dur": 8,  "f0": 130, "f1": 380, "f2": 1100, "f3": 2500, "voiced": True},
        {"dur": 14, "f0": 120, "f1": 520, "f2": 1050, "f3": 2500, "voiced": True},
        {"dur": 8,  "f0": 95,  "f1": 450, "f2": 1400, "f3": 2550, "voiced": False},
    ],
    "hai": [
        {"dur": 10, "f0": 0,   "f1": 650, "f2": 1300, "f3": 2550, "voiced": False},
        {"dur": 16, "f0": 132, "f1": 760, "f2": 1350, "f3": 2600, "voiced": True},
        {"dur": 12, "f0": 128, "f1": 450, "f2": 2100, "f3": 2750, "voiced": True},
    ],
    "ba": [
        {"dur": 8,  "f0": 125, "f1": 400, "f2": 1100, "f3": 2500, "voiced": True},
        {"dur": 20, "f0": 130, "f1": 780, "f2": 1220, "f3": 2550, "voiced": True},
        {"dur": 10, "f0": 126, "f1": 720, "f2": 1200, "f3": 2500, "voiced": True},
    ],
    "bốn": [
        {"dur": 8,  "f0": 125, "f1": 400, "f2": 1000, "f3": 2450, "voiced": True},
        {"dur": 16, "f0": 132, "f1": 520, "f2": 880,  "f3": 2450, "voiced": True},
        {"dur": 12, "f0": 158, "f1": 420, "f2": 1400, "f3": 2550, "voiced": True},
    ],
    "năm": [
        {"dur": 10, "f0": 128, "f1": 380, "f2": 1300, "f3": 2550, "voiced": True},
        {"dur": 18, "f0": 130, "f1": 740, "f2": 1300, "f3": 2550, "voiced": True},
        {"dur": 12, "f0": 126, "f1": 420, "f2": 1150, "f3": 2500, "voiced": True},
    ],
    "sáu": [
        {"dur": 12, "f0": 0,   "f1": 500, "f2": 1200, "f3": 2500, "voiced": False},
        {"dur": 16, "f0": 130, "f1": 720, "f2": 1200, "f3": 2550, "voiced": True},
        {"dur": 10, "f0": 162, "f1": 450, "f2": 850,  "f3": 2450, "voiced": True},
    ],
    "bảy": [
        {"dur": 8,  "f0": 125, "f1": 400, "f2": 1200, "f3": 2550, "voiced": True},
        {"dur": 18, "f0": 120, "f1": 740, "f2": 1350, "f3": 2600, "voiced": True},
        {"dur": 12, "f0": 128, "f1": 480, "f2": 1950, "f3": 2700, "voiced": True},
    ],
    "tám": [
        {"dur": 10, "f0": 0,   "f1": 550, "f2": 1300, "f3": 2500, "voiced": False},
        {"dur": 16, "f0": 132, "f1": 760, "f2": 1260, "f3": 2550, "voiced": True},
        {"dur": 12, "f0": 156, "f1": 420, "f2": 1100, "f3": 2500, "voiced": True},
    ],
    "chín": [
        {"dur": 10, "f0": 0,   "f1": 400, "f2": 1800, "f3": 2650, "voiced": False},
        {"dur": 18, "f0": 132, "f1": 320, "f2": 2250, "f3": 2950, "voiced": True},
        {"dur": 12, "f0": 160, "f1": 350, "f2": 1750, "f3": 2750, "voiced": True},
    ],
    "nhận": [
        {"dur": 8,  "f0": 128, "f1": 380, "f2": 1600, "f3": 2650, "voiced": True},
        {"dur": 14, "f0": 118, "f1": 520, "f2": 1100, "f3": 2500, "voiced": True},
        {"dur": 8,  "f0": 90,  "f1": 420, "f2": 1400, "f3": 2550, "voiced": False},
    ],
    "em": [
        {"dur": 8,  "f0": 126, "f1": 480, "f2": 1800, "f3": 2600, "voiced": True},
        {"dur": 18, "f0": 130, "f1": 520, "f2": 1850, "f3": 2650, "voiced": True},
        {"dur": 10, "f0": 126, "f1": 420, "f2": 1300, "f3": 2500, "voiced": True},
    ],
    "xin": [
        {"dur": 12, "f0": 0,   "f1": 400, "f2": 1800, "f3": 2650, "voiced": False},
        {"dur": 16, "f0": 130, "f1": 320, "f2": 2200, "f3": 2850, "voiced": True},
        {"dur": 10, "f0": 128, "f1": 380, "f2": 1800, "f3": 2700, "voiced": True},
    ],
    "hàng": [
        {"dur": 10, "f0": 0,   "f1": 650, "f2": 1200, "f3": 2450, "voiced": False},
        {"dur": 22, "f0": 132, "f1": 760, "f2": 1250, "f3": 2500, "voiced": True},
        {"dur": 12, "f0": 110, "f1": 450, "f2": 1100, "f3": 2450, "voiced": True},
    ]
}

def smooth_trajectory_with_deltas(target_frames, window_len=7):
    if len(target_frames) < window_len:
        return target_frames
    win = np.hanning(window_len)
    win = win / np.sum(win)
    smoothed = np.convolve(target_frames, win, mode='same')
    pad = window_len // 2
    smoothed[:pad] = target_frames[:pad]
    smoothed[-pad:] = target_frames[-pad:]
    return smoothed

def synthesize_hmm(text: str, duration_scale: float = 1.0, f0_scale: float = 1.0, gender: str = "male", sr: int = 16000):
    tokens = normalize_text_simple(text)
    frame_shift_ms = 10
    frame_samples = int(sr * frame_shift_ms / 1000.0)

    # Nếu là giọng nữ, nhân thêm hệ số F0 (~1.73x) và Formants (~1.18x)
    gender_f0_mult = 1.73 if gender.lower() == "female" else 1.0
    gender_formant_mult = 1.18 if gender.lower() == "female" else 1.0

    all_states = []
    dur_list = []
    f0_steps = []
    f1_steps = []
    f2_steps = []
    f3_steps = []
    voiced_steps = []

    for word in tokens:
        states = HMM_UNIT_MODELS.get(word, HMM_UNIT_MODELS["chào"])
        for idx, s in enumerate(states):
            state_label = f"{word}_s{idx+1}"
            all_states.append(state_label)
            dur = max(3, int(s["dur"] * duration_scale))
            dur_list.append(dur)
            
            f0_val = s["f0"] * f0_scale * gender_f0_mult if s["voiced"] else 0.0
            f0_steps.extend([f0_val] * dur)
            f1_steps.extend([s["f1"] * gender_formant_mult] * dur)
            f2_steps.extend([s["f2"] * gender_formant_mult] * dur)
            f3_steps.extend([s["f3"] * gender_formant_mult] * dur)
            voiced_steps.extend([s["voiced"]] * dur)

    sil_dur = 10
    all_states.append("sil")
    dur_list.append(sil_dur)
    f0_steps.extend([0.0] * sil_dur)
    f1_steps.extend([500.0 * gender_formant_mult] * sil_dur)
    f2_steps.extend([1500.0 * gender_formant_mult] * sil_dur)
    f3_steps.extend([2500.0 * gender_formant_mult] * sil_dur)
    voiced_steps.extend([False] * sil_dur)

    f0_raw = np.array(f0_steps, dtype=np.float32)
    f1_raw = np.array(f1_steps, dtype=np.float32)
    f2_raw = np.array(f2_steps, dtype=np.float32)
    f3_raw = np.array(f3_steps, dtype=np.float32)
    voiced_arr = np.array(voiced_steps, dtype=bool)

    f0_smooth = smooth_trajectory_with_deltas(f0_raw, window_len=9)
    f0_smooth[~voiced_arr] = 0.0

    f1_smooth = smooth_trajectory_with_deltas(f1_raw, window_len=9)
    f2_smooth = smooth_trajectory_with_deltas(f2_raw, window_len=9)
    f3_smooth = smooth_trajectory_with_deltas(f3_raw, window_len=9)

    tracks = {
        "f0": f0_smooth,
        "duration": dur_list,
        "states": all_states,
        "spectral": {
            "f1": f1_smooth,
            "f2": f2_smooth,
            "f3": f3_smooth
        },
        "voiced": voiced_arr
    }

    total_frames = len(f0_smooth)
    total_samples = total_frames * frame_samples
    
    f0_sample = np.repeat(f0_smooth, frame_samples)
    f1_sample = np.repeat(f1_smooth, frame_samples)
    f2_sample = np.repeat(f2_smooth, frame_samples)
    f3_sample = np.repeat(f3_smooth, frame_samples)
    voiced_sample = np.repeat(voiced_arr, frame_samples)

    excitation = np.zeros(total_samples, dtype=np.float32)
    phase = 0.0
    for i in range(total_samples):
        if voiced_sample[i] and f0_sample[i] > 30.0:
            phase += f0_sample[i] / sr
            if phase >= 1.0:
                phase -= 1.0
                excitation[i] = 1.0
        else:
            phase = 0.0
            excitation[i] = np.random.randn() * 0.15

    excitation = signal.lfilter([1.0, -0.96], [1.0], excitation)

    y = excitation.copy()
    bw_list = [80, 110, 150]
    for form_traj, bw in zip([f1_sample, f2_sample, f3_sample], bw_list):
        zi = np.zeros(2)
        y_filtered = np.zeros_like(y)
        for b_start in range(0, total_samples, frame_samples):
            b_end = min(b_start + frame_samples, total_samples)
            fc = np.mean(form_traj[b_start:b_end])
            r = np.exp(-np.pi * bw / sr)
            theta = 2.0 * np.pi * fc / sr
            a_poly = [1.0, -2.0 * r * np.cos(theta), r**2]
            b_poly = [1.0 - r]
            y_block, zi = signal.lfilter(b_poly, a_poly, y[b_start:b_end], zi=zi)
            y_filtered[b_start:b_end] = y_block
        y = y_filtered

    rms = np.sqrt(np.mean(y**2) + 1e-12)
    y_hmm = (y * (0.08 / rms)).clip(-0.99, 0.99).astype(np.float32)

    return y_hmm, sr, tracks

if __name__ == "__main__":
    y, sr, tr = synthesize_hmm("Kính chào quý khách, mời quý khách đến quầy số 25 nhận ưu đãi hôm nay.")
    print(f"HMM Synthesis Test Complete: Audio shape = {y.shape}, F0 frames = {len(tr['f0'])}")
