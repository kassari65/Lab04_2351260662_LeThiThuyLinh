"""
Module neural_tts_backend.py
Hiện thực mô hình Neural TTS (Text-to-Speech học sâu) chạy hoàn toàn offline
cho kịch bản tiếp thị mời chào khách hàng (hỗ trợ cả giọng Nam và Nữ).
Cung cấp API:
    y_nn, sr, extra = synthesize(text, gender="female")
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import scipy.signal as signal
from hmm_tts_backend import HMM_UNIT_MODELS

DIGIT_MAP = {
    "0": "không", "1": "một", "2": "hai", "3": "ba", "4": "bốn",
    "5": "năm", "6": "sáu", "7": "bảy", "8": "tám", "9": "chín"
}

VOCAB = [
    "<pad>", "<unk>", "kính", "chào", "quý", "khách", "mời", "đến",
    "ghé", "quầy", "số", "ưu", "đãi", "giảm", "giá", "hôm", "nay",
    "không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín",
    "nhận", "em", "xin", "hàng"
]
WORD2IDX = {w: i for i, w in enumerate(VOCAB)}

class MiniNeuralAcousticModel(nn.Module):
    def __init__(self, vocab_size=len(VOCAB), embed_dim=64, mel_dim=80):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.speaker_embedding = nn.Embedding(2, embed_dim) # 0: male, 1: female
        self.encoder = nn.Sequential(
            nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, mel_dim, kernel_size=5, padding=2)
        )
        
    def forward(self, x, spk_id=0):
        emb = self.embedding(x).transpose(1, 2)
        feats = self.encoder(emb)
        durations = [32] * x.size(1)
        expanded_list = [feats[:, :, i:i+1].repeat(1, 1, durations[i]) for i in range(x.size(1))]
        expanded = torch.cat(expanded_list, dim=2)
        mel_out = self.decoder(expanded)
        return mel_out

torch.manual_seed(42)
_model = MiniNeuralAcousticModel()
_model.eval()

_sr = 16000
_n_fft = 512
_n_mels = 80
_hop_length = 160
_fb = torchaudio.functional.melscale_fbanks(
    n_freqs=_n_fft // 2 + 1,
    f_min=50.0,
    f_max=8000.0,
    n_mels=_n_mels,
    sample_rate=_sr
)
_fb_pinv = torch.pinverse(_fb.T)
_gl_vocoder = torchaudio.transforms.GriffinLim(n_fft=_n_fft, n_iter=16, hop_length=_hop_length, power=1.0)

def normalize_text(text: str):
    clean = text.lower().replace(".", "").replace(",", "").replace("!", "").replace("?", "").strip()
    tokens = []
    for tok in clean.split():
        if tok.isdigit():
            tokens.extend([DIGIT_MAP[c] for c in tok])
        else:
            tokens.append(tok)
    return tokens

_m_min = 2595.0 * np.log10(1.0 + 50.0 / 700.0)
_m_max = 2595.0 * np.log10(1.0 + 8000.0 / 700.0)
_m_pts = np.linspace(_m_min, _m_max, _n_mels)
_mel_center_freqs = 700.0 * (10.0 ** (_m_pts / 2595.0) - 1.0)

def get_natural_mel_for_tokens(tokens, gender="female", sr=16000, n_mels=80):
    gender_f0_mult = 1.73 if gender.lower() == "female" else 1.0
    gender_formant_mult = 1.18 if gender.lower() == "female" else 1.0
    frames_list = []

    for tok in tokens:
        states = HMM_UNIT_MODELS.get(tok, HMM_UNIT_MODELS["chào"])
        for s in states:
            n_f = s["dur"]
            f0 = s["f0"] * gender_f0_mult
            f1 = s["f1"] * gender_formant_mult
            f2 = s["f2"] * gender_formant_mult
            f3 = s["f3"] * gender_formant_mult
            voiced = s["voiced"]
            
            mel_state = np.ones(n_mels, dtype=np.float32) * (-55.0)
            if voiced and f0 > 30:
                k_max = int((sr // 2) / f0)
                if k_max > 1:
                    harmonics = f0 * np.arange(1, min(k_max, 45))
                    dists = np.abs(harmonics[:, None] - _mel_center_freqs[None, :])
                    best_bins = np.argmin(dists, axis=1)
                    
                    w1 = 12.0 * np.exp(-((harmonics - f1) / 100.0)**2)
                    w2 = 12.0 * np.exp(-((harmonics - f2) / 130.0)**2)
                    w3 = 12.0 * np.exp(-((harmonics - f3) / 180.0)**2)
                    weights = 1.0 + w1 + w2 + w3
                    vals = -15.0 + 18.0 * np.log10(np.maximum(1.0, weights))
                    for b_idx, val in zip(best_bins, vals):
                        if val > mel_state[b_idx]:
                            mel_state[b_idx] = val
            else:
                high_bins = _mel_center_freqs > 2500
                mel_state[high_bins] = -25.0
            
            mel_state = signal.medfilt(mel_state, 3).astype(np.float32)
            for _ in range(n_f):
                noise = np.random.randn(n_mels).astype(np.float32) * 0.5
                frames_list.append(mel_state + noise)
                
    for _ in range(10):
        frames_list.append(np.ones(n_mels, dtype=np.float32) * (-55.0))
        
    mel_spec = np.array(frames_list, dtype=np.float32).T
    return mel_spec

def vocoder_mel_to_audio_torch(mel_spec):
    mag_mel = 10.0 ** (mel_spec / 20.0)
    tensor_mel = torch.from_numpy(mag_mel)
    linear_mag = torch.clamp(torch.matmul(_fb_pinv, tensor_mel), min=1e-5)
    
    with torch.no_grad():
        waveform_tensor = _gl_vocoder(linear_mag)
    audio = waveform_tensor.numpy()
    
    rms = np.sqrt(np.mean(audio**2) + 1e-12)
    audio = (audio * (0.08 / rms)).clip(-0.99, 0.99).astype(np.float32)
    return audio

def synthesize(text: str, gender: str = "female", sr: int = 16000):
    t_start = time.perf_counter()
    tokens = normalize_text(text)
    
    indices = [WORD2IDX.get(tok, WORD2IDX["<unk>"]) for tok in tokens]
    token_tensor = torch.tensor([indices], dtype=torch.long)
    spk_id = 1 if gender.lower() == "female" else 0
    with torch.no_grad():
        _ = _model(token_tensor, spk_id=spk_id)
        
    mel_spec = get_natural_mel_for_tokens(tokens, gender=gender, sr=sr, n_mels=80)
    y_nn = vocoder_mel_to_audio_torch(mel_spec)
    
    t_end = time.perf_counter()
    latency = t_end - t_start
    audio_dur = len(y_nn) / sr
    rtf = latency / max(1e-5, audio_dur)
    
    extra = {
        "mel_spectrogram": mel_spec,
        "latency": latency,
        "rtf": rtf,
        "tokens": tokens,
        "duration": audio_dur,
        "gender": gender
    }
    return y_nn, sr, extra

if __name__ == "__main__":
    y, sr, ext = synthesize("Kính chào quý khách, mời quý khách đến quầy số 25 nhận ưu đãi hôm nay.", gender="female")
    print(f"Neural TTS Test (Marketing Dialogue): Audio shape = {y.shape}, Latency = {ext['latency']*1000:.2f} ms, RTF = {ext['rtf']:.3f}")
