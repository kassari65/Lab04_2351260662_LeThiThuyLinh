# CSE457 • LAB 04: TỔNG HỢP TIẾNG NÓI (TEXT-TO-SPEECH SYNTHESIS)

**Trường Đại học Thủy Lợi** – Khoa Công nghệ Thông tin – Bộ môn Trí tuệ Nhân tạo  
**Học phần:** CSE457 – Xử lý Âm thanh và Tiếng nói  
**Sinh viên thực hiện:** Lê Thị Thùy Linh  
**Mã số sinh viên:** 2351260662  

---

## 📌 Giới thiệu dự án
Dự án hoàn thiện toàn bộ yêu cầu thực hành của **Lab 04: Tổng Hợp Tiếng Nói** từ kiến trúc TTS, chuẩn hóa văn bản, ghép nối đơn vị âm thanh (Unit Selection Viterbi), làm trơn biên nối (Cross-fade), mô hình Nguồn – Bộ lọc (Source–Filter), tổng hợp tham số HMM (HMM-based Parametric Synthesis với MLPG) đến tổng hợp tiếng nói học sâu (Neural TTS Pipeline: CNN + Mel-80 + Griffin-Lim Vocoder).

Dự án áp dụng trên miền bài toán thực tế: **Đoạn thoại tiếp thị mời chào khách hàng quầy thu ngân/ưu đãi** với cả **Giọng Nam** và **Giọng Nữ**.

---

## 📂 Cấu trúc thư mục

```text
.
├── Lab04.ipynb                     # Notebook chính thực thi 22 cell toàn bộ quy trình
├── report.md                       # Bản báo cáo khoa học chi tiết toàn diện
├── README.md                       # Tài liệu hướng dẫn dự án
├── .gitignore                      # File cấu hình bỏ qua file tạm
│
├── lab4_data/                      # Kho dữ liệu âm thanh (Speech Inventory)
│   ├── metadata.csv                # Quản lý 87 đơn vị âm thanh (29 từ x 3 ứng viên Nam & Nữ)
│   └── units/                      # 87 file WAV (16kHz, mono, PCM_16, RMS=0.08)
│
├── audio_output/                   # Thư mục kết quả âm thanh và hình ảnh
│   ├── raw_concat.wav              # Ghép nối cơ sở (Baseline, xfade = 0ms)
│   ├── best_concat.wav             # Ghép nối tối ưu Viterbi Giọng Nam (xfade = 15ms)
│   ├── best_concat_female.wav      # Ghép nối tối ưu Viterbi Giọng Nữ (xfade = 15ms)
│   ├── concat_xfade_*.wav          # Thử nghiệm 4 mức Cross-fade (0, 5, 15, 30 ms)
│   ├── prosody_*.wav               # Thử nghiệm co giãn thời lượng & dịch cao độ
│   ├── hmm_tts.wav                 # Kết quả tổng hợp tham số HMM Giọng Nam
│   ├── hmm_female_tts.wav          # Kết quả tổng hợp tham số HMM Giọng Nữ
│   ├── neural_tts_*.wav            # Kết quả tổng hợp học sâu Neural TTS
│   ├── ratings.csv                 # Dữ liệu thẩm định nghe chủ quan (MOS & Intelligibility)
│   └── fig1 đến fig7 (.png)        # 7 biểu đồ minh họa kết quả thực nghiệm
│
├── hmm_tts_backend.py              # Backend sinh tham số HMM và tái tổng hợp IIR
├── neural_tts_backend.py           # Backend PyTorch Neural TTS
└── Lab 4.pdf                       # Đề bài và quy chuẩn thực hành gốc
```

---

## 🚀 Hướng dẫn cài đặt và chạy

### 1. Cài đặt môi trường Python
```bash
pip install numpy scipy matplotlib soundfile librosa torch torchaudio pandas
```

### 2. Thực thi bài thực hành
Mở và chạy tuần tự các cell trong file **`Lab04.ipynb`** bằng Jupyter Notebook hoặc VS Code.

---

## 📊 Tóm tắt kết quả thực nghiệm

| Phương pháp | Độ rõ (Intelligibility) | Điểm tự nhiên (MOS) | Tính liên tục (Continuity) | Hệ số thời gian thực (RTF) |
| :--- | :---: | :---: | :--- | :---: |
| **Concatenative TTS (Viterbi + 15ms xfade)** | 100% | 3.96 ± 0.46 | Khá cao, không click | < 0.01 (Cực nhanh) |
| **HMM-based Parametric TTS** | 100% | 2.96 ± 0.46 | Rất cao, quỹ đạo trơn | ~0.03 |
| **Deep Neural TTS (PyTorch + Vocoder)** | 100% | **4.46 ± 0.51** | Xuất sắc, tự nhiên nhất | 0.054 – 0.093 (< 0.1) |

Chi tiết các phân tích toán học và trả lời 10 câu hỏi lý thuyết xem tại file [**`report.md`**](report.md).
