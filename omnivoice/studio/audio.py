"""Âm thanh: đọc file, kiểm tra chất lượng giọng mẫu, cắt lặng, chọn đoạn mẫu tốt nhất, lưu WAV/MP3."""
import os
import subprocess
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from .config import FFMPEG, SR_OUT


def load_mono(path, sr: int | None = SR_OUT) -> tuple[np.ndarray, int]:
    a, rate = librosa.load(str(path), sr=sr, mono=True)
    return a.astype(np.float32), rate


def _frames_rms(a: np.ndarray, sr: int, hop_s: float = 0.02) -> np.ndarray:
    hop = max(1, int(hop_s * sr))
    return librosa.feature.rms(y=a, frame_length=hop * 2, hop_length=hop)[0]


def analyze(a: np.ndarray, sr: int) -> dict:
    """Thông số chất lượng của một đoạn giọng mẫu."""
    dur = len(a) / sr
    rms = _frames_rms(a, sr)
    db = 20 * np.log10(rms + 1e-9)
    loud = np.percentile(db, 90) if len(db) else -99
    floor = np.percentile(db, 10) if len(db) else -99
    speech = db > loud - 25
    peak = float(np.max(np.abs(a))) if len(a) else 0.0
    return {
        "duration": dur,
        "speech": float(speech.mean() * dur) if len(speech) else 0.0,
        "peak_db": 20 * np.log10(peak + 1e-9),
        "clip": float(np.mean(np.abs(a) > 0.99)) if len(a) else 0.0,
        "level_db": float(loud),
        "snr_db": float(loud - floor),
    }


def _v(x: float) -> str:
    return f"{x:.1f}".replace(".", ",")


def verdict(info: dict) -> list[tuple[str, str]]:
    """Danh sách (lời khuyên, mức ok/warn/bad) theo thông số."""
    out = []
    d = info["duration"]
    if d < 3:
        out.append((f"Ngắn quá ({_v(d)}s). Cần 3–10 giây, tốt nhất 6–10 giây.", "bad"))
    elif d < 5:
        out.append((f"Dài {_v(d)}s: dùng được, nhưng 6–10 giây sẽ giống hơn.", "warn"))
    else:
        out.append((f"Độ dài {_v(d)}s: tốt.", "ok"))
    if info["clip"] > 0.001:
        out.append(("Tiếng bị rè / vỡ (clipping). Thu lại nhỏ hơn hoặc lùi xa micro.", "bad"))
    if info["level_db"] < -35:
        out.append(("Tiếng nhỏ quá. App sẽ tự tăng, nhưng thu to hơn sẽ sạch hơn.", "warn"))
    snr = info["snr_db"]
    if snr < 15:
        out.append((f"Nhiều tạp âm (chênh tiếng nói / nền ≈ {snr:.0f} dB). Nên thu ở phòng yên tĩnh.", "bad"))
    elif snr < 25:
        out.append((f"Có ít tạp âm (≈ {snr:.0f} dB). Dùng được.", "warn"))
    else:
        out.append((f"Nền sạch (≈ {snr:.0f} dB).", "ok"))
    return out


def trim(a: np.ndarray, sr: int, top_db: float = 35, pad: float = 0.08) -> np.ndarray:
    _, (s, e) = librosa.effects.trim(a, top_db=top_db)
    p = int(pad * sr)
    return a[max(0, s - p):min(len(a), e + p)]


def best_window(a: np.ndarray, sr: int, target: float = 9.0, max_len: float = 11.0) -> tuple[np.ndarray, float, float]:
    """File dài: chọn đoạn ~9 giây nhiều tiếng nói nhất, cắt ở chỗ lặng để không đứt chữ."""
    dur = len(a) / sr
    if dur <= max_len:
        return a, 0.0, dur
    hop = 0.02
    rms = _frames_rms(a, sr, hop)
    db = 20 * np.log10(rms + 1e-9)
    speech = (db > np.percentile(db, 90) - 25).astype(float)
    win = int(target / hop)
    score = np.convolve(speech, np.ones(win), "valid")
    quiet = db < np.percentile(db, 30)
    best, best_s = 0, -1.0
    for i in range(0, len(score), 5):                     # ưu tiên đoạn bắt đầu và kết thúc ở chỗ lặng
        s = score[i] + 15 * quiet[i] + 15 * quiet[min(i + win, len(quiet) - 1)]
        if s > best_s:
            best, best_s = i, s
    start, end = best * hop, min(dur, best * hop + target)
    return a[int(start * sr):int(end * sr)], start, end


def normalize(a: np.ndarray, peak_db: float = -1.5) -> np.ndarray:
    peak = float(np.max(np.abs(a))) if len(a) else 0.0
    return a if peak <= 0 else (a * (10 ** (peak_db / 20)) / peak).astype(np.float32)


def prepare_ref(path) -> tuple[np.ndarray, int, dict]:
    """Chuẩn bị giọng mẫu: mono 24 kHz, cắt lặng, chọn đoạn tốt nhất nếu dài, chuẩn hoá độ to."""
    a, sr = load_mono(path)
    raw = analyze(a, sr)
    a = trim(a, sr)
    a, s, e = best_window(a, sr)
    a = normalize(trim(a, sr))
    info = analyze(a, sr)
    info.update(raw_duration=raw["duration"], window=(s, e), clip=raw["clip"])
    return a, sr, info


def save(a: np.ndarray, sr: int, path: Path, fmt: str = "wav") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "mp3":
        tmp = path.with_suffix(".tmp.wav")
        sf.write(str(tmp), a, sr)
        subprocess.run([FFMPEG, "-v", "error", "-y", "-i", str(tmp), "-b:a", "192k", str(path.with_suffix(".mp3"))],
                       check=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        os.remove(tmp)
        return path.with_suffix(".mp3")
    sf.write(str(path.with_suffix(".wav")), a, sr)
    return path.with_suffix(".wav")
