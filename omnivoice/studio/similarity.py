"""Đo độ giống giọng: WavLM speaker verification (microsoft/wavlm-base-plus-sv), cosine giữa hai x-vector.

Ngưỡng của model: cosine ≥ 0,86 thường được xem là cùng một người nói. App hiện cosine × 100 thành "% giống".
"""
import threading

import librosa
import numpy as np
import torch

from .config import SV_MODEL

_lock = threading.Lock()
_model = _fe = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


def _load():
    global _model, _fe
    with _lock:
        if _model is None:
            import transformers.modeling_utils as mu
            from transformers import AutoFeatureExtractor, WavLMForXVector
            mu.auto_conversion = lambda *a, **k: None    # model chỉ có file .bin: tắt bước tự xin chuyển đổi qua mạng
            _fe = AutoFeatureExtractor.from_pretrained(SV_MODEL)
            _model = WavLMForXVector.from_pretrained(SV_MODEL).to(_device).eval()
    return _model, _fe


def _wave16(x) -> np.ndarray:
    if isinstance(x, (str, bytes)) or hasattr(x, "__fspath__"):
        a, _ = librosa.load(str(x), sr=16000, mono=True)
    else:
        wav, sr = x
        a = np.asarray(wav, dtype=np.float32).squeeze()
        if sr != 16000:
            a = librosa.resample(a, orig_sr=sr, target_sr=16000)
    a, _ = librosa.effects.trim(a, top_db=35)
    return a.astype(np.float32)


@torch.inference_mode()
def embedding(x) -> np.ndarray:
    """x: đường dẫn file hoặc (waveform, sample_rate). Trả về vector đã chuẩn hoá."""
    model, fe = _load()
    a = _wave16(x)
    inputs = fe(a, sampling_rate=16000, return_tensors="pt").to(_device)
    e = model(**inputs).embeddings[0]
    e = torch.nn.functional.normalize(e, dim=-1)
    return e.cpu().numpy()


def similarity(a, b) -> float:
    """Độ giống giữa hai giọng, 0–1 (cosine). a, b là file hoặc (waveform, sr) hoặc vector embedding."""
    ea = a if isinstance(a, np.ndarray) and a.ndim == 1 and a.shape[0] < 2000 else embedding(a)
    eb = b if isinstance(b, np.ndarray) and b.ndim == 1 and b.shape[0] < 2000 else embedding(b)
    return float(np.dot(ea, eb))


def label(score: float) -> tuple[str, str]:
    """(nhãn, mức) cho giao diện. Ngưỡng hiệu chỉnh 2026-10-19 trên 5 giọng × 8 bộ thông số × 2 bản:
    bản nhân bản so với chính giọng mẫu trung bình 0,955 (thấp nhất 0,91); so với giọng khác gần nhất 0,915."""
    if score >= 0.95:
        return "Rất giống / Very close", "ok"
    if score >= 0.92:
        return "Giống / Close", "ok"
    if score >= 0.88:
        return "Khá giống / Fair", "warn"
    return "Khác giọng mẫu / Different", "bad"


def rank(out_emb: np.ndarray, target_slug: str, library: dict[str, np.ndarray]) -> tuple[int, str | None, float]:
    """Xếp hạng giọng mẫu trong thư viện theo độ giống với bản tạo ra.
    Trả về (hạng của giọng mẫu, slug giọng khác giống nhất, điểm của giọng đó)."""
    scores = sorted(((float(np.dot(out_emb, e)), s) for s, e in library.items() if e is not None), reverse=True)
    pos = next((i + 1 for i, (_, s) in enumerate(scores) if s == target_slug), 0)
    other = next(((s, sc) for sc, s in scores if s != target_slug), (None, 0.0))
    return pos, other[0], other[1]
