"""Lõi tạo giọng của Studio: nạp giọng mẫu đã lưu, tạo nhiều bản, đo độ giống, lưu kết quả và lịch sử."""
import datetime as dt
import threading

import numpy as np

from omnivoice import OmniVoiceGenerationConfig
from omnivoice.models.omnivoice import VoiceClonePrompt

from . import audio, history, similarity, store
from .config import OUTPUTS, TEST_SENTENCE


class Studio:
    def __init__(self, model):
        self.m = model
        self.sr = model.sampling_rate
        self._prompts: dict[str, tuple[tuple, VoiceClonePrompt]] = {}
        self.lock = threading.Lock()              # một việc dùng GPU mỗi lúc

    # ── Giọng mẫu ────────────────────────────────────────────────────────
    def prompt(self, slug: str) -> VoiceClonePrompt:
        """Giọng mẫu đã mã hoá. Lấy từ bộ nhớ, rồi tới prompt.pt, cuối cùng mới mã hoá lại từ ref.wav."""
        p, m = store.paths(slug), store.get(slug)
        key = (p["ref"].stat().st_mtime, m.get("ref_text") or "")
        hit = self._prompts.get(slug)
        if hit and hit[0] == key:
            return hit[1]
        pr = None
        if p["prompt"].exists() and p["prompt"].stat().st_mtime >= p["ref"].stat().st_mtime:
            try:
                pr = VoiceClonePrompt.load(str(p["prompt"]))
            except Exception:
                pr = None
        if pr is None:
            pr = self.m.create_voice_clone_prompt(ref_audio=str(p["ref"]), ref_text=m.get("ref_text") or None)
            pr.save(str(p["prompt"]))
            if not m.get("ref_text"):
                store.update(slug, ref_text=pr.ref_text)
        self._prompts[slug] = (key, pr)
        return pr

    def emb(self, slug: str) -> np.ndarray:
        e = store.load_emb(slug)
        if e is None:
            e = similarity.embedding(str(store.paths(slug)["ref"]))
            np.save(store.paths(slug)["emb"], e)
        return e

    def library_embs(self) -> dict[str, np.ndarray]:
        return {m["slug"]: self.emb(m["slug"]) for m in store.list_voices()}

    def transcribe(self, path) -> str:
        try:
            return self.m.transcribe(str(path))
        except Exception:
            return ""

    # ── Tạo giọng ────────────────────────────────────────────────────────
    @staticmethod
    def config(num_step=32, guidance_scale=2.0, denoise=True, preprocess=True, postprocess=True):
        return OmniVoiceGenerationConfig(num_step=int(num_step), guidance_scale=float(guidance_scale),
                                         denoise=bool(denoise), preprocess_prompt=bool(preprocess),
                                         postprocess_output=bool(postprocess))

    def _kw(self, text, language, speed, duration, cfg):
        kw = dict(text=text, language=None if language in (None, "", "Auto") else language, generation_config=cfg)
        if duration and float(duration) > 0:
            kw["duration"] = float(duration)
        elif speed and abs(float(speed) - 1) > 1e-3:
            kw["speed"] = float(speed)
        return kw

    TEMP = "__giong-tam__"

    def clone(self, slug, text, *, takes=1, language="Vietnamese", speed=1.0, duration=None, progress=None,
              instruct=None, temp_ref=None, temp_text=None, **cfg):
        """Tạo n bản bằng giọng đã lưu (slug) hoặc giọng tạm (temp_ref: file tải lên, không lưu vào thư viện).
        Mỗi bản: dict(wav, sim, rank, other, other_sim). Xếp bản giống nhất lên đầu."""
        with self.lock:
            lib = self.library_embs()
            dup = None
            if temp_ref:
                pr = self.m.create_voice_clone_prompt(ref_audio=str(temp_ref), ref_text=(temp_text or "").strip() or None,
                                                      preprocess_prompt=bool(cfg.get("preprocess", True)))
                target = similarity.embedding(str(temp_ref))
                # giọng tạm trùng (gần như y hệt) một giọng đã lưu → báo để dùng luôn giọng trong thư viện
                best = max(((float(np.dot(target, e)), s) for s, e in lib.items() if e is not None), default=(0.0, None))
                dup = best[1] if best[0] >= 0.99 else None
                key = self.TEMP
                lib = {**lib, key: target}
            else:
                pr, target, key = self.prompt(slug), self.emb(slug), slug
            kw = self._kw(text, language, speed, duration, self.config(**cfg))
            if instruct and instruct.strip():
                kw["instruct"] = instruct.strip()
            out = []
            for i in range(int(takes)):
                if progress:
                    progress(i / takes, desc=f"Đang tạo bản {i + 1}/{takes}…")
                wav = self.m.generate(voice_clone_prompt=pr, **kw)[0]
                e = similarity.embedding((wav, self.sr))
                pos, other, other_sim = similarity.rank(e, key, lib)
                out.append({"wav": wav, "sim": float(np.dot(e, target)), "rank": pos, "of": len(lib),
                            "other": other, "other_sim": other_sim, "take": i + 1, "dup": dup})
        return sorted(out, key=lambda t: -t["sim"])

    def design(self, text, instruct, *, takes=1, language="Vietnamese", speed=1.0, duration=None, progress=None, **cfg):
        with self.lock:
            kw = self._kw(text, language, speed, duration, self.config(**cfg))
            out = []
            for i in range(int(takes)):
                if progress:
                    progress(i / takes, desc=f"Đang thiết kế bản {i + 1}/{takes}…")
                wav = self.m.generate(instruct=instruct or None, **kw)[0]
                out.append({"wav": wav, "take": i + 1})
        return out

    # ── Lưu ──────────────────────────────────────────────────────────────
    def save_output(self, wav, stem: str, fmt: str = "wav"):
        from .textproc import unique_path
        folder = OUTPUTS / dt.date.today().isoformat()
        path = unique_path(folder, stem, "." + fmt)
        return audio.save(np.asarray(wav, dtype=np.float32), self.sr, path, fmt)

    def make_sample(self, slug: str, sentence: str = TEST_SENTENCE, takes: int = 2) -> float:
        """Bản nghe thử: câu thử chung đọc bằng giọng này (chọn bản giống giọng mẫu nhất). Trả về độ giống."""
        res = self.clone(slug, sentence, takes=takes, num_step=32)
        best = res[0]
        audio.save(np.asarray(best["wav"], dtype=np.float32), self.sr, store.paths(slug)["sample"], "wav")
        store.update(slug, sample_sim=round(best["sim"], 4))
        return best["sim"]

    def add_voice(self, wav: np.ndarray, sr: int, name: str, ref_text: str, make_sample: bool = True, **meta) -> str:
        slug = store.create(wav, sr, name, ref_text, **meta)
        with self.lock:
            self.prompt(slug)
            self.emb(slug)
        if make_sample:
            self.make_sample(slug)
        return slug

    def log(self, mode, slug, name, text, settings, file, wav, sim, take, best):
        history.add(mode, slug, name, text, settings, file, round(len(wav) / self.sr, 2),
                    None if sim is None else round(sim, 4), take, best)
