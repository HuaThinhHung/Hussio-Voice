"""Thư viện giọng: mỗi giọng một thư mục data/studio/voices/<slug>/

  ref.wav      giọng mẫu đã chuẩn bị (mono 24 kHz)
  meta.json    tên, nhãn, lời mẫu, nguồn, xác nhận đồng ý, điểm nghe thử
  prompt.pt    giọng mẫu đã mã hoá sẵn (VoiceClonePrompt.save), tạo giọng không phải mã hoá lại
  emb.npy      dấu vân giọng (WavLM) để đo độ giống
  sample.wav   bản nghe thử: câu thử chung đọc bằng giọng này
"""
import datetime as dt
import json
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import TRASH, VOICES, ensure_dirs
from .textproc import slugify

FIELDS = ("name", "gender", "age", "region", "style", "notes", "ref_text", "language", "source", "consent",
          "design_instruct", "favorite", "uses", "sample_sim", "duration", "created", "updated")


def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def vdir(slug: str) -> Path:
    return VOICES / slug


def paths(slug: str) -> dict:
    d = vdir(slug)
    return {"dir": d, "ref": d / "ref.wav", "meta": d / "meta.json", "prompt": d / "prompt.pt",
            "emb": d / "emb.npy", "sample": d / "sample.wav"}


def get(slug: str) -> dict | None:
    p = paths(slug)
    try:
        m = json.loads(p["meta"].read_text(encoding="utf-8"))
    except Exception:
        return None
    m["slug"] = slug
    m["has_sample"] = p["sample"].exists()
    return m


def list_voices(gender: str | None = None, query: str | None = None) -> list[dict]:
    ensure_dirs()
    out = []
    for d in VOICES.iterdir():
        if d.is_dir() and (d / "meta.json").exists() and (d / "ref.wav").exists():
            m = get(d.name)
            if not m:
                continue
            if gender and gender not in ("Tất cả", "All") and m.get("gender") != gender:
                continue
            if query and query.strip().lower() not in " ".join(str(m.get(k, "")) for k in ("name", "notes", "style", "region", "age")).lower():
                continue
            out.append(m)
    return sorted(out, key=lambda m: (not m.get("favorite"), m.get("gender", ""), m.get("name", "").lower()))


def choices() -> list[tuple[str, str]]:
    return [(label(m), m["slug"]) for m in list_voices()]


def label(m: dict) -> str:
    tags = " · ".join(x for x in (m.get("gender"), m.get("age"), m.get("region") if m.get("region") != "Chưa rõ" else "") if x)
    star = "★ " if m.get("favorite") else ""
    return f"{star}{m['name']}" + (f"  ({tags})" if tags else "")


def create(audio: np.ndarray, sr: int, name: str, ref_text: str, **meta) -> str:
    """Lưu giọng mới. Trả về slug."""
    ensure_dirs()
    base = slugify(name)
    slug, k = base, 2
    while vdir(slug).exists():
        slug, k = f"{base}-{k}", k + 1
    p = paths(slug)
    p["dir"].mkdir(parents=True)
    sf.write(str(p["ref"]), audio, sr)
    m = {f: None for f in FIELDS}
    m.update(name=name.strip(), ref_text=ref_text.strip(), language="Vietnamese", favorite=False, uses=0,
             duration=round(len(audio) / sr, 2), created=_now(), updated=_now())
    m.update({k: v for k, v in meta.items() if k in FIELDS})
    p["meta"].write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    return slug


def update(slug: str, **fields) -> dict:
    p = paths(slug)
    m = json.loads(p["meta"].read_text(encoding="utf-8"))
    changed_text = "ref_text" in fields and fields["ref_text"] != m.get("ref_text")
    m.update({k: v for k, v in fields.items() if k in FIELDS})
    m["updated"] = _now()
    p["meta"].write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    if changed_text:                                   # lời mẫu đổi thì phải mã hoá lại giọng mẫu
        p["prompt"].unlink(missing_ok=True)
    return get(slug)


def bump_use(slug: str) -> None:
    m = get(slug)
    if m:
        update(slug, uses=int(m.get("uses") or 0) + 1)


def delete(slug: str) -> Path:
    """Xoá mềm: chuyển giọng vào data/studio/_da-xoa (xem / khôi phục / xoá vĩnh viễn ở Thư viện → Đã xoá)."""
    TRASH.mkdir(parents=True, exist_ok=True)
    dst = TRASH / f"{slug}_{dt.datetime.now():%Y%m%d-%H%M%S}"
    p = paths(slug)["meta"]
    try:
        m = json.loads(p.read_text(encoding="utf-8"))
        m["deleted"] = _now()
        p.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
    shutil.move(str(vdir(slug)), str(dst))
    return dst


def list_trash() -> list[dict]:
    """Giọng đã xoá, mới xoá lên đầu: dict(id, name, slug, deleted, gender, age)."""
    out = []
    if TRASH.exists():
        for d in TRASH.iterdir():
            if not d.is_dir():
                continue
            try:
                m = json.loads((d / "meta.json").read_text(encoding="utf-8"))
            except Exception:
                m = {}
            slug, _, stamp = d.name.rpartition("_")
            deleted = m.get("deleted") or (dt.datetime.strptime(stamp, "%Y%m%d-%H%M%S").isoformat() if stamp[:2] == "20" else "")
            out.append({"id": d.name, "name": m.get("name") or slug or d.name, "slug": slug or d.name, "deleted": deleted,
                        "gender": m.get("gender"), "age": m.get("age")})
    return sorted(out, key=lambda x: x["deleted"], reverse=True)


def restore(trash_id: str) -> str:
    """Khôi phục giọng đã xoá về thư viện. Trùng tên thư mục thì thêm hậu tố. Trả về slug."""
    src = TRASH / trash_id
    base = trash_id.rpartition("_")[0] or trash_id
    slug, k = base, 2
    while vdir(slug).exists():
        slug, k = f"{base}-{k}", k + 1
    shutil.move(str(src), str(vdir(slug)))
    p = paths(slug)["meta"]
    try:
        m = json.loads(p.read_text(encoding="utf-8"))
        m.pop("deleted", None)
        m["updated"] = _now()
        p.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
    return slug


def purge(trash_id: str) -> None:
    """Xoá vĩnh viễn một giọng trong mục Đã xoá (chỉ khi người dùng bấm "Xoá vĩnh viễn")."""
    target = (TRASH / trash_id).resolve()
    if target.parent != TRASH.resolve() or not target.is_dir():
        raise ValueError("Không thấy giọng này trong mục Đã xoá.")
    shutil.rmtree(target)


def load_emb(slug: str) -> np.ndarray | None:
    p = paths(slug)["emb"]
    return np.load(p) if p.exists() else None
