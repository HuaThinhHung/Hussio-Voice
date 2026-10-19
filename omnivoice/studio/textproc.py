"""Chữ: tên ngắn (slug), từ điển phát âm, ước lượng thời lượng, đặt tên file kết quả."""
import datetime as dt
import json
import re
import unicodedata
from pathlib import Path

from .config import DICT_FILE

DEFAULT_DICT = {
    "cotton": "cốt tông", "oxford": "óc phớt", "slim fit": "sờ lim phít", "form": "phom", "logo": "lô gô",
    "set": "xét", "combo": "com bô", "ykk": "Y Ca Ca", "jacquard": "jắc ca", "polo": "pô lô", "basic": "bây xích",
    "navy": "nây vi", "size": "sai", "sale": "seo", "online": "on lai", "voucher": "vao chơ",
}


def slugify(name: str) -> str:
    s = unicodedata.normalize("NFD", (name or "").lower()).replace("đ", "d")
    s = s.encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")[:48] or "giong"


def load_dict() -> dict:
    try:
        return json.loads(DICT_FILE.read_text(encoding="utf-8"))
    except Exception:
        return dict(DEFAULT_DICT)


def save_dict(table: dict) -> None:
    DICT_FILE.parent.mkdir(parents=True, exist_ok=True)
    DICT_FILE.write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")


def apply_dict(text: str, table: dict | None = None) -> str:
    """Đổi chữ tiếng Anh sang cách đọc tiếng Việt (khớp nguyên từ, không phân biệt hoa thường)."""
    table = load_dict() if table is None else table
    for src in sorted(table, key=len, reverse=True):
        text = re.sub(rf"(?<!\w){re.escape(src)}(?!\w)", table[src], text, flags=re.IGNORECASE)
    return text


def estimate(text: str, syl_per_sec: float = 4.5) -> tuple[int, float]:
    syl = len(re.findall(r"\w+", text or ""))
    pauses = len(re.findall(r"[.!?…]", text or "")) * 0.3 + len(re.findall(r"[,;:]", text or "")) * 0.12
    return syl, syl / syl_per_sec + pauses


def output_name(voice_name: str, text: str, custom: str | None = None) -> str:
    """Tên file (không có đuôi): tên tự đặt, hoặc <giọng>_<ngàygiờ>_<vài chữ đầu>."""
    if custom and custom.strip():
        return slugify(Path(custom.strip()).stem)
    words = "-".join(re.findall(r"\w+", text or "")[:4])
    return f"{slugify(voice_name)}_{dt.datetime.now():%Y%m%d-%H%M%S}_{slugify(words)}"[:90]


def unique_path(folder: Path, stem: str, ext: str) -> Path:
    p = folder / f"{stem}{ext}"
    k = 2
    while p.exists():
        p = folder / f"{stem}_{k}{ext}"
        k += 1
    return p
