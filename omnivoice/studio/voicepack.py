"""Tạo bộ giọng khởi đầu bằng Thiết kế giọng của OmniVoice (giọng tổng hợp, không nhái người thật).

Chạy:  uv --directory D:/Du_An/OmniVoice run python -m omnivoice.studio.voicepack [--them-giong file.wav "Tên" "lời mẫu"]
Giọng đã có (cùng tên) thì bỏ qua.
"""
import argparse
import time

import numpy as np
import torch

from . import audio, store
from .config import DESIGN_REF_SENTENCE
from .engine import Studio

# (tên, giới tính, độ tuổi, mô tả, chỉ dẫn thiết kế cho model)
PACK = [
    ("Nam thanh niên · giọng vừa", "Nam", "Thanh niên", "Ấm, rõ, hợp review sản phẩm", "Male, Young Adult, Moderate Pitch"),
    ("Nam thanh niên · giọng trầm", "Nam", "Thanh niên", "Trầm, chắc", "Male, Young Adult, Low Pitch"),
    ("Nam trung niên · trầm", "Nam", "Trung niên", "Trầm, đĩnh đạc", "Male, Middle-aged, Low Pitch"),
    ("Nam trung niên · rất trầm", "Nam", "Trung niên", "Rất trầm, dày", "Male, Middle-aged, Very Low Pitch"),
    ("Nam cao tuổi", "Nam", "Cao tuổi", "Già dặn, chậm rãi", "Male, Elderly, Low Pitch"),
    ("Nam thì thầm", "Nam", "Thanh niên", "Thì thầm, nhẹ", "Male, Young Adult, Whisper"),
    ("Bé trai", "Nam", "Trẻ em", "Trẻ em", "Male, Child"),
    ("Nữ thanh niên · giọng vừa", "Nữ", "Thanh niên", "Trong, dễ nghe", "Female, Young Adult, Moderate Pitch"),
    ("Nữ thanh niên · giọng cao", "Nữ", "Thanh niên", "Cao, tươi", "Female, Young Adult, High Pitch"),
    ("Nữ thiếu niên", "Nữ", "Thiếu niên", "Trẻ, tươi", "Female, Teenager, High Pitch"),
    ("Nữ trung niên", "Nữ", "Trung niên", "Ấm, điềm đạm", "Female, Middle-aged, Moderate Pitch"),
    ("Nữ cao tuổi", "Nữ", "Cao tuổi", "Hiền, chậm rãi", "Female, Elderly, Moderate Pitch"),
    ("Nữ thì thầm", "Nữ", "Thanh niên", "Thì thầm, nhẹ", "Female, Young Adult, Whisper"),
    ("Bé gái", "Nữ", "Trẻ em", "Trẻ em", "Female, Child"),
]


def build(st: Studio, only_missing: bool = True) -> list[str]:
    have = {m["name"] for m in store.list_voices()}
    made = []
    for name, gender, age, style, instruct in PACK:
        if only_missing and name in have:
            continue
        t = time.time()
        cands = st.design(DESIGN_REF_SENTENCE, instruct, takes=2, num_step=32)
        # chọn bản có tỉ lệ tiếng nói cao, không có khoảng lặng lạ
        best = max(cands, key=lambda c: audio.analyze(np.asarray(c["wav"], np.float32), st.sr)["speech"])
        wav = audio.normalize(audio.trim(np.asarray(best["wav"], np.float32), st.sr))
        slug = st.add_voice(wav, st.sr, name, DESIGN_REF_SENTENCE, gender=gender, age=age, region="Chưa rõ",
                            style=style, source="design", consent=True, design_instruct=instruct,
                            notes="Giọng tổng hợp bằng Thiết kế giọng của OmniVoice, không phải người thật.")
        m = store.get(slug)
        print(f"  + {name}: nghe thử giống giọng mẫu {m['sample_sim']:.3f} ({time.time() - t:.0f}s)")
        made.append(slug)
    return made


def add_file(st: Studio, path: str, name: str, ref_text: str, **meta) -> str:
    wav, sr, info = audio.prepare_ref(path)
    slug = st.add_voice(wav, sr, name, ref_text, **meta)
    print(f"  + {name}: {info['duration']:.1f}s, nghe thử giống giọng mẫu {store.get(slug)['sample_sim']:.3f}")
    return slug


def main():
    ap = argparse.ArgumentParser(description="Tạo bộ giọng khởi đầu cho OmniVoice Studio")
    ap.add_argument("--them-giong", nargs=3, metavar=("FILE", "TEN", "LOI_MAU"), action="append", default=[],
                    help="Thêm một giọng mẫu từ file (người trong file phải đồng ý)")
    ap.add_argument("--bo-qua-bo-giong", action="store_true", help="Không tạo bộ giọng thiết kế")
    a = ap.parse_args()
    from omnivoice import OmniVoice
    m = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda" if torch.cuda.is_available() else "cpu",
                                  dtype=torch.float16, load_asr=False)
    st = Studio(m)
    for path, name, text in a.them_giong:
        add_file(st, path, name, text, source="restore", consent=True, gender="Nam", age="Thanh niên", region="Bắc")
    if not a.bo_qua_bo_giong:
        build(st)
    print(f"Thư viện có {len(store.list_voices())} giọng.")


if __name__ == "__main__":
    main()
