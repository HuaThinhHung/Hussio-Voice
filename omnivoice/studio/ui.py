"""Giao diện HUSSIO Voice (http://localhost:10601), phong cách portfolio tối giản, hai bản Sáng / Tối.
Trang chủ · Tạo giọng · Thư viện giọng (nghe thử) · Thêm giọng · Thiết kế giọng · Lịch sử · Cài đặt & Hướng dẫn."""
import datetime as dt
import json
import os
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf
import torch

from . import audio, history, similarity, store
from .config import (AGES, BALANCED, BEST, DATA, DESIGN_REF_SENTENCE, GENDERS, MODE_AUTO, MODE_CUSTOM, OUTPUTS, PRESETS,
                     REGIONS, SETTINGS_FILE, SOURCES, SPEEDS, TEST_SENTENCE, ensure_dirs)
from .engine import Studio
from .textproc import apply_dict, estimate, load_dict, output_name, save_dict
from .theme import (CSS, TOGGLE_JS, brand_html, empty, foot_html, hero_html, icon, msg, page_head, pill, step,
                    studio_theme)

MAX_TAKES = 4
NAV = [("home", "Trang chủ"), ("gen", "Tạo giọng"), ("lib", "Thư viện"), ("add", "Thêm giọng"), ("design", "Thiết kế"),
       ("hist", "Lịch sử"), ("set", "Cài đặt")]
CARDS = [("gen", "Tạo giọng"), ("lib", "Thư viện"), ("add", "Thêm giọng"), ("design", "Thiết kế"), ("hist", "Lịch sử"),
         ("set", "Cài đặt")]
TAKE_HEAD = ["Bản", "Giống", "Đánh giá", "Xếp hạng", "Dài"]
TMP_SRC = "Giọng tạm (tải file, không lưu)"
HIST_HEAD = ["ID", "Lúc / Time", "Giọng / Voice", "Văn bản / Text", "Độ giống", "Độ dài", "File"]
ALL = "Tất cả"
TRASH_HEAD = ["Mã", "Tên giọng", "Giới tính", "Độ tuổi", "Xoá lúc"]
# cuộn mượt tới form Sửa; trình duyệt không chạy hiệu ứng (tab nền, tắt hiệu ứng) thì nhảy thẳng tới
SCROLL_MANAGE = """() => setTimeout(() => { const el = document.querySelector('#manage'); if (!el) return;
  el.scrollIntoView({behavior: 'smooth', block: 'start'});
  setTimeout(() => { if (Math.abs(el.getBoundingClientRect().top) > 40) el.scrollIntoView({block: 'start'}); }, 700); }, 200)"""
# Gợi ý nhanh ở Thiết kế giọng: (giới tính, độ tuổi, cao độ, phong cách) theo tên tiếng Anh của OmniVoice
QUICK = {
    "Nam trầm ấm": ("Male", "Middle-aged", "Low Pitch", None),
    "Nam trẻ, rõ ràng": ("Male", "Young Adult", "Moderate Pitch", None),
    "Nam lớn tuổi": ("Male", "Elderly", "Low Pitch", None),
    "Nữ trẻ, tươi": ("Female", "Young Adult", "High Pitch", None),
    "Nữ trung niên, ấm": ("Female", "Middle-aged", "Moderate Pitch", None),
    "Thì thầm": (None, None, None, "Whisper"),
}
AGE_VI = {"Child": "Trẻ em", "Teenager": "Thiếu niên", "Young Adult": "Thanh niên", "Middle-aged": "Trung niên",
          "Elderly": "Cao tuổi"}
DESIGN_EXPLAIN = """<div class="explain">
  <div class="ex on"><b>🎨 Thiết kế giọng</b> <span>(trang này)</span><br>AI <b>tự tạo một giọng mới</b> theo mô tả: nam/nữ, độ tuổi,
  trầm/cao. <b>Không dùng giọng người thật.</b></div>
  <div class="ex"><b>➕ Thêm giọng</b><br>Đưa <b>file ghi âm giọng thật</b> (người đó đã đồng ý). AI bắt chước giọng đó và lưu vào
  thư viện để dùng mãi.</div>
  <div class="ex"><b>⚡ Giọng tạm</b> <span>(ở Tạo giọng)</span><br>Giống Thêm giọng nhưng <b>dùng một lần</b>, không lưu vào thư viện.</div>
</div>"""


def _settings() -> dict:
    try:
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_settings(**kw) -> None:
    s = _settings() | kw
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")


def _pct(x) -> str:
    return "—" if x is None else f"{x * 100:.1f}%".replace(".", ",")


def _secs(x) -> str:
    return f"{x:.1f}s".replace(".", ",")


def _gpu() -> str:
    if not torch.cuda.is_available():
        return "CPU"
    free, total = torch.cuda.mem_get_info()
    return f"GPU {(total - free) / 2**30:.1f}/{total / 2**30:.0f} GB".replace(".", ",")


def header_html() -> str:
    """Dòng trạng thái nhỏ trên thanh menu."""
    return f'<div class="status">{len(store.list_voices())} giọng · {_gpu()}</div>'


def voice_card(m: dict) -> str:
    g = m.get("gender")
    gcls = "m" if g == "Nam" else ("f" if g == "Nữ" else "")
    tags = "".join(f'<span class="tag {c}">{t}</span>' for t, c in
                   ((g, gcls), (m.get("age"), ""), (m.get("region") if m.get("region") not in (None, "Chưa rõ") else None, ""),
                    (SOURCES.get(m.get("source"), m.get("source")), "src")) if t)
    s = m.get("sample_sim")
    lab, kind = similarity.label(s) if s else ("Chưa có bản nghe thử", "warn")
    sim = f'<div class="vsim {kind}">Bản nghe thử giống giọng mẫu: <b>{_pct(s)}</b> · {lab.split(" / ")[0]}</div>' if s else ""
    star = "★ " if m.get("favorite") else ""
    used = f" · đã dùng {m.get('uses') or 0} lần" if m.get("uses") else ""
    return (f'<div class="vname">{star}{m["name"]}</div><div class="vtags">{tags}</div>'
            f'<div class="vnote">{m.get("style") or ""}{used}</div>{sim}')


def trash_rows() -> list:
    """Bảng mục Đã xoá: mã, tên, giới tính, độ tuổi, lúc xoá."""
    return [[t["id"], t["name"], t.get("gender") or "", t.get("age") or "", (t["deleted"] or "")[:16].replace("T", " ")]
            for t in store.list_trash()]


def go(target: str) -> list:
    """Mở trang `target` và tô sáng mục tương ứng trên thanh menu (thứ tự: tabs, rồi các nút NAV)."""
    return [gr.Tabs(selected=target)] + [gr.update(variant="primary" if k == target else "secondary") for k, _ in NAV]


def voice_info(slug) -> tuple[str, str | None]:
    m = store.get(slug) if slug else None
    if not m:
        return empty("Chưa chọn giọng. Thêm giọng ở thẻ Thêm giọng hoặc Thiết kế giọng."), None
    p = store.paths(slug)
    return voice_card(m), (str(p["sample"]) if p["sample"].exists() else str(p["ref"]))


# ── Cách tạo giọng: app đề xuất theo độ dài văn bản và giọng đang dùng ────────
def recommend(sec: float, m: dict | None, temp: bool) -> tuple[str, str]:
    """Cách tạo app đề xuất (khoá trong PRESETS) và lý do."""
    if temp:
        return BEST, "Giọng tạm chưa được kiểm tra trước, nên tạo 3 bản rồi chọn bản giống giọng mẫu nhất."
    sim = (m or {}).get("sample_sim")
    if sec <= 20:
        return BEST, f"Đoạn ngắn (khoảng {_secs(sec)}): tạo 3 bản để chọn bản giống nhất mà vẫn chờ không lâu."
    if sec <= 60:
        if sim is not None and sim < 0.92:
            return BEST, f"Giọng này khó bắt chước hơn (bản nghe thử giống {_pct(sim)}): nên tạo 3 bản để chọn."
        good = f", giọng này bắt chước tốt ({_pct(sim)})" if sim else ""
        return BALANCED, f"Đoạn vừa (khoảng {_secs(sec)}){good}: 1 bản là đủ."
    return BALANCED, f"Đoạn dài (khoảng {_secs(sec)}): tạo 1 bản cho đỡ chờ. Câu nào chưa ổn thì tạo lại riêng câu đó."


def plan(mode, text, use_d, slug, src, takes, steps, gs) -> dict:
    """Thông số sẽ dùng khi bấm TẠO GIỌNG. Tính lại ở máy chủ nên luôn khớp cách tạo đang chọn."""
    temp = src == TMP_SRC
    t = apply_dict(text or "") if use_d else (text or "")
    syl, sec = estimate(t)
    rec, why = recommend(sec, None if temp or not slug else store.get(slug), temp)
    out = dict(rec=rec, why=why, syl=syl, sec=sec)
    if mode == MODE_CUSTOM:
        return out | dict(key=MODE_CUSTOM, takes=int(takes), steps=int(steps), gs=float(gs))
    key = rec if mode == MODE_AUTO else (mode if mode in PRESETS else BALANCED)
    p = PRESETS[key]
    return out | dict(key=key, takes=p["takes"], steps=p["num_step"], gs=p["guidance_scale"])


def rec_html(mode, p: dict, speed, fmt) -> str:
    """Khung "App đề xuất" ở bước 3: cách tạo, lý do, các thông số sẽ dùng."""
    if mode == MODE_AUTO and not p["syl"]:
        return ('<div class="rec"><div class="h">✨ Để app chọn</div><div class="why">Nhập văn bản, app sẽ chọn cách tạo '
                'hợp với độ dài đoạn và giọng đang dùng.</div></div>')
    q = "chất lượng " + ("cao" if p["steps"] >= 40 else "tốt" if p["steps"] >= 24 else "nháp")
    sp = next((lab for lab, v in SPEEDS if abs(v - float(speed or 1)) < 1e-6), f"{speed}×")
    pills = "".join(pill(x) for x in (f"{p['takes']} bản", f"{q} ({p['steps']} bước)", f"tốc độ {sp.lower()}",
                                      (fmt or "wav").upper()))
    if mode == MODE_AUTO:
        head, why, alt = f"✨ App chọn: <b>{p['key']}</b>", p["why"], f'<div class="desc">{PRESETS[p["key"]]["desc"]}</div>'
    else:
        head = f"Đang dùng: <b>{mode}</b>"
        why = PRESETS[mode]["desc"] if mode in PRESETS else "Dùng đúng các thông số bạn chỉnh trong mục Tuỳ chỉnh chi tiết."
        alt = (f'<div class="alt">App gợi ý: <b>{p["rec"]}</b>. {p["why"]}</div>' if p["syl"] and p["rec"] != mode else "")
    return f'<div class="rec"><div class="h">{head}</div><div class="why">{why}</div>{alt}<div class="set">{pills}</div></div>'


def build_studio(model) -> gr.Blocks:
    from omnivoice.cli.demo import _ALL_LANGUAGES, _ATTR_INFO, _AUTO_LABEL, _CATEGORIES, _VI
    ensure_dirs()
    tam = DATA / "_tam"                                    # file tạm (giọng mẫu đang kiểm tra, phương án thiết kế)
    if tam.exists():
        old = dt.datetime.now().timestamp() - 2 * 86400
        for f in tam.glob("*.wav"):
            if f.stat().st_mtime < old:
                f.unlink(missing_ok=True)
    st = Studio(model)
    langs = [(_AUTO_LABEL, "Auto")] + _ALL_LANGUAGES[1:]
    first = store.choices()
    first_slug = first[0][1] if first else None
    fmt_default = _settings().get("format", "wav")

    with gr.Blocks(title="HUSSIO Voice") as app:
        # ── Thanh menu trên cùng: một app, mọi chức năng ở một chỗ ────────────
        with gr.Row(elem_id="topnav", equal_height=True):
            with gr.Column(scale=0, min_width=140):
                brand = gr.HTML(brand_html(), elem_id="brand")      # bấm logo = về Trang chủ
            with gr.Column(scale=1, min_width=0):
                with gr.Row(elem_id="navpills"):
                    nav = {k: gr.Button(lbl, size="sm", elem_classes="navpill",
                                        variant="primary" if k == "home" else "secondary") for k, lbl in NAV}
            with gr.Column(scale=0, min_width=50):
                theme_btn = gr.Button("☾", elem_id="themebtn", size="sm")
        lib_ver = gr.State(0)
        with gr.Tabs(elem_id="main-tabs", selected="home") as tabs:
            # ══ 0. Trang chủ (kiểu portfolio) ══════════════════════════════
            with gr.Tab("Trang chủ", id="home"):
                gr.HTML(hero_html())
                with gr.Row(elem_id="ask", equal_height=True):
                    home_text = gr.Textbox(show_label=False, container=False, lines=1, max_lines=4, scale=8,
                                           placeholder="Nhập câu cần đọc rồi bấm ↑ …")
                    home_send = gr.Button("↑", elem_id="send", variant="primary", scale=0, min_width=44)
                home_hint = gr.HTML()
                with gr.Row(elem_id="qcards"):
                    cards = {k: gr.Button(lbl, icon=icon(k), elem_classes="qcard") for k, lbl in CARDS}
                gr.HTML('<div class="wm">HUSSIO</div>')

            # ══ 1. Tạo giọng ═══════════════════════════════════════════════
            with gr.Tab("Tạo giọng / Generate", id="gen"):
                gr.HTML(page_head("Tạo giọng", "Dán văn bản, chọn giọng, bấm tạo. Bản ★ là bản giống giọng mẫu nhất."))
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5, min_width=520):
                        with gr.Column(elem_classes="card"):
                            gr.HTML(step("1", "Văn bản / Text", "Nội dung cần đọc. Viết số thành chữ để đọc chuẩn."))
                            g_text = gr.Textbox(show_label=False, lines=6, placeholder="Nhập nội dung cần đọc... / Enter text...")
                            g_est = gr.HTML(pill("Chưa có văn bản", ""))
                            g_dict = gr.Checkbox(value=True, label="Đổi chữ tiếng Anh sang cách đọc / Pronunciation dictionary (cotton → cốt tông)")
                        with gr.Column(elem_classes="card"):
                            gr.HTML(step("2", "Giọng / Voice", "Chọn giọng trong thư viện, hoặc tải một giọng mẫu dùng ngay (không lưu)."))
                            g_src = gr.Radio(["Giọng trong thư viện", TMP_SRC], value="Giọng trong thư viện", show_label=False)
                            g_voice = gr.Dropdown(choices=first, value=first_slug, show_label=False)
                            g_vinfo = gr.HTML(voice_info(first_slug)[0])
                            g_vsample = gr.Audio(value=voice_info(first_slug)[1], label="Nghe thử giọng này / Sound test",
                                                 type="filepath", interactive=False, buttons=["download"])
                            g_tmp_audio = gr.Audio(sources=["upload", "microphone"], type="filepath", visible="hidden",
                                                   label="Giọng mẫu tạm / Temporary reference (3–10 giây)")
                            g_tmp_text = gr.Textbox(label="Lời trong giọng mẫu / Reference text", lines=2, visible="hidden",
                                                    placeholder="Để trống thì máy tự nghe và chép lời")
                            g_tmp_ok = gr.Checkbox(label="Người trong file đã đồng ý cho dùng giọng này / Speaker consent given",
                                                   visible="hidden")
                        with gr.Column(elem_classes="card"):
                            gr.HTML(step("3", "Cách tạo / How", "Không chắc thì cứ để app chọn: app xem độ dài văn bản và giọng đang dùng."))
                            g_mode = gr.Radio([MODE_AUTO, *PRESETS, MODE_CUSTOM], value=MODE_AUTO, show_label=False,
                                              elem_id="gmode")
                            g_rec = gr.HTML(rec_html(MODE_AUTO, plan(MODE_AUTO, "", True, first_slug, "", 1, 32, 2.0), 1.0,
                                                     fmt_default))
                            with gr.Row():
                                g_speed = gr.Radio(SPEEDS, value=1.0, label="Tốc độ đọc / Speed")
                                g_fmt = gr.Radio([("WAV · dựng video", "wav"), ("MP3 · gửi nhanh", "mp3")], value=fmt_default,
                                                 label="Định dạng file / Format")
                            with gr.Accordion("🛠 Tuỳ chỉnh chi tiết (chỉ khi cần) / Advanced", open=False):
                                gr.HTML('<div class="tips">Kéo 3 thanh đầu thì cách tạo tự chuyển sang <b>🛠 Tự chỉnh</b>. '
                                        'Bấm lại <b>✨ Để app chọn</b> để quay về đề xuất.</div>')
                                with gr.Row():
                                    g_takes = gr.Slider(1, MAX_TAKES, value=PRESETS[BEST]["takes"], step=1,
                                                        label="Số bản / Takes", info="Nhiều bản để chọn bản giống nhất")
                                    g_steps = gr.Slider(4, 64, value=PRESETS[BEST]["num_step"], step=1,
                                                        label="Số bước / Steps", info="Cao = sạch hơn, chậm hơn")
                                with gr.Row():
                                    g_gs = gr.Slider(0.0, 4.0, value=2.0, step=0.1, label="Mức bám / Guidance (CFG)",
                                                     info="Mặc định 2,0. Ảnh hưởng rất ít đến độ giống")
                                    g_dur = gr.Number(value=None, label="Độ dài (giây) / Duration",
                                                      info="0 = theo tốc độ đọc")
                                with gr.Row():
                                    g_lang = gr.Dropdown(langs, value="Vietnamese", label="Ngôn ngữ / Language")
                                    g_name = gr.Textbox(label="Tên file / File name", placeholder="Để trống = tên giọng + giờ")
                                with gr.Row():
                                    g_dn = gr.Checkbox(value=True, label="Lọc tạp âm / Denoise")
                                    g_pre = gr.Checkbox(value=True, label="Xử lý giọng mẫu / Preprocess")
                                    g_post = gr.Checkbox(value=True, label="Bỏ lặng dài / Postprocess")
                                g_instruct = gr.Textbox(label="Chỉ dẫn / Instruct (không bắt buộc)", lines=1,
                                                        placeholder="VD: Low Pitch, Whisper… (tiếng Anh, như ô Instruct của OmniVoice gốc)")
                        g_btn = gr.Button("TẠO GIỌNG / GENERATE", variant="primary", elem_classes="btn-main")
                    with gr.Column(scale=4, min_width=420):
                        with gr.Column(elem_classes="card"):
                            gr.HTML(step("✓", "Kết quả / Output", "Bản ★ là bản giống giọng mẫu nhất. File tự lưu."))
                            g_msg = gr.HTML(empty("Kết quả hiện ở đây sau khi bấm “TẠO GIỌNG”."))
                            g_outs = [gr.Audio(type="filepath", visible="hidden", interactive=False, buttons=["download"])
                                      for _ in range(MAX_TAKES)]
                            g_table = gr.Dataframe(value=[], headers=TAKE_HEAD, interactive=False, wrap=True, buttons=[],
                                                   visible="hidden")
                        gr.HTML("""<div class="tips"><b>Mẹo để giọng ra giống giọng mẫu hơn</b><br>
                        1. Giọng mẫu sạch: 6–10 giây, một người nói, không nhạc nền, không vang.<br>
                        2. Lời mẫu gõ đúng từng chữ (Thư viện → ✎ Sửa).<br>
                        3. Viết số thành chữ; câu cần đọc cùng ngôn ngữ với giọng mẫu.<br>
                        4. Nghe chưa ưng thì bấm <b>TẠO GIỌNG</b> lần nữa: mỗi lần ra bản hơi khác.</div>""")

            # ══ 2. Thư viện giọng ══════════════════════════════════════════
            with gr.Tab("Thư viện giọng / Voice Library", id="lib"):
                gr.HTML(page_head("Thư viện giọng", "Nghe thử từng giọng trên cùng một câu. Bấm Dùng để chọn giọng."))
                with gr.Row(equal_height=True):
                    l_gender = gr.Radio([ALL, "Nam", "Nữ"], value=ALL, label="Giới tính / Gender", scale=2)
                    l_q = gr.Textbox(label="Tìm / Search", placeholder="tên, độ tuổi, kiểu giọng…", scale=3)
                    l_refresh = gr.Button("Làm mới / Refresh", variant="secondary", scale=1)
                gr.HTML('<div class="tips">Mỗi giọng có <b>bản nghe thử</b> đọc cùng một câu để so giọng với nhau. '
                        '<b>Dùng giọng này</b>: chọn giọng để tạo. <b>✎ Sửa</b>: đổi tên, nhãn, lời mẫu. '
                        '<b>🗑 Xoá</b>: chuyển vào mục <b>Đã xoá</b> ở cuối trang, khôi phục hoặc xoá vĩnh viễn ở đó.</div>')

                @gr.render(inputs=[l_gender, l_q, lib_ver])
                def gallery(gender, q, _ver):
                    vs = store.list_voices(gender, q)
                    if not vs:
                        gr.HTML(empty("Không có giọng nào khớp. Thêm giọng ở thẻ Thêm giọng / Thiết kế giọng."))
                        return
                    for i in range(0, len(vs), 3):
                        with gr.Row(equal_height=True, elem_classes="vrow"):
                            for m in vs[i:i + 3]:
                                p = store.paths(m["slug"])
                                with gr.Column(elem_classes="vcard", min_width=240):
                                    gr.HTML(voice_card(m))
                                    gr.Audio(value=str(p["sample"] if p["sample"].exists() else p["ref"]),
                                             label="Nghe thử / Sound test", type="filepath", interactive=False,
                                             buttons=["download"])
                                    b_use = gr.Button("Dùng giọng này", size="sm", variant="primary", elem_classes="vuse")
                                    with gr.Row(elem_classes="vacts"):
                                        b_edit = gr.Button("✎ Sửa", size="sm", elem_classes="vbtn")
                                        b_del = gr.Button("🗑 Xoá", size="sm", elem_classes="vbtn del")
                                        b_fav = gr.Button("★ Bỏ ghim" if m.get("favorite") else "☆ Ghim", size="sm",
                                                          elem_classes="vbtn")
                                    b_use.click(_use(m["slug"]), None, [g_voice, tabs, *nav.values()])
                                    b_fav.click(_fav(m["slug"]), [lib_ver, g_voice], [lib_ver, g_voice])
                                    b_edit.click(_edit(m["slug"]), None, [l_sel, l_acc]).then(None, None, None, js=SCROLL_MANAGE)
                                    b_del.click(_del(m["slug"]), [lib_ver, g_voice],
                                                [lib_ver, g_voice, l_sel, h_voice, hdr, t_table])

                with gr.Accordion("✎ Sửa giọng / Edit voice", open=False, elem_id="manage") as l_acc:
                    with gr.Row(equal_height=False):
                        with gr.Column(elem_classes="card"):
                            l_sel = gr.Dropdown(choices=first, value=None, label="Giọng / Voice")
                            l_ref = gr.Audio(label="Giọng mẫu gốc / Reference", type="filepath", interactive=False,
                                             buttons=["download"])
                            l_reftext = gr.Textbox(label="Lời trong giọng mẫu / Reference text", lines=2,
                                                   info="Phải khớp từng chữ với giọng mẫu. Sửa xong bấm Lưu.")
                        with gr.Column(elem_classes="card"):
                            l_name = gr.Textbox(label="Tên giọng / Name")
                            with gr.Row():
                                l_gen = gr.Radio(GENDERS, label="Giới tính / Gender")
                                l_reg = gr.Radio(REGIONS, label="Vùng miền / Region")
                            l_age = gr.Dropdown(AGES, label="Độ tuổi / Age")
                            l_style = gr.Textbox(label="Kiểu giọng / Style", placeholder="VD: trầm ấm, hợp review")
                            l_notes = gr.Textbox(label="Ghi chú / Notes", lines=2)
                            with gr.Row():
                                l_save = gr.Button("Lưu / Save", variant="primary")
                                l_resample = gr.Button("Tạo lại bản nghe thử / Remake sound test")
                            l_del = gr.Button("🗑 Xoá giọng này (chuyển vào Đã xoá)", variant="stop")
                            l_msg = gr.HTML()

                # ── Đã xoá: khôi phục hoặc xoá vĩnh viễn ──
                with gr.Accordion("🗑 Đã xoá / Deleted voices", open=False, elem_id="trash"):
                    with gr.Column(elem_classes="card"):
                        gr.HTML('<div class="tips">Giọng đã xoá nằm ở đây, chưa mất. Bấm một dòng rồi chọn <b>Khôi phục</b> để '
                                'đưa về thư viện, hoặc <b>Xoá vĩnh viễn</b> nếu chắc chắn không dùng nữa.</div>')
                        t_table = gr.Dataframe(value=trash_rows, headers=TRASH_HEAD, interactive=False, type="array",
                                               wrap=True, buttons=[], max_height=320,
                                               column_widths=["26%", "30%", "13%", "13%", "18%"])
                        t_sel = gr.State(None)
                        t_info = gr.HTML(empty("Chưa chọn giọng nào."))
                        with gr.Row(equal_height=True):
                            t_restore = gr.Button("↩ Khôi phục / Restore", variant="primary")
                            t_ok = gr.Checkbox(label="Tôi hiểu: xoá vĩnh viễn thì không lấy lại được")
                            t_purge = gr.Button("Xoá vĩnh viễn / Delete forever", variant="stop")

            # ══ 3. Thêm giọng ══════════════════════════════════════════════
            with gr.Tab("Thêm giọng / Add Voice", id="add"):
                gr.HTML(page_head("Thêm giọng", "Nạp giọng mẫu 6–10 giây, đặt tên, lưu vào thư viện để dùng mãi."))
                a_prep = gr.State(None)
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=440):
                        with gr.Column(elem_classes="card"):
                            gr.HTML(step("1", "Giọng mẫu / Reference audio",
                                         "Tải file hoặc ghi âm 6–10 giây, một người nói, không nhạc nền. File dài: app tự chọn đoạn tốt nhất."))
                            a_audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Giọng mẫu / Reference")
                            a_check = gr.Button("KIỂM TRA & CHUẨN BỊ / ANALYZE", variant="primary")
                            a_report = gr.HTML()
                            a_prepared = gr.Audio(label="Đoạn sẽ dùng làm mẫu / Prepared reference", type="filepath",
                                                  interactive=False)
                            a_text = gr.Textbox(label="Lời trong giọng mẫu / Reference text", lines=2,
                                                info="Máy tự chép lời; sửa lại cho đúng từng chữ để giọng ra giống hơn.")
                    with gr.Column(scale=1, min_width=440):
                        with gr.Column(elem_classes="card"):
                            gr.HTML(step("2", "Thông tin giọng / Details", "Đặt tên và nhãn để dễ tìm trong thư viện."))
                            a_name = gr.Textbox(label="Tên giọng / Name", placeholder="VD: Minh · nam, giọng Bắc")
                            with gr.Row():
                                a_gen = gr.Radio(GENDERS, value="Nam", label="Giới tính / Gender")
                                a_reg = gr.Radio(REGIONS, value="Bắc", label="Vùng miền / Region")
                            a_age = gr.Dropdown(AGES, value="Thanh niên", label="Độ tuổi / Age")
                            a_style = gr.Textbox(label="Kiểu giọng / Style", placeholder="VD: trầm ấm, đọc chậm")
                            a_consent = gr.Checkbox(label="Người trong file đã đồng ý cho dùng giọng này / Speaker consent given")
                            a_save = gr.Button("LƯU VÀO THƯ VIỆN / SAVE TO LIBRARY", variant="primary", elem_classes="btn-main")
                            a_msg = gr.HTML()
                            a_sample = gr.Audio(label="Bản nghe thử giọng vừa lưu / Sound test", type="filepath",
                                                interactive=False, visible="hidden")

            # ══ 4. Thiết kế giọng ══════════════════════════════════════════
            with gr.Tab("Thiết kế giọng / Voice Design", id="design"):
                gr.HTML(page_head("Thiết kế giọng", "AI tự tạo một giọng MỚI theo mô tả của bạn. Không cần file giọng người thật."))
                gr.HTML(DESIGN_EXPLAIN)
                d_state = gr.State(None)
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=440):
                        with gr.Column(elem_classes="card"):
                            gr.HTML(step("1", "Chọn kiểu giọng / Describe the voice",
                                         "Bấm một gợi ý nhanh, hoặc tự chỉnh từng đặc điểm. Để “Tự động” thì AI tự chọn."))
                            d_quick = gr.Radio(list(QUICK), value=None, label="Gợi ý nhanh / Quick picks")
                            d_attrs = []
                            cats = list(_CATEGORIES.items())
                            for c, ch in cats[:4]:
                                d_attrs.append(gr.Dropdown(label=_VI.get(c, c), value="Auto", info=_ATTR_INFO.get(c),
                                                           choices=[(_AUTO_LABEL, "Auto")] + [(_VI.get(x, x), x) for x in ch]))
                            with gr.Accordion("Nâng cao: giọng nước ngoài (chỉ có tác dụng khi đọc tiếng Anh / tiếng Trung)",
                                              open=False):
                                for c, ch in cats[4:]:
                                    d_attrs.append(gr.Dropdown(label=_VI.get(c, c), value="Auto", info=_ATTR_INFO.get(c),
                                                               choices=[(_AUTO_LABEL, "Auto")] + [(_VI.get(x, x), x) for x in ch]))
                            d_text = gr.Textbox(value=DESIGN_REF_SENTENCE, label="Câu đọc mẫu / Text", lines=3,
                                                info="Câu này thành lời mẫu của giọng khi lưu. Nên dài 6–10 giây.")
                            d_takes = gr.Slider(1, MAX_TAKES, value=3, step=1, label="Số phương án / Variants")
                            d_btn = gr.Button("THIẾT KẾ / DESIGN", variant="primary", elem_classes="btn-main")
                    with gr.Column(scale=1, min_width=440):
                        with gr.Column(elem_classes="card"):
                            gr.HTML(step("2", "Nghe và lưu / Listen & save",
                                         "Mỗi lần bấm, AI tạo giọng hơi khác nhau. Ưng phương án nào thì đặt tên và lưu: "
                                         "giọng đã lưu sẽ giữ nguyên mỗi lần dùng."))
                            d_msg = gr.HTML(empty("Chọn đặc điểm rồi bấm THIẾT KẾ."))
                            d_outs = [gr.Audio(type="filepath", visible="hidden", interactive=False) for _ in range(MAX_TAKES)]
                            d_pick = gr.Radio([], label="Chọn phương án / Pick variant")
                            d_name = gr.Textbox(label="Tên giọng / Name", placeholder="VD: Nam trầm ấm 2")
                            with gr.Row():
                                d_gen = gr.Radio(GENDERS, value="Nam", label="Giới tính / Gender")
                                d_age = gr.Dropdown(AGES, value="Thanh niên", label="Độ tuổi / Age")
                            d_save = gr.Button("LƯU VÀO THƯ VIỆN / SAVE TO LIBRARY", variant="primary")
                            d_saved = gr.HTML()

            # ══ 5. Lịch sử ═════════════════════════════════════════════════
            with gr.Tab("Lịch sử / History", id="hist"):
                gr.HTML(page_head("Lịch sử", "Mọi lần tạo giọng. Bấm một dòng để nghe lại."))
                with gr.Row(equal_height=True):
                    h_voice = gr.Dropdown([(ALL, "")] + first, value="", label="Giọng / Voice", scale=3)
                    h_refresh = gr.Button("Làm mới / Refresh", variant="secondary", scale=1)
                    h_open = gr.Button("Mở thư mục kết quả / Open folder", variant="secondary", scale=1)
                h_table = gr.Dataframe(value=[], headers=HIST_HEAD, interactive=False, wrap=True, type="array", buttons=[],
                                       max_height=460, column_widths=["6%", "12%", "18%", "34%", "10%", "8%", "12%"])
                with gr.Row(equal_height=False):
                    h_audio = gr.Audio(label="Nghe lại / Play", type="filepath", interactive=False, buttons=["download"])
                    with gr.Column():
                        h_info = gr.HTML(empty("Bấm một dòng để nghe lại."))
                        h_reuse = gr.Button("Đưa văn bản & giọng sang Tạo giọng / Reuse")
                        h_del = gr.Button("Xoá khỏi lịch sử (giữ file) / Remove from history")
                h_sel = gr.State(None)

            # ══ 6. Cài đặt & Hướng dẫn ═════════════════════════════════════
            with gr.Tab("Cài đặt & Hướng dẫn / Settings & Guide", id="set"):
                gr.HTML(page_head("Cài đặt & Hướng dẫn", "Từ điển phát âm, định dạng file, câu nghe thử, cách dùng."))
                with gr.Row(equal_height=False):
                    with gr.Column(elem_classes="card"):
                        gr.HTML(step("Aa", "Từ điển phát âm / Pronunciation",
                                     "Cột trái: chữ gốc. Cột phải: cách đọc. Áp dụng khi tích ô ở thẻ Tạo giọng."))
                        s_dict = gr.Dataframe(value=lambda: [[k, v] for k, v in load_dict().items()],
                                              headers=["Chữ gốc / Source", "Đọc thành / Read as"], interactive=True,
                                              type="array", column_count=2, buttons=[])
                        s_dict_save = gr.Button("Lưu từ điển / Save", variant="primary")
                        s_dict_msg = gr.HTML()
                    with gr.Column(elem_classes="card"):
                        gr.HTML(step("⚙", "Cài đặt / Settings"))
                        s_fmt = gr.Radio(["wav", "mp3"], value=fmt_default, label="Định dạng mặc định / Default format")
                        s_test = gr.Textbox(value=_settings().get("test_sentence", TEST_SENTENCE), lines=2,
                                            label="Câu nghe thử / Sound-test sentence")
                        s_save = gr.Button("Lưu cài đặt / Save", variant="primary")
                        s_remake = gr.Button("Tạo lại bản nghe thử cho mọi giọng / Remake all sound tests")
                        s_open = gr.Button("Mở thư mục dữ liệu / Open data folder")
                        s_msg = gr.HTML()
                with gr.Column(elem_classes="card guide"):
                    gr.Markdown(GUIDE)

        gr.HTML(foot_html())
        hdr = gr.HTML(header_html(), elem_id="footstatus")

        # ── Xử lý: Tạo giọng ────────────────────────────────────────────────
        def refresh(text, use_d, slug, src, mode, takes, steps, gs, speed, fmt):
            """Ước tính độ dài + khung đề xuất; cách tạo khác Tự chỉnh thì đặt luôn các thanh tuỳ chỉnh cho khớp."""
            p = plan(mode, text, use_d, slug, src, takes, steps, gs)
            sec = p["sec"] / float(speed or 1)
            est = pill(f"<b>{p['syl']}</b> âm tiết · đọc khoảng <b>{_secs(sec)}</b>") if p["syl"] else pill("Chưa có văn bản")
            sliders = ([gr.update()] * 3 if mode == MODE_CUSTOM else
                       [gr.update(value=p["takes"]), gr.update(value=p["steps"]), gr.update(value=p["gs"])])
            return est, rec_html(mode, p, speed, fmt), *sliders

        def on_voice(slug):
            return voice_info(slug)

        def generate(slug, text, use_d, takes, lang, name, fmt, steps, gs, speed, dur, dn, pre, post,
                     src, tmp_audio, tmp_text, tmp_ok, instruct, mode, progress=gr.Progress()):
            hide = {o: gr.update(value=None, visible="hidden") for o in g_outs}
            temp = src == TMP_SRC
            p = plan(mode, text, use_d, slug, src, takes, steps, gs)
            takes, steps, gs = p["takes"], p["steps"], p["gs"]
            if temp:
                if not tmp_audio:
                    return {g_msg: msg("Chưa tải giọng mẫu tạm.", "bad"), **hide}
                if not tmp_ok:
                    return {g_msg: msg("Tích ô xác nhận người trong file đã đồng ý cho dùng giọng.", "bad"), **hide}
                m = {"name": "giong-tam"}
                slug = st.TEMP
            elif not slug or not store.get(slug):
                return {g_msg: msg("Chưa chọn giọng.", "bad"), **hide}
            else:
                m = store.get(slug)
            if not (text or "").strip():
                return {g_msg: msg("Chưa có văn bản cần đọc.", "bad"), **hide}
            t = apply_dict(text) if use_d else text
            progress(0, desc="Đang chuẩn bị giọng mẫu…")
            try:
                res = st.clone(None if temp else slug, t.strip(), takes=int(takes), language=lang, speed=speed, duration=dur,
                               progress=progress, instruct=instruct, temp_ref=tmp_audio if temp else None,
                               temp_text=tmp_text, num_step=steps, guidance_scale=gs, denoise=dn, preprocess=pre,
                               postprocess=post)
            except Exception as e:
                return {g_msg: msg(f"Lỗi khi tạo giọng: {type(e).__name__}: {e}", "bad"), **hide}
            stem = output_name(m["name"], text, name)
            settings = dict(mode=p["key"], takes=int(takes), language=lang, num_step=steps, guidance_scale=gs, speed=speed,
                            duration=dur, denoise=dn, preprocess=pre, postprocess=post, dictionary=use_d,
                            instruct=instruct or None, temp_voice=temp)
            out, rows, best_file = {}, [], None
            for i, r in enumerate(res):
                f = st.save_output(r["wav"], stem if i == 0 else f"{stem}_ban{i + 1}", fmt)
                best_file = best_file or f
                st.log("clone", slug, "Giọng tạm" if temp else m["name"], text, settings, f, r["wav"], r["sim"], r["take"],
                       i == 0)
                lab, _ = similarity.label(r["sim"])
                rk = f"{r['rank']}/{r['of']}"
                out[g_outs[i]] = gr.update(value=str(f), visible=True,
                                           label=f"{'★ ' if i == 0 else ''}Bản {r['take']} · giống {_pct(r['sim'])} · {_secs(len(r['wav']) / st.sr)}")
                rows.append([("★ " if i == 0 else "") + f"Bản {r['take']}", _pct(r["sim"]), lab.split(" / ")[0], rk,
                             _secs(len(r["wav"]) / st.sr)])
            for o in g_outs[len(res):]:
                out[o] = gr.update(value=None, visible="hidden")
            if not temp:
                store.bump_use(slug)
            b = res[0]
            lab, kind = similarity.label(b["sim"])
            note = ""
            if temp and b.get("dup"):
                dm = store.get(b["dup"])
                note = (f"<br><span class='small'>Giọng tạm này trùng giọng “{dm['name'] if dm else b['dup']}” đã có trong "
                        f"thư viện: lần sau chọn luôn giọng đó cho nhanh.</span>")
            elif b["rank"] != 1 and b["other"]:
                om = store.get(b["other"])
                note = (f"<br><span class='small'>Bản này nghe gần giọng “{om['name'] if om else 'khác'}” "
                        f"({_pct(b['other_sim'])}) hơn giọng mẫu. Thử tạo thêm bản hoặc dùng giọng mẫu dài, sạch hơn.</span>")
            again = "" if note else "<br><span class='small'>Nghe chưa ưng? Bấm TẠO GIỌNG lần nữa: mỗi lần ra bản hơi khác.</span>"
            out[g_msg] = msg(f"Xong {len(res)} bản ({p['key']}). Bản giống nhất: <b>{_pct(b['sim'])}</b> "
                             f"({lab.split(' / ')[0]}). Đã lưu: <b>{Path(best_file).name}</b>"
                             f"<br><span class='small'>Thư mục: {OUTPUTS / dt.date.today().isoformat()}</span>{note}{again}",
                             kind)
            out[g_table] = gr.update(value=rows, visible=True)
            out[hdr] = header_html()
            return out

        def home_hint_html(slug):
            m = store.get(slug) if slug else None
            return (f'<div class="askhint">Đọc bằng giọng <b>{m["name"]}</b> · đổi giọng ở Thư viện</div>' if m
                    else '<div class="askhint">Chưa có giọng: thêm giọng ở Thư viện</div>')

        def home_go(text):
            if not (text or "").strip():
                raise gr.Error("Nhập câu cần đọc trước đã.")
            return text.strip(), *go("gen")

        gen_inputs = [g_voice, g_text, g_dict, g_takes, g_lang, g_name, g_fmt, g_steps, g_gs, g_speed, g_dur, g_dn, g_pre, g_post,
                      g_src, g_tmp_audio, g_tmp_text, g_tmp_ok, g_instruct, g_mode]
        gen_outputs = [g_msg, g_table, hdr, *g_outs]
        for trig in (home_send.click, home_text.submit):
            trig(home_go, home_text, [g_text, tabs, *nav.values()]).success(
                generate, gen_inputs, gen_outputs, concurrency_id="gpu")
        g_voice.change(home_hint_html, g_voice, home_hint)
        app.load(home_hint_html, g_voice, home_hint)
        theme_btn.click(None, None, None, js=TOGGLE_JS)

        g_voice.change(on_voice, g_voice, [g_vinfo, g_vsample])

        def on_src(src):
            tmp = src == TMP_SRC
            show, hid = gr.update(visible=True), gr.update(visible="hidden")
            return [hid if tmp else show] * 3 + [show if tmp else hid] * 3

        g_src.change(on_src, g_src, [g_voice, g_vinfo, g_vsample, g_tmp_audio, g_tmp_text, g_tmp_ok])

        # khung đề xuất cập nhật theo văn bản, giọng, cách tạo, tốc độ, định dạng
        ref_in = [g_text, g_dict, g_voice, g_src, g_mode, g_takes, g_steps, g_gs, g_speed, g_fmt]
        ref_out = [g_est, g_rec, g_takes, g_steps, g_gs]
        g_text.change(refresh, ref_in, ref_out, show_progress="hidden", trigger_mode="always_last")
        for c in (g_dict, g_voice, g_src, g_mode, g_speed, g_fmt):
            c.change(refresh, ref_in, ref_out, show_progress="hidden")
        app.load(refresh, ref_in, ref_out, show_progress="hidden")
        for s in (g_takes, g_steps, g_gs):                  # người dùng tự kéo thanh → chế độ Tự chỉnh
            s.release(lambda: MODE_CUSTOM, None, g_mode, show_progress="hidden").then(
                refresh, ref_in, ref_out, show_progress="hidden")
        g_btn.click(generate, gen_inputs, gen_outputs, concurrency_id="gpu")

        # ── Xử lý: Thư viện ─────────────────────────────────────────────────
        def _refresh_lists(ver, keep=None):
            ch = store.choices()
            vals = [v for _, v in ch]
            return (ver + 1, gr.update(choices=ch, value=keep if keep in vals else (vals[0] if vals else None)),
                    gr.update(choices=ch), gr.update(choices=[(ALL, "")] + ch), header_html())

        def load_manage(slug):
            m = store.get(slug) if slug else None
            if not m:
                return None, "", "", None, None, None, "", "", ""
            return (str(store.paths(slug)["ref"]), m.get("ref_text") or "", m["name"], m.get("gender"), m.get("region"),
                    m.get("age"), m.get("style") or "", m.get("notes") or "", "")

        def save_manage(slug, ref_text, name, gender, region, age, style, notes, ver, cur):
            if not slug:
                return msg("Chọn giọng trước.", "warn"), ver, gr.update(), gr.update(), gr.update(), gr.update()
            if not (name or "").strip():
                return msg("Tên giọng không được trống.", "bad"), ver, gr.update(), gr.update(), gr.update(), gr.update()
            store.update(slug, ref_text=(ref_text or "").strip(), name=name.strip(), gender=gender, region=region, age=age,
                         style=style, notes=notes)
            v, gv, ls, hv, h = _refresh_lists(ver, cur)
            return msg("Đã lưu. Nếu đổi lời mẫu, lần tạo giọng tới app sẽ mã hoá lại giọng mẫu.", "ok"), v, gv, ls, hv, h

        def remake_one(slug, ver, progress=gr.Progress()):
            if not slug:
                return msg("Chọn giọng trước.", "warn"), ver
            progress(0.2, desc="Đang tạo bản nghe thử…")
            s = st.make_sample(slug, _settings().get("test_sentence", TEST_SENTENCE))
            return msg(f"Đã tạo lại bản nghe thử. Giống giọng mẫu {_pct(s)}.", "ok"), ver + 1

        def soft_delete(slug, ver, cur):
            """Xoá mềm → mục Đã xoá. Trả về (tên, các cập nhật danh sách)."""
            name = store.get(slug)["name"]
            store.delete(slug)
            v, gv, ls, hv, h = _refresh_lists(ver, None if cur == slug else cur)
            return name, (v, gv, gr.update(choices=store.choices(), value=None), hv, h, trash_rows())

        def delete_voice(slug, ver, cur):
            if not slug or not store.get(slug):
                return msg("Chọn giọng trước.", "warn"), ver, gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
            name, upd = soft_delete(slug, ver, cur)
            return (msg(f"Đã chuyển “{name}” vào mục <b>Đã xoá</b> (cuối trang). Khôi phục được.", "ok"), *upd)

        def trash_pick(table, evt: gr.SelectData):
            row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
            try:
                tid, name = table[row][0], table[row][1]
            except Exception:
                return None, empty("Không đọc được dòng này.")
            return tid, msg(f"Đã chọn <b>{name}</b>. Bấm <b>Khôi phục</b> hoặc <b>Xoá vĩnh viễn</b>.")

        def trash_restore(tid, ver, cur):
            if not tid:
                return (msg("Bấm chọn một dòng trong bảng trước.", "warn"), ver, gr.update(), gr.update(), gr.update(),
                        gr.update(), gr.update(), None)
            slug = store.restore(tid)
            m = store.get(slug)
            v, gv, ls, hv, h = _refresh_lists(ver, cur)
            return msg(f"Đã khôi phục “{m['name'] if m else slug}” về thư viện.", "ok"), v, gv, ls, hv, h, trash_rows(), None

        def trash_purge(tid, ok):
            if not tid:
                return msg("Bấm chọn một dòng trong bảng trước.", "warn"), gr.update(), None, False
            if not ok:
                return msg("Tích ô “Tôi hiểu: xoá vĩnh viễn…” trước khi xoá.", "warn"), gr.update(), tid, False
            name = next((t["name"] for t in store.list_trash() if t["id"] == tid), tid)
            store.purge(tid)
            return msg(f"Đã xoá vĩnh viễn “{name}”.", "ok"), trash_rows(), None, False

        l_refresh.click(lambda v: v + 1, lib_ver, lib_ver)
        l_sel.change(load_manage, l_sel, [l_ref, l_reftext, l_name, l_gen, l_reg, l_age, l_style, l_notes, l_msg])
        l_save.click(save_manage, [l_sel, l_reftext, l_name, l_gen, l_reg, l_age, l_style, l_notes, lib_ver, g_voice],
                     [l_msg, lib_ver, g_voice, l_sel, h_voice, hdr])
        l_resample.click(remake_one, [l_sel, lib_ver], [l_msg, lib_ver], concurrency_id="gpu")
        l_del.click(delete_voice, [l_sel, lib_ver, g_voice], [l_msg, lib_ver, g_voice, l_sel, h_voice, hdr, t_table])
        t_table.select(trash_pick, t_table, [t_sel, t_info])
        t_restore.click(trash_restore, [t_sel, lib_ver, g_voice], [t_info, lib_ver, g_voice, l_sel, h_voice, hdr, t_table, t_sel])
        t_purge.click(trash_purge, [t_sel, t_ok], [t_info, t_table, t_sel, t_ok])

        # ── Xử lý: Thêm giọng ───────────────────────────────────────────────
        def analyze_ref(path, progress=gr.Progress()):
            if not path:
                return msg("Chưa có file giọng mẫu.", "bad"), None, None, gr.update()
            progress(0.2, desc="Đang kiểm tra và cắt đoạn mẫu…")
            wav, sr, info = audio.prepare_ref(path)
            tmp = DATA / "_tam" / f"mau_{dt.datetime.now():%H%M%S}.wav"
            tmp.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(tmp), wav, sr)
            progress(0.6, desc="Máy đang chép lời giọng mẫu…")
            text = st.transcribe(tmp)
            items = "".join(pill(t, k) for t, k in audio.verdict(info))
            cut = info["window"]
            extra = (f"<br><span class='small'>File gốc {_secs(info['raw_duration'])}: app chọn đoạn "
                     f"{_secs(cut[0])}–{_secs(cut[1])} làm mẫu.</span>") if info["raw_duration"] > 11.5 else ""
            bad = any(k == "bad" for _, k in audio.verdict(info))
            head = ("<b>Chưa đạt.</b> Nên thay file khác để giọng ra giống hơn." if bad
                    else "<b>Dùng được.</b> Kiểm tra lời mẫu bên dưới rồi lưu.")
            return msg(f"{head}<br>{items}{extra}", "bad" if bad else "ok"), str(tmp), str(tmp), text

        def name_taken(name):
            """Chặn lưu trùng (vd bấm Lưu hai lần): đã có giọng cùng tên, không phân biệt hoa thường."""
            key = (name or "").strip().casefold()
            return any(m["name"].strip().casefold() == key for m in store.list_voices())

        def save_new(prep, text, name, gender, region, age, style, consent, ver, cur, progress=gr.Progress()):
            fail = lambda t: (msg(t, "bad"), gr.update(), ver, gr.update(), gr.update(), gr.update(), gr.update())
            if not consent:
                return fail("Cần tích ô xác nhận người trong file đã đồng ý cho dùng giọng.")
            if not prep or not Path(prep).exists():
                return fail("Bấm “KIỂM TRA & CHUẨN BỊ” trước.")
            if not (name or "").strip():
                return fail("Đặt tên cho giọng.")
            if name_taken(name):
                return fail(f"Thư viện đã có giọng tên “{name.strip()}”. Đặt tên khác, hoặc xoá giọng cũ ở Thư viện → Sửa.")
            if not (text or "").strip():
                return fail("Cần lời trong giọng mẫu (gõ đúng từng chữ).")
            progress(0.2, desc="Đang lưu giọng, mã hoá giọng mẫu và tạo bản nghe thử…")
            wav, sr = audio.load_mono(prep)
            slug = st.add_voice(wav, sr, name.strip(), text.strip(), gender=gender, region=region, age=age, style=style,
                                source="upload", consent=True)
            m = store.get(slug)
            lab, kind = similarity.label(m["sample_sim"])
            v, gv, ls, hv, h = _refresh_lists(ver, slug)
            return (msg(f"Đã lưu “{m['name']}”. Bản nghe thử giống giọng mẫu <b>{_pct(m['sample_sim'])}</b> ({lab}).", kind),
                    gr.update(value=str(store.paths(slug)["sample"]), visible=True), v, gv, ls, hv, h)

        a_check.click(analyze_ref, a_audio, [a_report, a_prepared, a_prep, a_text], concurrency_id="gpu")
        a_save.click(save_new, [a_prep, a_text, a_name, a_gen, a_reg, a_age, a_style, a_consent, lib_ver, g_voice],
                     [a_msg, a_sample, lib_ver, g_voice, l_sel, h_voice, hdr], concurrency_id="gpu")

        # ── Xử lý: Thiết kế giọng ───────────────────────────────────────────
        def build_instruct(values):
            parts = []
            for v in values:
                if not v or v == "Auto":
                    continue
                en, _, zh = v.partition(" / ")
                parts.append(zh.strip() if "Dialect" in en else en.strip())
            return ", ".join(parts)

        def design(text, takes, *vals, progress=gr.Progress()):
            hide = {o: gr.update(value=None, visible="hidden") for o in d_outs}
            if not (text or "").strip():
                return {d_msg: msg("Cần câu đọc mẫu.", "bad"), **hide}
            instruct = build_instruct(vals)
            res = st.design(text.strip(), instruct, takes=int(takes), progress=progress, num_step=32)
            files, out = [], dict(hide)
            folder = DATA / "_tam"
            folder.mkdir(parents=True, exist_ok=True)
            for i, r in enumerate(res):
                f = folder / f"thiet-ke_{dt.datetime.now():%H%M%S}_{i + 1}.wav"
                sf.write(str(f), audio.normalize(audio.trim(np.asarray(r["wav"], np.float32), st.sr)), st.sr)
                files.append(str(f))
                out[d_outs[i]] = gr.update(value=str(f), visible=True, label=f"Phương án {i + 1}")
            out[d_msg] = msg(f"Đã tạo {len(res)} phương án" + (f" · {instruct}" if instruct else " · để máy tự chọn giọng")
                             + ". Nghe rồi chọn phương án để lưu.", "ok")
            out[d_pick] = gr.update(choices=[f"Phương án {i + 1}" for i in range(len(res))], value="Phương án 1")
            out[d_state] = {"files": files, "text": text.strip(), "instruct": instruct}
            return out

        def save_design(state, pick, name, gender, age, ver, cur, progress=gr.Progress()):
            fail = lambda t: (msg(t, "bad"), ver, gr.update(), gr.update(), gr.update(), gr.update())
            if not state:
                return fail("Bấm THIẾT KẾ trước.")
            if not (name or "").strip():
                return fail("Đặt tên cho giọng.")
            if name_taken(name):
                return fail(f"Thư viện đã có giọng tên “{name.strip()}”. Đặt tên khác.")
            i = int(str(pick).split()[-1]) - 1 if pick else 0
            progress(0.2, desc="Đang lưu giọng và tạo bản nghe thử…")
            wav, sr = audio.load_mono(state["files"][i])
            slug = st.add_voice(wav, sr, name.strip(), state["text"], gender=gender, age=age, region="Chưa rõ",
                                style=state["instruct"], source="design", consent=True, design_instruct=state["instruct"],
                                notes="Giọng tổng hợp bằng Thiết kế giọng, không phải người thật.")
            m = store.get(slug)
            v, gv, ls, hv, h = _refresh_lists(ver, slug)
            return msg(f"Đã lưu “{m['name']}” vào thư viện. Bản nghe thử giống giọng mẫu {_pct(m['sample_sim'])}.", "ok"), v, gv, ls, hv, h

        def quick_pick(name):
            g, a, pch, sty = QUICK.get(name, (None, None, None, None))
            vals = []
            for (c, ch), want in zip(list(_CATEGORIES.items()), (g, a, pch, sty, None, None)):
                vals.append(next((x for x in ch if x.split(" / ")[0] == want), "Auto") if want else "Auto")
            gen_vi = {"Male": "Nam", "Female": "Nữ"}.get(g)
            return (*vals, gr.update(value=gen_vi) if gen_vi else gr.update(), gr.update(value=AGE_VI[a]) if a else gr.update())

        d_quick.change(quick_pick, d_quick, [*d_attrs, d_gen, d_age])
        d_btn.click(design, [d_text, d_takes, *d_attrs], [d_msg, d_pick, d_state, *d_outs], concurrency_id="gpu")
        d_save.click(save_design, [d_state, d_pick, d_name, d_gen, d_age, lib_ver, g_voice],
                     [d_saved, lib_ver, g_voice, l_sel, h_voice, hdr], concurrency_id="gpu")

        # ── Xử lý: Lịch sử ──────────────────────────────────────────────────
        def hist_rows(slug):
            rows = []
            for r in history.rows(slug or None):
                rows.append([r["id"], r["created"][5:16].replace("T", " "), r["voice_name"], (r["text"] or "")[:120],
                             _pct(r["sim"]), _secs(r["duration"] or 0), Path(r["file"]).name])
            return rows

        def hist_pick(table, evt: gr.SelectData):
            row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
            try:
                rid = int(table[row][0])
            except Exception:
                return None, empty("Không đọc được dòng này."), None
            r = history.get(rid)
            if not r:
                return None, empty("Không thấy bản ghi."), None
            ok = Path(r["file"]).exists()
            info = msg(f"<b>{r['voice_name']}</b> · {r['created'].replace('T', ' ')} · giống {_pct(r['sim'])}<br>{r['text']}"
                       f"<br><span class='small'>{r['file']}{'' if ok else ' (file không còn)'}</span>")
            return (r["file"] if ok else None), info, rid

        def hist_reuse(rid):
            r = history.get(rid) if rid else None
            if not r:
                return gr.update(), gr.update(), *[gr.update()] * (1 + len(NAV))
            voice_upd = (gr.update(choices=store.choices(), value=r["voice_slug"]) if store.get(r["voice_slug"])
                         else gr.update())
            return r["text"], voice_upd, *go("gen")

        def hist_del(rid, slug):
            if rid:
                history.remove(rid)
            return hist_rows(slug), None, empty("Đã xoá khỏi lịch sử. File kết quả vẫn còn trong thư mục."), None

        h_refresh.click(hist_rows, h_voice, h_table)
        h_voice.change(hist_rows, h_voice, h_table)
        h_table.select(hist_pick, h_table, [h_audio, h_info, h_sel])
        h_reuse.click(hist_reuse, h_sel, [g_text, g_voice, tabs, *nav.values()])
        h_del.click(hist_del, [h_sel, h_voice], [h_table, h_audio, h_info, h_sel])
        h_open.click(lambda: os.startfile(str(OUTPUTS)), None, None)

        # ── Xử lý: Cài đặt ──────────────────────────────────────────────────
        def dict_save(rows):
            table = {str(r[0]).strip().lower(): str(r[1]).strip() for r in rows or []
                     if r and len(r) > 1 and str(r[0]).strip() not in ("", "None") and str(r[1]).strip() not in ("", "None")}
            save_dict(table)
            return msg(f"Đã lưu {len(table)} từ.", "ok")

        def settings_save(fmt, test):
            _save_settings(format=fmt, test_sentence=(test or TEST_SENTENCE).strip())
            return msg("Đã lưu cài đặt.", "ok"), fmt

        def remake_all(test, ver, progress=gr.Progress()):
            vs = store.list_voices()
            for i, m in enumerate(vs):
                progress(i / max(len(vs), 1), desc=f"Bản nghe thử {i + 1}/{len(vs)}: {m['name']}")
                st.make_sample(m["slug"], (test or TEST_SENTENCE).strip())
            _save_settings(test_sentence=(test or TEST_SENTENCE).strip())
            return msg(f"Đã tạo lại bản nghe thử cho {len(vs)} giọng.", "ok"), ver + 1

        s_dict_save.click(dict_save, s_dict, s_dict_msg)
        s_save.click(settings_save, [s_fmt, s_test], [s_msg, g_fmt])
        s_remake.click(remake_all, [s_test, lib_ver], [s_msg, lib_ver], concurrency_id="gpu")
        s_open.click(lambda: os.startfile(str(DATA)), None, None)
        app.load(header_html, None, hdr)

        # Danh sách giọng trong các ô chọn được tạo lúc mở trang. Giọng thêm sau đó (ở tab / lần mở khác) chưa có trong
        # danh sách → chọn nó sẽ báo "not in the list of choices". Luôn lấy lại danh sách mới trước khi chọn.
        def sync_lists(cur, sel, hv):
            ch = store.choices()
            vals = [v for _, v in ch]
            return (gr.update(choices=ch, value=cur if cur in vals else (vals[0] if vals else None)),
                    gr.update(choices=ch, value=sel if sel in vals else None),
                    gr.update(choices=[(ALL, "")] + ch, value=hv if (hv in vals or hv == "") else ""))

        lists = [g_voice, l_sel, h_voice]
        app.load(sync_lists, lists, lists)

        # ── Thanh menu: chuyển trang + tô sáng mục đang mở ─────────────────
        def _go(target):
            return lambda: go(target)

        for k, b in list(nav.items()) + list(cards.items()) + [("home", brand)]:
            ev = b.click(_go(k), None, [tabs, *nav.values()]).then(sync_lists, lists, lists)
            if k == "hist":
                ev.then(hist_rows, h_voice, h_table)
            elif k == "lib":
                ev.then(lambda v: v + 1, lib_ver, lib_ver).then(trash_rows, None, t_table)

        # các nút trong thẻ giọng (tạo trong gr.render) dùng các hàm này
        def _use(slug):
            return lambda: (gr.update(choices=store.choices(), value=slug), *go("gen"))

        def _fav(slug):
            def run(ver, cur):
                m = store.get(slug)
                store.update(slug, favorite=not m.get("favorite"))
                ch = store.choices()
                return ver + 1, gr.update(choices=ch, value=cur if cur in [v for _, v in ch] else slug)
            return run

        def _del(slug):
            def run(ver, cur):
                if not store.get(slug):
                    return ver + 1, gr.update(), gr.update(), gr.update(), header_html(), trash_rows()
                name, upd = soft_delete(slug, ver, cur)
                gr.Info(f"Đã chuyển “{name}” vào mục Đã xoá (cuối trang Thư viện). Khôi phục được.")
                return upd
            return run

        def _edit(slug):
            return lambda: (gr.update(choices=store.choices(), value=slug), gr.Accordion(open=True))

    return app


GUIDE = """
### Dùng nhanh

| Bước | Thẻ | Làm gì |
|---|---|---|
| 1 | **Thư viện giọng** | Nghe các bản nghe thử, bấm **Dùng** ở giọng ưng ý |
| 2 | **Tạo giọng** | Dán văn bản, chọn giọng, để **✨ Để app chọn** cách tạo, bấm **TẠO GIỌNG**. File tự lưu, bản ★ là bản giống giọng mẫu nhất |
| 3 | **Thêm giọng** | Tải file 6–10 giây → **KIỂM TRA** → sửa lời mẫu → đặt tên, tích đồng ý → **LƯU** |
| 4 | **Thiết kế giọng** | Chọn giới tính, độ tuổi, cao độ… → **THIẾT KẾ** → nghe → lưu phương án ưng ý |
| 5 | **Lịch sử** | Nghe lại, đưa văn bản và giọng sang thẻ Tạo giọng để làm lại |

### Độ giống giọng mẫu (con số % trong app)
- Máy so "dấu vân giọng" của bản tạo ra với giọng mẫu bằng model WavLM-SV (Microsoft). 100% là trùng khít.
- Đo thử trên 15 giọng: bản tạo ra thường đạt **93–98%**. Trên 95% là rất giống, 92–95% là giống, dưới 88% là khác.
- **Lưu ý:** thang đo này cho điểm khá cao cả với hai giọng khác nhau cùng giới (có cặp đạt 96%). Vì vậy con số dùng tốt nhất để
  **so các bản với nhau** và chọn bản giống nhất, không phải bằng chứng hai giọng là một người. Cột **Xếp hạng** cho biết bản tạo ra
  có giống giọng mẫu hơn mọi giọng khác trong thư viện không (1/15 là tốt nhất). Tai người vẫn là thước đo cuối cùng.

### Muốn giống hơn
- Giọng mẫu sạch, 6–10 giây, một người nói, không nhạc, không vang. App tự cắt lặng và chọn đoạn tốt nhất khi file dài.
- Lời mẫu đúng từng chữ. Sai lời mẫu là nguyên nhân hay gặp nhất khiến giọng ra lệch.
- Chọn cách tạo **🎯 Giống giọng nhất** (3 bản, 48 bước) rồi nghe bản ★. "Để app chọn" tự dùng cách này cho đoạn ngắn
  (dưới 20 giây), giọng tạm, và giọng có bản nghe thử giống dưới 92%; đoạn dài hơn dùng **⚖️ Cân bằng** (1 bản) cho đỡ chờ.
- Mức bám (CFG) và số bước ảnh hưởng rất ít: đo trên 5 giọng × 8 bộ thông số chỉ chênh 94,6–95,7%.

### Quy tắc
- Model OmniVoice có giấy phép **CC-BY-NC**: giọng tạo ra chỉ dùng thử nội bộ, không đăng quảng cáo hay bán hàng.
- Chỉ lưu giọng của người đã đồng ý. Giọng ở thẻ Thiết kế là giọng tổng hợp, không phải người thật.

### File nằm ở đâu
- Giọng: `data\\studio\\voices\\<tên>` · Kết quả: `data\\studio\\outputs\\<ngày>` · Giọng đã xoá: `data\\studio\\_da-xoa`
- Mọi chức năng của OmniVoice gốc đã gộp vào app này: nhân bản giọng (thư viện hoặc **giọng tạm** tải lên dùng ngay),
  thiết kế giọng, chỉ dẫn (Instruct), tốc độ, độ dài, số bước, lọc tạp âm. Một địa chỉ duy nhất: `http://localhost:10601`.
"""
