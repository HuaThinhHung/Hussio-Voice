"""Giao diện HUSSIO Voice: phong cách portfolio tối giản (nền trắng, chữ đậm lớn, ô nhập dạng viên thuốc, thẻ bo tròn,
chữ nền mờ khổng lồ). Có hai bản Sáng / Tối, đổi bằng nút ☀/☾ trên thanh menu (nhớ lựa chọn trong trình duyệt)."""
import base64
import html
from pathlib import Path

import gradio as gr

ASSETS = Path(__file__).with_name("assets")
NAVY = "#1F2A44"                                          # màu chính bộ nhận diện HUSSIO (Main)

# Bảng màu: (sáng, tối).
# Sáng: trắng + navy HUSSIO #1F2A44. Tối: than chì trung tính (như giao diện moodboard), không đen tuyền, không navy:
# nền #18181B, khung #1F1F23, ô nhập #27272A, chữ #F4F4F5 (tương phản ≈ 16:1), chữ phụ #A1A1AA (≈ 7,6:1).
# Nút chính bản Tối là nền trắng chữ than chì; đỏ HUSSIO #DA3F3F chỉ làm điểm nhấn nhỏ.
C = {
    "bg": ("#FFFFFF", "#18181B"), "soft": ("#F5F5F5", "#27272A"), "card": ("#FFFFFF", "#1F1F23"),
    "line": ("#E5E7EB", "#2E2E33"), "line2": ("#D1D5DB", "#3F3F46"),
    "text": ("#1A2039", "#F4F4F5"), "text2": ("#3A3A3A", "#D4D4D8"), "muted": ("#6B7280", "#A1A1AA"),
    "brand": (NAVY, "#F4F4F5"), "brand_fg": ("#FFFFFF", "#18181B"), "brand_hover": ("#2B395C", "#FFFFFF"),
    "accent": ("#DA3F3F", "#DA3F3F"), "wm": ("#E8EBF1", "#26262B"),
    "ok": ("#16A34A", "#4ADE80"), "warn": ("#D97706", "#FBBF24"), "bad": ("#DA3F3F", "#F87171"),
}
FONT = 'ui-sans-serif, system-ui, -apple-system, "Segoe UI", Inter, Roboto, Arial, sans-serif'


def _b64(name: str) -> str:
    p = ASSETS / name
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode() if p.exists() else ""


MARK = _b64("hussio-mark.png")
MARK_W = _b64("hussio-mark-white.png")
WORD = _b64("wordmark-navy.png")                # wordmark chính thức (D:\HUSSIO Images\logo\Group-1.png)
WORD_W = _b64("wordmark-white.png")             # bản trắng (Group.png)


def icon(name: str) -> str:
    return str(ASSETS / "icons" / f"{name}.svg")


def studio_theme():
    t = gr.themes.Base(font=[FONT], radius_size="lg", text_size="md")
    keys = dict(
        body_background_fill="bg", body_text_color="text", body_text_color_subdued="muted",
        background_fill_primary="card", background_fill_secondary="soft",
        block_background_fill="card", block_border_color="line", block_label_text_color="muted",
        block_title_text_color="text", block_info_text_color="muted", block_label_background_fill="card",
        panel_background_fill="card", panel_border_color="line", border_color_primary="line", border_color_accent="brand",
        input_background_fill="soft", input_border_color="line", input_border_color_focus="brand",
        input_placeholder_color="muted", color_accent="brand", color_accent_soft="line",
        button_primary_background_fill="brand", button_primary_background_fill_hover="brand_hover",
        button_primary_text_color="brand_fg", button_primary_border_color="brand",
        button_secondary_background_fill="card", button_secondary_background_fill_hover="soft",
        button_secondary_text_color="text", button_secondary_border_color="line",
        slider_color="brand", checkbox_background_color="soft", checkbox_border_color="line2",
        checkbox_background_color_selected="brand", checkbox_border_color_selected="brand",
        checkbox_label_background_fill="card", checkbox_label_text_color="text",
        loader_color="brand", link_text_color="brand",
        table_even_background_fill="card", table_odd_background_fill="soft", table_border_color="line",
        table_row_focus="soft",
    )
    vals = {}
    for k, c in keys.items():
        if hasattr(t, k):
            vals[k] = C[c][0]
        if hasattr(t, k + "_dark"):
            vals[k + "_dark"] = C[c][1]
    for k in ("shadow_drop", "shadow_drop_lg", "block_shadow", "button_primary_shadow", "button_secondary_shadow"):
        if hasattr(t, k):
            vals[k] = "none"
        if hasattr(t, k + "_dark"):
            vals[k + "_dark"] = "none"
    return t.set(**vals)


# Nhớ Sáng/Tối trong trình duyệt (mặc định Sáng). Địa chỉ giữ gọn http://localhost:10601: bỏ "?__theme=…" khỏi thanh
# địa chỉ và tự đặt lớp .dark cho trang (Gradio mặc định theo chế độ của Windows; script giữ đúng lựa chọn của người dùng).
THEME_HEAD = """<script>
(function () {
  let want = 'light';
  try { want = localStorage.getItem('hx-theme') || 'light'; } catch (e) {}
  window.__hxTheme = want;
  const u = new URL(window.location.href);
  if (u.searchParams.has('__theme')) {
    u.searchParams.delete('__theme');
    history.replaceState(null, '', u.pathname + (u.search || '') + u.hash);
  }
  const apply = () => {
    if (!document.body) return;
    const d = window.__hxTheme === 'dark';
    if (document.body.classList.contains('dark') !== d) document.body.classList.toggle('dark', d);
  };
  const start = () => {
    apply();
    new MutationObserver(apply).observe(document.body, {attributes: true, attributeFilter: ['class']});
  };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start); else start();
})();

// Gradio 6.9: danh sách thả xuống (ul.options, position: fixed) đo lại khi cuộn nhưng không dời theo ô chọn
// → danh sách đứng yên trên màn hình. Tự dời danh sách theo ô chọn; ô chọn ra khỏi màn hình thì đóng danh sách.
(function () {
  const st = new WeakMap();
  const num = v => (v === '' || v == null ? NaN : parseFloat(v));
  function base(ul) {
    const ref = ul.previousElementSibling;
    if (!ref) return null;
    const s = {ref, refTop: ref.getBoundingClientRect().top, top: num(ul.style.top), bottom: num(ul.style.bottom),
               setTop: ul.style.top, setBottom: ul.style.bottom};
    st.set(ul, s);
    return s;
  }
  function follow() {
    document.querySelectorAll('ul.options').forEach(ul => {
      let s = st.get(ul);
      if (!s || ul.style.top !== s.setTop || ul.style.bottom !== s.setBottom) s = base(ul);  // Gradio vừa đặt lại vị trí
      if (!s) return;
      const r = s.ref.getBoundingClientRect(), d = r.top - s.refTop;
      if (!isNaN(s.top)) { s.setTop = (s.top + d) + 'px'; ul.style.top = s.setTop; }
      if (!isNaN(s.bottom)) { s.setBottom = (s.bottom - d) + 'px'; ul.style.bottom = s.setBottom; }
      if (r.bottom < 0 || r.top > innerHeight) {
        const inp = s.ref.parentElement && s.ref.parentElement.querySelector('input');
        if (inp) inp.blur();
      }
    });
  }
  new MutationObserver(ms => {
    for (const m of ms) for (const n of m.addedNodes) {
      if (n.nodeType === 1 && n.tagName === 'UL' && n.classList.contains('options')) {
        setTimeout(() => document.querySelectorAll('ul.options').forEach(u => { if (!st.get(u)) base(u); }), 0);
        return;
      }
    }
  }).observe(document.documentElement, {childList: true, subtree: true});
  addEventListener('scroll', follow, true);   // true: bắt cả cuộn trong khung con
  addEventListener('resize', follow);
})();
</script>"""

# Nút ☀/☾: đổi chế độ ngay tại chỗ, không tải lại trang, không thêm gì vào địa chỉ
TOGGLE_JS = """() => {
  const dark = !document.body.classList.contains('dark');
  window.__hxTheme = dark ? 'dark' : 'light';
  document.body.classList.toggle('dark', dark);
  try { localStorage.setItem('hx-theme', window.__hxTheme); } catch (e) {}
}"""


def _vars(i: int) -> str:
    return "".join(f"--hx-{k.replace('_', '-')}: {v[i]};" for k, v in C.items())


CSS = f"""
:root, .gradio-container {{{_vars(0)}}}
.dark, .dark .gradio-container {{{_vars(1)}}}
body, .gradio-container {{background: var(--hx-bg) !important; color: var(--hx-text); font-family: {FONT} !important;}}
.gradio-container {{max-width: 1120px !important; margin: 0 auto !important; padding-top: 0 !important;}}
footer {{display: none !important;}}
/* Gradio tự bó khung trang theo cỡ màn hình (768px…) và chừa lề 32px: mở rộng ra, lề 16px */
.gradio-container .app {{max-width: 1120px !important; width: 100% !important; box-sizing: border-box !important;
  padding-left: 16px !important; padding-right: 16px !important; margin: 0 auto !important;}}
.dark-only {{display: none !important;}}
.dark .light-only {{display: none !important;}} .dark .dark-only {{display: inline-block !important;}}
.dark img.avatar.dark-only {{display: block !important;}}

/* ── Thanh menu trên cùng ── */
#topnav {{position: sticky; top: 0; z-index: 60; background: color-mix(in srgb, var(--hx-bg) 86%, transparent);
  backdrop-filter: blur(12px); border-bottom: 1px solid var(--hx-line); padding: 10px 2px !important;
  display: grid !important; grid-template-columns: auto minmax(0, 1fr) auto; align-items: center !important;
  column-gap: 16px !important;}}
#topnav > * {{min-width: 0 !important; width: auto !important; flex: none !important;}}
#topnav .brand {{display: flex; align-items: center; white-space: nowrap; padding: 0 4px; cursor: pointer;}}
#topnav .brand img.wd {{height: 16px; width: auto; max-width: none;}}
#navpills {{flex-wrap: nowrap !important; gap: 5px !important; justify-content: flex-start !important; align-items: center !important;
  overflow-x: auto; scrollbar-width: none; padding: 2px 0 !important;}}
#navpills > :first-child {{margin-left: auto !important;}} #navpills > :last-child {{margin-right: auto !important;}}
#navpills::-webkit-scrollbar {{display: none;}}
.navpill, .navpill * {{font-weight: 500 !important; white-space: nowrap !important;}}
.navpill {{border-radius: 999px !important; font-size: 13px !important; padding: 5px 12px !important; letter-spacing: 0 !important;
  min-width: 0 !important; width: auto !important; flex: 0 0 auto !important; box-shadow: none !important; min-height: 0 !important;}}
.navpill.secondary {{background: transparent !important; border: 1px solid var(--hx-line) !important; color: var(--hx-text) !important;}}
.navpill.secondary:hover {{background: var(--hx-soft) !important;}}
.navpill.primary {{background: var(--hx-brand) !important; border: 1px solid var(--hx-brand) !important; color: var(--hx-brand-fg) !important;}}
#themebtn {{border-radius: 999px !important; width: 38px !important; min-width: 38px !important; height: 38px; padding: 0 !important;
  font-size: 16px !important; flex: 0 0 38px !important; background: transparent !important; border: 1px solid var(--hx-line) !important;
  color: var(--hx-text) !important;}}
#themebtn {{font-size: 0 !important;}}
#themebtn::after {{content: '☾'; font-size: 16px;}}
.dark #themebtn::after {{content: '☀';}}
.status {{font-size: 12px; color: var(--hx-muted); white-space: nowrap; text-align: right;}}

/* ── Trang chủ kiểu portfolio ── */
#hero {{text-align: center; padding: 30px 0 4px;}}
#hero .chip {{display: inline-flex; gap: 8px; align-items: center; border: 1px solid var(--hx-line); border-radius: 999px;
  padding: 6px 14px; font-size: 13px; color: var(--hx-text2); background: var(--hx-card);}}
#hero .chip i {{width: 8px; height: 8px; border-radius: 50%; background: var(--hx-accent); display: inline-block;}}
#hero .tile {{width: 58px; height: 58px; margin: 22px auto 14px; border-radius: 16px; background: {NAVY}; display: flex;
  align-items: center; justify-content: center; box-shadow: 0 6px 22px rgba(31,42,68,.25);}}
.dark #hero .tile {{background: #F5F5F5; box-shadow: 0 6px 22px rgba(0,0,0,.25);}}
#hero .tile img {{height: 32px; width: auto;}}
#hero .hi {{font-size: 22px; font-weight: 600; color: var(--hx-text);}}
#hero h1 {{font-size: clamp(40px, 6.4vw, 64px); line-height: 1.04; font-weight: 800; letter-spacing: -0.025em;
  color: var(--hx-text); margin: 6px 0 20px; border: 0;}}
#hero .avatar {{height: 190px; width: auto; margin: 0 auto 22px;}}
#hero img.avatar.light-only {{display: block;}}
#ask {{max-width: 640px; margin: 0 auto !important; border: 1px solid var(--hx-line) !important; border-radius: 999px !important;
  padding: 5px 6px 5px 16px !important; background: color-mix(in srgb, var(--hx-soft) 60%, transparent) !important;
  align-items: center !important; gap: 6px !important; flex-wrap: nowrap !important;}}
#ask .block, #ask .form {{border: none !important; background: transparent !important; box-shadow: none !important; padding: 0 !important;}}
#ask textarea, #ask input {{border: none !important; background: transparent !important; box-shadow: none !important; font-size: 16px !important;
  padding: 8px 4px !important;}}
#send {{border-radius: 999px !important; width: 44px !important; min-width: 44px !important; height: 44px; padding: 0 !important;
  flex: 0 0 44px !important; font-size: 19px !important; font-weight: 700 !important;}}
.askhint {{text-align: center; font-size: 13px; color: var(--hx-muted); margin-top: 10px;}}
.askhint b {{color: var(--hx-text);}}
#qcards {{max-width: 760px; margin: 22px auto 0 !important; gap: 10px !important; justify-content: center !important;
  flex-wrap: wrap !important;}}
.qcard {{flex-direction: column !important; gap: 7px !important; width: 104px !important; min-width: 104px !important;
  flex: 0 0 104px !important; height: 88px !important; border-radius: 16px !important; border: 1px solid var(--hx-line) !important;
  background: color-mix(in srgb, var(--hx-card) 40%, transparent) !important; color: var(--hx-text) !important;
  font-size: 13.5px !important; font-weight: 500 !important; box-shadow: none !important; transition: transform .15s, background .15s;}}
.qcard:hover {{background: var(--hx-soft) !important; transform: translateY(-2px);}}
.qcard img {{width: 22px !important; height: 22px !important; margin: 0 !important;}}
.wm {{font-size: clamp(90px, 17vw, 200px); font-weight: 900; text-align: center; letter-spacing: .01em; line-height: .9;
  margin: 34px 0 0; background-image: linear-gradient(180deg, var(--hx-wm) 0%, transparent 92%) !important;
  -webkit-background-clip: text !important; background-clip: text !important; color: transparent !important;
  -webkit-text-fill-color: transparent !important; user-select: none; overflow: hidden;}}

/* ── Các trang bên trong ── */
#main-tabs > .tab-wrapper, #main-tabs [role="tablist"] {{display: none !important;}}
#main-tabs {{border: none !important; background: transparent !important;}}
.pagehead {{text-align: center; padding: 26px 0 14px;}}
.pagehead .t {{font-size: clamp(28px, 4vw, 40px); font-weight: 800; letter-spacing: -0.02em; color: var(--hx-text);}}
.pagehead .d {{font-size: 14px; color: var(--hx-muted); margin-top: 4px;}}
.card {{background: var(--hx-card) !important; border: 1px solid var(--hx-line) !important; border-radius: 20px !important;
  padding: 18px !important; gap: 10px !important; margin-bottom: 12px;}}
.step-head {{display: flex; gap: 12px; align-items: flex-start; padding-bottom: 10px; border-bottom: 1px solid var(--hx-line);}}
.step-head .num {{flex: none; min-width: 26px; height: 26px; padding: 0 7px; border-radius: 13px; background: var(--hx-brand);
  color: var(--hx-brand-fg); font-weight: 700; font-size: 13px; display: flex; align-items: center; justify-content: center;}}
.step-head .t {{font-size: 16px; font-weight: 700; color: var(--hx-text);}}
.step-head .d {{font-size: 13px; color: var(--hx-muted); margin-top: 2px;}}
.pill {{display: inline-block; border: 1px solid var(--hx-line); background: var(--hx-soft); color: var(--hx-text2); border-radius: 999px;
  padding: 4px 12px; font-size: 13px; margin: 0 6px 6px 0;}}
.pill.ok {{border-color: var(--hx-ok); color: var(--hx-ok);}} .pill.warn {{border-color: var(--hx-warn); color: var(--hx-warn);}}
.pill.bad {{border-color: var(--hx-bad); color: var(--hx-bad);}} .pill b {{color: inherit;}}
.msg {{border-radius: 14px; padding: 10px 14px; font-size: 14px; border: 1px solid var(--hx-line); background: var(--hx-soft);
  color: var(--hx-text);}}
.msg.ok {{border-color: var(--hx-ok);}} .msg.warn {{border-color: var(--hx-warn);}} .msg.bad {{border-color: var(--hx-bad);}}
.msg .small {{color: var(--hx-muted); font-size: 12px;}}
.empty {{color: var(--hx-muted); border: 1px dashed var(--hx-line2); border-radius: 14px; padding: 16px; text-align: center; font-size: 14px;}}
.btn-main {{min-height: 52px; font-size: 15px !important; font-weight: 700 !important; letter-spacing: .04em; border-radius: 999px !important;}}
.vcard {{background: var(--hx-card) !important; border: 1px solid var(--hx-line) !important; border-radius: 20px !important;
  padding: 14px !important; gap: 8px !important; transition: transform .15s, border-color .15s;}}
.vcard:hover {{border-color: var(--hx-line2) !important; transform: translateY(-2px);}}
.vname {{font-size: 16px; font-weight: 700; color: var(--hx-text);}}
.vuse {{border-radius: 999px !important;}}
.vacts {{gap: 6px !important; flex-wrap: nowrap !important;}}
.vbtn {{flex: 1 1 0 !important; min-width: 0 !important; border-radius: 999px !important; font-size: 12.5px !important;
  padding: 5px 8px !important; background: transparent !important; border: 1px solid var(--hx-line) !important;
  color: var(--hx-text) !important; box-shadow: none !important; white-space: nowrap !important;}}
.vbtn:hover {{background: var(--hx-soft) !important;}}
.vbtn.del:hover {{border-color: var(--hx-bad) !important; color: var(--hx-bad) !important;}}
.explain {{display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; margin: 0 0 14px;}}
.ex {{border: 1px solid var(--hx-line); border-radius: 16px; padding: 12px 14px; font-size: 13.5px; line-height: 1.6;
  color: var(--hx-text2); background: var(--hx-card);}}
.ex b {{color: var(--hx-text);}} .ex span {{color: var(--hx-muted);}} .ex.on {{border-color: var(--hx-brand);}}
.vtags {{margin-top: 6px;}}
.tag {{display: inline-block; font-size: 12px; padding: 2px 9px; border-radius: 999px; background: var(--hx-soft);
  border: 1px solid var(--hx-line); color: var(--hx-text2); margin: 0 4px 4px 0;}}
.tag.m {{border-color: #93C5FD; color: #2563EB;}} .tag.f {{border-color: #F9A8D4; color: #DB2777;}}
.dark .tag.m {{color: #93C5FD;}} .dark .tag.f {{color: #F9A8D4;}}
.tag.src {{border-style: dashed;}}
.vnote {{font-size: 13px; color: var(--hx-muted); margin-top: 2px;}}
.vsim {{font-size: 12px; margin-top: 6px;}} .vsim.ok {{color: var(--hx-ok);}} .vsim.warn {{color: var(--hx-warn);}}
.vsim.bad {{color: var(--hx-bad);}}
.tips {{border: 1px dashed var(--hx-line2); border-radius: 16px; padding: 12px 16px; color: var(--hx-text2); font-size: 13px;
  line-height: 1.75;}}
.tips b {{color: var(--hx-text);}}
.rec {{border: 1px solid var(--hx-brand); border-radius: 16px; padding: 12px 14px; background: var(--hx-soft); font-size: 13.5px;
  line-height: 1.6; color: var(--hx-text2);}}
.rec .h {{font-size: 15px; color: var(--hx-text); margin-bottom: 2px;}} .rec b {{color: var(--hx-text);}}
.rec .desc, .rec .alt {{color: var(--hx-muted); font-size: 12.5px; margin-top: 2px;}}
.rec .set {{margin-top: 8px;}} .rec .set .pill {{background: var(--hx-card); font-size: 12.5px; margin-bottom: 0;}}
.guide h3 {{color: var(--hx-text) !important;}}
.guide table {{width: 100%;}}
.foot {{text-align: center; font-size: 12px; color: var(--hx-muted); padding: 22px 0 4px;}}
#footstatus .status {{text-align: center; padding-bottom: 12px;}}
.foot img {{height: 11px !important; width: auto !important; vertical-align: middle; margin: 0 6px !important; opacity: .8;}}
.foot img.light-only {{display: inline-block !important;}}
.dark .foot img.light-only {{display: none !important;}}

/* ── Màn hình hẹp (đặt cuối để đè quy tắc gốc) ── */
@media (max-width: 900px) {{
  .navpill {{font-size: 12px !important; padding: 4px 9px !important;}}
  #navpills {{gap: 4px !important;}}
  #topnav .brand img.wd {{height: 13px;}} #topnav {{column-gap: 10px !important;}}
}}
/* Thẻ giọng tự xuống hàng khi thiếu chỗ, không tràn ngang */
.vrow {{flex-wrap: wrap !important;}}
.vrow > .vcard {{min-width: 240px !important; flex: 1 1 240px !important;}}
html, body {{overflow-x: hidden;}}
"""


def step(num: str, title: str, desc: str = "") -> str:
    return (f'<div class="step-head"><span class="num">{num}</span><div><div class="t">{html.escape(title)}</div>'
            f'<div class="d">{desc}</div></div></div>')


def page_head(title: str, desc: str = "") -> str:
    return f'<div class="pagehead"><div class="t">{html.escape(title)}</div><div class="d">{desc}</div></div>'


def pill(text: str, kind: str = "") -> str:
    return f'<span class="pill {kind}">{text}</span>'


def msg(text: str, kind: str = "") -> str:
    return f'<div class="msg {kind}">{text}</div>'


def empty(text: str) -> str:
    return f'<div class="empty">{text}</div>'


def brand_html() -> str:
    return (f'<div class="brand"><img class="wd light-only" src="{WORD}" alt="HUSSIO">'
            f'<img class="wd dark-only" src="{WORD_W}" alt="HUSSIO"></div>')


def hero_html() -> str:
    return f"""<div id="hero">
      <div class="chip"><i></i>OmniVoice · chỉ dùng thử nội bộ (CC-BY-NC)</div>
      <div class="tile"><img class="light-only" src="{MARK_W}"><img class="dark-only" src="{MARK}"></div>
      <div class="hi">Xin chào, đây là studio giọng đọc 👋</div>
      <h1>HUSSIO Voice</h1>
      <img class="avatar light-only" src="{MARK}" alt="HUSSIO"><img class="avatar dark-only" src="{MARK_W}" alt="HUSSIO">
    </div>"""


def foot_html() -> str:
    return (f'<div class="foot">© HUSSIO<img class="light-only" src="{WORD}"><img class="dark-only" src="{WORD_W}">· '
            f'Giọng OmniVoice chỉ dùng thử nội bộ, không đăng quảng cáo · Chỉ lưu giọng của người đã đồng ý</div>')
