#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Gradio demo for OmniVoice.

Supports voice cloning and voice design.

Usage:
    omnivoice-demo --model /path/to/checkpoint --port 8000
"""

import argparse
import logging
from typing import Any, Dict

import gradio as gr
import numpy as np
import torch

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.common import get_best_device
from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name


# ---------------------------------------------------------------------------
# Language list — all 600+ supported languages
# ---------------------------------------------------------------------------
_ALL_LANGUAGES = ["Auto"] + sorted(lang_display_name(n) for n in LANG_NAMES)


# ---------------------------------------------------------------------------
# Voice Design instruction templates
# ---------------------------------------------------------------------------
# Each option is displayed as "English / 中文".
# The model expects English for accents and Chinese for dialects.
_CATEGORIES = {
    "Gender / 性别": ["Male / 男", "Female / 女"],
    "Age / 年龄": [
        "Child / 儿童",
        "Teenager / 少年",
        "Young Adult / 青年",
        "Middle-aged / 中年",
        "Elderly / 老年",
    ],
    "Pitch / 音调": [
        "Very Low Pitch / 极低音调",
        "Low Pitch / 低音调",
        "Moderate Pitch / 中音调",
        "High Pitch / 高音调",
        "Very High Pitch / 极高音调",
    ],
    "Style / 风格": ["Whisper / 耳语"],
    "English Accent / 英文口音": [
        "American Accent / 美式口音",
        "Australian Accent / 澳大利亚口音",
        "British Accent / 英国口音",
        "Chinese Accent / 中国口音",
        "Canadian Accent / 加拿大口音",
        "Indian Accent / 印度口音",
        "Korean Accent / 韩国口音",
        "Portuguese Accent / 葡萄牙口音",
        "Russian Accent / 俄罗斯口音",
        "Japanese Accent / 日本口音",
    ],
    "Chinese Dialect / 中文方言": [
        "Henan Dialect / 河南话",
        "Shaanxi Dialect / 陕西话",
        "Sichuan Dialect / 四川话",
        "Guizhou Dialect / 贵州话",
        "Yunnan Dialect / 云南话",
        "Guilin Dialect / 桂林话",
        "Jinan Dialect / 济南话",
        "Shijiazhuang Dialect / 石家庄话",
        "Gansu Dialect / 甘肃话",
        "Ningxia Dialect / 宁夏话",
        "Qingdao Dialect / 青岛话",
        "Northeast Dialect / 东北话",
    ],
}

_ATTR_INFO = {
    "English Accent / 英文口音": "Only effective for English speech. / Chỉ có tác dụng khi đọc tiếng Anh.",
    "Chinese Dialect / 中文方言": "Only effective for Chinese speech. / Chỉ có tác dụng khi đọc tiếng Trung.",
}

# Chữ hiển thị "English / Tiếng Việt". Giá trị gửi vào model vẫn là chuỗi gốc ở _CATEGORIES
# (phương ngữ cần phần tiếng Trung), nên chỉ đổi phần hiển thị.
_VI = {
    "Gender / 性别": "Gender / Giới tính",
    "Male / 男": "Male / Nam",
    "Female / 女": "Female / Nữ",
    "Age / 年龄": "Age / Độ tuổi",
    "Child / 儿童": "Child / Trẻ em",
    "Teenager / 少年": "Teenager / Thiếu niên",
    "Young Adult / 青年": "Young Adult / Thanh niên",
    "Middle-aged / 中年": "Middle-aged / Trung niên",
    "Elderly / 老年": "Elderly / Người cao tuổi",
    "Pitch / 音调": "Pitch / Cao độ giọng",
    "Very Low Pitch / 极低音调": "Very Low Pitch / Rất trầm",
    "Low Pitch / 低音调": "Low Pitch / Trầm",
    "Moderate Pitch / 中音调": "Moderate Pitch / Vừa",
    "High Pitch / 高音调": "High Pitch / Cao",
    "Very High Pitch / 极高音调": "Very High Pitch / Rất cao",
    "Style / 风格": "Style / Phong cách",
    "Whisper / 耳语": "Whisper / Thì thầm",
    "English Accent / 英文口音": "English Accent / Giọng tiếng Anh theo vùng",
    "American Accent / 美式口音": "American Accent / Giọng Mỹ",
    "Australian Accent / 澳大利亚口音": "Australian Accent / Giọng Úc",
    "British Accent / 英国口音": "British Accent / Giọng Anh",
    "Chinese Accent / 中国口音": "Chinese Accent / Giọng Trung Quốc",
    "Canadian Accent / 加拿大口音": "Canadian Accent / Giọng Canada",
    "Indian Accent / 印度口音": "Indian Accent / Giọng Ấn Độ",
    "Korean Accent / 韩国口音": "Korean Accent / Giọng Hàn Quốc",
    "Portuguese Accent / 葡萄牙口音": "Portuguese Accent / Giọng Bồ Đào Nha",
    "Russian Accent / 俄罗斯口音": "Russian Accent / Giọng Nga",
    "Japanese Accent / 日本口音": "Japanese Accent / Giọng Nhật",
    "Chinese Dialect / 中文方言": "Chinese Dialect / Phương ngữ tiếng Trung",
    "Henan Dialect / 河南话": "Henan Dialect / Tiếng Hà Nam",
    "Shaanxi Dialect / 陕西话": "Shaanxi Dialect / Tiếng Thiểm Tây",
    "Sichuan Dialect / 四川话": "Sichuan Dialect / Tiếng Tứ Xuyên",
    "Guizhou Dialect / 贵州话": "Guizhou Dialect / Tiếng Quý Châu",
    "Yunnan Dialect / 云南话": "Yunnan Dialect / Tiếng Vân Nam",
    "Guilin Dialect / 桂林话": "Guilin Dialect / Tiếng Quế Lâm",
    "Jinan Dialect / 济南话": "Jinan Dialect / Tiếng Tế Nam",
    "Shijiazhuang Dialect / 石家庄话": "Shijiazhuang Dialect / Tiếng Thạch Gia Trang",
    "Gansu Dialect / 甘肃话": "Gansu Dialect / Tiếng Cam Túc",
    "Ningxia Dialect / 宁夏话": "Ningxia Dialect / Tiếng Ninh Hạ",
    "Qingdao Dialect / 青岛话": "Qingdao Dialect / Tiếng Thanh Đảo",
    "Northeast Dialect / 东北话": "Northeast Dialect / Tiếng Đông Bắc",
}
_AUTO_LABEL = "Auto / Tự động"

# Chữ có sẵn của Gradio (ô tải file, ghi âm…) không đổi được qua tham số, nên thêm tiếng Việt bằng script nhỏ.
_HEAD_VI = """<script>
(function () {
  const VI = {
    'Drop Audio Here': 'Drop Audio Here / Kéo thả file âm thanh vào đây', '- or -': '- or / hoặc -',
    'Click to Upload': 'Click to Upload / Bấm để chọn file', 'Record': 'Record / Ghi âm',
    'Stop': 'Stop / Dừng', 'Stop recording': 'Stop recording / Dừng ghi âm', 'Pause': 'Pause / Tạm dừng',
    'Resume': 'Resume / Tiếp tục', 'Uploading...': 'Uploading... / Đang tải lên…', 'Error': 'Error / Lỗi',
    'No microphone found': 'No microphone found / Không thấy micro',
    'Info': 'Info / Thông báo', 'Warning': 'Warning / Lưu ý'
  };
  const RX = [[/^processing/, 'processing / đang xử lý'], [/^queue:/, 'queue / hàng chờ:']];
  function tr(n) {
    const raw = n.nodeValue; if (!raw) return; const t = raw.trim(); if (!t) return;
    if (VI[t]) { n.nodeValue = raw.replace(t, VI[t]); return; }
    for (const [re, to] of RX) if (re.test(t) && !t.includes(' / ')) { n.nodeValue = raw.replace(re, to); return; }
  }
  function walk(root) {
    if (root.nodeType === 3) return tr(root);
    if (root.nodeType !== 1 || /^(SCRIPT|STYLE|TEXTAREA|INPUT)$/.test(root.tagName)) return;
    const w = document.createTreeWalker(root, NodeFilter.SHOW_TEXT); let n; while ((n = w.nextNode())) tr(n);
  }
  function start() {
    walk(document.body);
    new MutationObserver(ms => { for (const m of ms) { if (m.type === 'characterData') tr(m.target); else m.addedNodes.forEach(walk); } })
      .observe(document.body, {childList: true, subtree: true, characterData: true});
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start); else start();
})();
</script>"""

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omnivoice-demo",
        description="Launch a Gradio demo for OmniVoice.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="k2-fsa/OmniVoice",
        help="Model checkpoint path or HuggingFace repo id.",
    )
    parser.add_argument(
        "--device", default=None, help="Device to use. Auto-detected if not specified."
    )
    # HUSSIO: cố định một địa chỉ http://localhost:10601 (port 601 bị Chrome/Edge chặn: ERR_UNSAFE_PORT)
    parser.add_argument("--ip", default="127.0.0.1", help="Server IP (default: 127.0.0.1, chỉ máy này).")
    parser.add_argument(
        "--port", type=int, default=10601, help="Server port (default: 10601)."
    )
    parser.add_argument(
        "--mo-trinh-duyet", action="store_true", default=False,
        help="Tự mở trình duyệt khi app sẵn sàng / Open the browser when ready.",
    )
    parser.add_argument(
        "--root-path",
        default=None,
        help="Root path for reverse proxy.",
    )
    parser.add_argument(
        "--share", action="store_true", default=False, help="Create public link."
    )
    parser.add_argument(
        "--no-asr",
        action="store_true",
        default=False,
        help="Skip loading Whisper ASR model. Reference text auto-transcription"
        " will be unavailable.",
    )
    parser.add_argument(
        "--asr-model",
        default="openai/whisper-large-v3-turbo",
        help="ASR model path or HuggingFace repo id"
        " (default: openai/whisper-large-v3-turbo).",
    )
    parser.add_argument(
        "--classic",
        action="store_true",
        default=False,
        help="Open the original demo UI instead of OmniVoice Studio. / Mở giao diện demo gốc thay cho Studio.",
    )
    return parser


# ---------------------------------------------------------------------------
# Build demo
# ---------------------------------------------------------------------------


def build_demo(
    model: OmniVoice,
    checkpoint: str,
    generate_fn=None,
) -> gr.Blocks:
    sampling_rate = model.sampling_rate

    # -- shared generation core --
    def _gen_core(
        text,
        language,
        ref_audio,
        instruct,
        num_step,
        guidance_scale,
        denoise,
        speed,
        duration,
        preprocess_prompt,
        postprocess_output,
        mode,
        ref_text=None,
    ):
        if not text or not text.strip():
            return None, "Please enter the text to synthesize. / Vui lòng nhập nội dung cần đọc."

        gen_config = OmniVoiceGenerationConfig(
            num_step=int(num_step or 32),
            guidance_scale=float(guidance_scale) if guidance_scale is not None else 2.0,
            denoise=bool(denoise) if denoise is not None else True,
            preprocess_prompt=bool(preprocess_prompt),
            postprocess_output=bool(postprocess_output),
        )

        lang = language if (language and language != "Auto") else None

        kw: Dict[str, Any] = dict(
            text=text.strip(), language=lang, generation_config=gen_config
        )

        if speed is not None and float(speed) != 1.0:
            kw["speed"] = float(speed)
        if duration is not None and float(duration) > 0:
            kw["duration"] = float(duration)

        if mode == "clone":
            if not ref_audio:
                return None, "Please upload a reference audio. / Vui lòng tải giọng mẫu lên."
            kw["voice_clone_prompt"] = model.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text,
            )

        if instruct and instruct.strip():
            kw["instruct"] = instruct.strip()

        try:
            audio = model.generate(**kw)
        except Exception as e:
            return None, f"Error / Lỗi: {type(e).__name__}: {e}"

        waveform = (audio[0] * 32767).astype(np.int16)
        return (sampling_rate, waveform), "Done. / Xong."

    # Allow external wrappers (e.g. spaces.GPU for ZeroGPU Spaces)
    _gen = generate_fn if generate_fn is not None else _gen_core

    # =====================================================================
    # UI
    # =====================================================================
    theme = gr.themes.Soft(
        font=["Inter", "Arial", "sans-serif"],
    )
    css = """
    .gradio-container {max-width: 100% !important; font-size: 16px !important;}
    .gradio-container h1 {font-size: 1.5em !important;}
    .gradio-container .prose {font-size: 1.1em !important;}
    .compact-audio audio {height: 60px !important;}
    .compact-audio .waveform {min-height: 80px !important;}
    """

    # Reusable: language dropdown component
    def _lang_dropdown(label="Language (optional) / Ngôn ngữ (không bắt buộc)", value="Auto"):
        return gr.Dropdown(
            label=label,
            choices=[(_AUTO_LABEL, "Auto")] + _ALL_LANGUAGES[1:],
            value=value,
            allow_custom_value=False,
            interactive=True,
            info="Keep as Auto to auto-detect the language. / Để Auto thì máy tự nhận ngôn ngữ.",
        )

    # Reusable: optional generation settings accordion
    def _gen_settings():
        with gr.Accordion("Generation Settings (optional) / Cài đặt tạo giọng (không bắt buộc)", open=False):
            sp = gr.Slider(
                0.5,
                1.5,
                value=1.0,
                step=0.05,
                label="Speed / Tốc độ",
                info="1.0 = normal. >1 faster, <1 slower. Ignored if Duration is set. / "
                "1,0 = bình thường. >1 nhanh hơn, <1 chậm hơn. Bỏ qua nếu đã điền Độ dài.",
            )
            du = gr.Number(
                value=None,
                label="Duration (seconds) / Độ dài (giây)",
                info=(
                    "Leave empty to use speed. Set a fixed duration to override speed. / "
                    "Để trống thì dùng Tốc độ. Điền số giây để ép độ dài cố định."
                ),
            )
            ns = gr.Slider(
                4,
                64,
                value=32,
                step=1,
                label="Inference Steps / Số bước tạo",
                info="Default: 32. Lower = faster, higher = better quality. / "
                "Mặc định 32. Thấp = nhanh hơn, cao = chất lượng tốt hơn.",
            )
            dn = gr.Checkbox(
                label="Denoise / Lọc tạp âm",
                value=True,
                info="Default: enabled. Uncheck to disable denoising. / Mặc định bật. Bỏ tích để tắt lọc tạp âm.",
            )
            gs = gr.Slider(
                0.0,
                4.0,
                value=2.0,
                step=0.1,
                label="Guidance Scale (CFG) / Mức bám chỉ dẫn (CFG)",
                info="Default: 2.0. / Mặc định 2,0.",
            )
            pp = gr.Checkbox(
                label="Preprocess Prompt / Xử lý giọng mẫu",
                value=True,
                info="apply silence removal and trimming to the reference "
                "audio, add punctuation in the end of reference text (if not already) / "
                "Cắt khoảng lặng và cắt gọn giọng mẫu, thêm dấu câu cuối lời mẫu (nếu chưa có).",
            )
            po = gr.Checkbox(
                label="Postprocess Output / Xử lý kết quả",
                value=True,
                info="Remove long silences from generated audio. / Bỏ khoảng lặng dài trong giọng tạo ra.",
            )
        return ns, gs, dn, sp, du, pp, po

    with gr.Blocks(theme=theme, css=css, title="OmniVoice Demo") as demo:
        gr.Markdown(
            """
# OmniVoice Demo

State-of-the-art text-to-speech model for **600+ languages**, supporting: /
Mô hình chuyển văn bản thành giọng nói cho **hơn 600 ngôn ngữ**, gồm:

- **Voice Clone / Nhân bản giọng** — Clone any voice from a reference audio / Đọc theo giọng của một file giọng mẫu
- **Voice Design / Thiết kế giọng** — Create custom voices with speaker attributes / Tạo giọng mới theo giới tính, tuổi, cao độ…

Built with [OmniVoice](https://github.com/k2-fsa/OmniVoice)
by Xiaomi AI Lab Next-gen Kaldi team. / Phát triển bởi nhóm Next-gen Kaldi, Xiaomi AI Lab.
"""
        )

        with gr.Tabs():
            # ==============================================================
            # Voice Clone
            # ==============================================================
            with gr.TabItem("Voice Clone / Nhân bản giọng"):
                with gr.Row():
                    with gr.Column(scale=1):
                        vc_text = gr.Textbox(
                            label="Text to Synthesize / Văn bản cần đọc",
                            lines=4,
                            placeholder="Enter the text you want to synthesize... / Nhập nội dung cần đọc...",
                        )
                        vc_ref_audio = gr.Audio(
                            label="Reference Audio / Giọng mẫu",
                            type="filepath",
                            elem_classes="compact-audio",
                        )
                        gr.Markdown(
                            "<span style='font-size:0.85em;color:#888;'>"
                            "Recommended: 3–10 seconds audio. / Nên dùng đoạn 3–10 giây. "
                            "</span>"
                        )
                        vc_ref_text = gr.Textbox(
                            label=("Reference Text (optional) / Lời của giọng mẫu (không bắt buộc)"),
                            lines=2,
                            placeholder="Transcript of the reference audio. Leave empty"
                            " to auto-transcribe via ASR models. / Gõ đúng lời trong giọng mẫu."
                            " Để trống thì máy tự nghe và chép lời.",
                        )
                        vc_lang = _lang_dropdown()
                        with gr.Accordion("Instruct (optional) / Chỉ dẫn (không bắt buộc)", open=False):
                            vc_instruct = gr.Textbox(label="Instruct / Chỉ dẫn", lines=2)
                        (
                            vc_ns,
                            vc_gs,
                            vc_dn,
                            vc_sp,
                            vc_du,
                            vc_pp,
                            vc_po,
                        ) = _gen_settings()
                        vc_btn = gr.Button("Generate / Tạo giọng", variant="primary")
                    with gr.Column(scale=1):
                        vc_audio = gr.Audio(
                            label="Output Audio / Kết quả",
                            type="numpy",
                        )
                        vc_status = gr.Textbox(label="Status / Trạng thái", lines=2)

                def _clone_fn(
                    text, lang, ref_aud, ref_text, instruct, ns, gs, dn, sp, du, pp, po
                ):
                    return _gen(
                        text,
                        lang,
                        ref_aud,
                        instruct,
                        ns,
                        gs,
                        dn,
                        sp,
                        du,
                        pp,
                        po,
                        mode="clone",
                        ref_text=ref_text or None,
                    )

                vc_btn.click(
                    _clone_fn,
                    inputs=[
                        vc_text,
                        vc_lang,
                        vc_ref_audio,
                        vc_ref_text,
                        vc_instruct,
                        vc_ns,
                        vc_gs,
                        vc_dn,
                        vc_sp,
                        vc_du,
                        vc_pp,
                        vc_po,
                    ],
                    outputs=[vc_audio, vc_status],
                )

            # ==============================================================
            # Voice Design
            # ==============================================================
            with gr.TabItem("Voice Design / Thiết kế giọng"):
                with gr.Row():
                    with gr.Column(scale=1):
                        vd_text = gr.Textbox(
                            label="Text to Synthesize / Văn bản cần đọc",
                            lines=4,
                            placeholder="Enter the text you want to synthesize... / Nhập nội dung cần đọc...",
                        )
                        vd_lang = _lang_dropdown()

                        _AUTO = "Auto"
                        vd_groups = []
                        for _cat, _choices in _CATEGORIES.items():
                            vd_groups.append(
                                gr.Dropdown(
                                    label=_VI.get(_cat, _cat),
                                    choices=[(_AUTO_LABEL, _AUTO)]
                                    + [(_VI.get(c, c), c) for c in _choices],
                                    value=_AUTO,
                                    info=_ATTR_INFO.get(_cat),
                                )
                            )

                        (
                            vd_ns,
                            vd_gs,
                            vd_dn,
                            vd_sp,
                            vd_du,
                            vd_pp,
                            vd_po,
                        ) = _gen_settings()
                        vd_btn = gr.Button("Generate / Tạo giọng", variant="primary")
                    with gr.Column(scale=1):
                        vd_audio = gr.Audio(
                            label="Output Audio / Kết quả",
                            type="numpy",
                        )
                        vd_status = gr.Textbox(label="Status / Trạng thái", lines=2)

                def _build_instruct(groups):
                    """Extract instruct text from UI dropdowns.

                    Language unification and validation is handled by
                    _resolve_instruct inside _preprocess_all.
                    """
                    selected = [g for g in groups if g and g != "Auto"]
                    if not selected:
                        return None
                    parts = []
                    for v in selected:
                        if " / " in v:
                            en, zh = v.split(" / ", 1)
                            # Dialects have no English equivalent
                            if "Dialect" in v.split(" / ")[0]:
                                parts.append(zh.strip())
                            else:
                                parts.append(en.strip())
                        else:
                            parts.append(v)
                    return ", ".join(parts)

                def _design_fn(text, lang, ns, gs, dn, sp, du, pp, po, *groups):
                    return _gen(
                        text,
                        lang,
                        None,
                        _build_instruct(groups),
                        ns,
                        gs,
                        dn,
                        sp,
                        du,
                        pp,
                        po,
                        mode="design",
                    )

                vd_btn.click(
                    _design_fn,
                    inputs=[
                        vd_text,
                        vd_lang,
                        vd_ns,
                        vd_gs,
                        vd_dn,
                        vd_sp,
                        vd_du,
                        vd_pp,
                        vd_po,
                    ]
                    + vd_groups,
                    outputs=[vd_audio, vd_status],
                )

    return demo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)

    device = args.device or get_best_device()

    checkpoint = args.model
    if not checkpoint:
        parser.print_help()
        return 0
    logging.info(f"Loading model from {checkpoint}, device={device} ...")
    model = OmniVoice.from_pretrained(
        checkpoint,
        device_map=device,
        dtype=torch.float16,
        load_asr=not args.no_asr,
        asr_model_name=args.asr_model,
    )
    print("Model loaded.")

    if args.classic:
        demo = build_demo(model, checkpoint)
        demo.queue().launch(
            server_name=args.ip,
            server_port=args.port,
            share=args.share,
            root_path=args.root_path,
            head=_HEAD_VI,
        )
        return 0

    # HUSSIO Voice (omnivoice/studio): thư viện giọng, nghe thử, đo độ giống, lịch sử; giao diện Sáng / Tối
    from omnivoice.studio.config import DATA
    from omnivoice.studio.theme import ASSETS, CSS, THEME_HEAD, studio_theme
    from omnivoice.studio.ui import build_studio

    demo = build_studio(model)
    print(f"HUSSIO Voice sẵn sàng / ready: http://localhost:{args.port}")
    demo.queue().launch(
        server_name=args.ip,
        server_port=args.port,
        inbrowser=args.mo_trinh_duyet,
        share=args.share,
        root_path=args.root_path,
        theme=studio_theme(),
        css=CSS,
        head=THEME_HEAD + _HEAD_VI,
        allowed_paths=[str(DATA), str(ASSETS)],
        footer_links=[],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
