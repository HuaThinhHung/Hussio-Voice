"""Đường dẫn dữ liệu, câu nghe thử và các hằng số dùng chung."""
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]          # D:\Du_An\OmniVoice
DATA = REPO / "data" / "studio"                      # thư mục data/ đã nằm trong .gitignore
VOICES = DATA / "voices"                              # mỗi giọng một thư mục: ref.wav, meta.json, prompt.pt, sample.wav
OUTPUTS = DATA / "outputs"                            # kết quả theo ngày
TRASH = DATA / "_da-xoa"                              # giọng / file đã xoá trong app (khôi phục được)
SETTINGS_FILE = DATA / "settings.json"
DICT_FILE = DATA / "dictionary.json"
HISTORY_DB = DATA / "history.db"

SR_OUT = 24000                                        # tần số mẫu của OmniVoice
SV_MODEL = "microsoft/wavlm-base-plus-sv"             # model đo độ giống giọng

# Câu nghe thử chung cho mọi giọng, để so giọng với nhau trên cùng một câu
TEST_SENTENCE = ("Xin chào, đây là bản nghe thử giọng đọc. "
                 "Sơ mi vải cốt tông thoáng mát, mặc đi làm rất chỉn chu.")
# Câu làm giọng mẫu cho các giọng thiết kế
DESIGN_REF_SENTENCE = ("Chào bạn, mình đang đọc thử một đoạn ngắn để làm giọng mẫu. "
                       "Giọng nói rõ ràng, tốc độ vừa phải, ngắt nghỉ tự nhiên.")

# Nhãn chọn được khi thêm giọng
GENDERS = ["Nam", "Nữ", "Khác"]
AGES = ["Trẻ em", "Thiếu niên", "Thanh niên", "Trung niên", "Cao tuổi"]
REGIONS = ["Bắc", "Trung", "Nam", "Chưa rõ"]
SOURCES = {"upload": "Tải lên", "record": "Ghi âm", "design": "Thiết kế", "restore": "Khôi phục"}

# Mẫu cài đặt tạo giọng
PRESETS = {
    "Giống giọng mẫu nhất / Max similarity": {"num_step": 48, "guidance_scale": 2.0, "takes": 3},
    "Cân bằng / Balanced": {"num_step": 32, "guidance_scale": 2.0, "takes": 1},
    "Nhanh / Fast": {"num_step": 16, "guidance_scale": 2.0, "takes": 1},
}
DEFAULT_PRESET = "Cân bằng / Balanced"

_WINGET_FF = Path.home() / r"AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin"
FFMPEG = shutil.which("ffmpeg") or str(_WINGET_FF / "ffmpeg.exe")


def ensure_dirs() -> None:
    for d in (VOICES, OUTPUTS, TRASH):
        d.mkdir(parents=True, exist_ok=True)
