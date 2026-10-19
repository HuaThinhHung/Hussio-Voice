# HUSSIO Voice

Bản chỉnh sửa nội bộ của [OmniVoice](https://github.com/k2-fsa/OmniVoice): một app tạo giọng đọc tiếng Việt với
thư viện giọng, nghe thử, đo độ giống, lịch sử, giao diện Sáng/Tối. Repo **riêng tư**, không công khai.

- Code gốc OmniVoice: Apache-2.0 (giữ nguyên `LICENSE`). Phần thêm: `omnivoice/studio/`, `omnivoice/cli/demo.py`.
- Model OmniVoice: **CC-BY-NC (phi thương mại)**. Không dùng âm thanh tạo ra cho quảng cáo hay video bán hàng.
- Chỉ lưu giọng của người đã đồng ý.

## Cài trên máy mới

Cần: GPU NVIDIA (đã chạy tốt trên RTX 3060 12 GB) + driver, [git](https://git-scm.com), [uv](https://docs.astral.sh/uv/),
[ffmpeg](https://ffmpeg.org) (để xuất MP3), khoảng 15 GB ổ trống. Lần chạy đầu tự tải model (~9,6 GB) từ Hugging Face.

```
git clone <link repo riêng tư> HUSSIO-Voice
cd HUSSIO-Voice
uv sync
```

## Chạy

| Mục đích | Lệnh | Mở |
|---|---|---|
| Dùng trên chính máy này | `uv run omnivoice-demo` | http://localhost:10601 |
| Cho máy khác trong mạng công ty dùng | `uv run omnivoice-demo --ip 0.0.0.0` | http://IP-máy-chạy:10601 |

Tắt: `Ctrl + C` trong cửa sổ Terminal.

Khi mở cho mạng công ty:
- Windows hỏi tường lửa thì chỉ cho phép **Private network** (mạng nội bộ). Không mở cổng 10601 ra internet.
- Chưa có đăng nhập: ai trong mạng cũng dùng và xoá được giọng (xoá mềm, khôi phục được ở Thư viện → Đã xoá).
- Nút Ghi âm chỉ chạy trên `localhost` hoặc trang HTTPS; vào bằng `http://IP` thì dùng nút tải file lên.
- Một GPU làm một việc một lúc: nhiều người bấm cùng lúc sẽ xếp hàng.

## Dữ liệu (không có trong repo)

`data/studio/` (bị `.gitignore` chặn): `voices/` thư viện giọng, `outputs/` file kết quả theo ngày, `history.db` lịch sử,
`_da-xoa/` giọng đã xoá. Muốn chuyển thư viện sang máy khác thì chép nguyên thư mục `data/studio/` sang.
