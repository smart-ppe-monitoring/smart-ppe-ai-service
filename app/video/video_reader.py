import re
"""VI-01: Đọc video từ file / webcam / RTSP.

Xử lý: lỗi mở nguồn, frame rỗng, hết video, giải phóng tài nguyên.
Dùng context manager để luôn release capture kể cả khi có exception.
"""
import time

import cv2

from app import config


class VideoSourceError(Exception):
    """Không mở được nguồn video hoặc lỗi nghiêm trọng khi đọc."""



def mask_source(source) -> str:
    """Mask credentials in RTSP/HTTP URL string."""
    if isinstance(source, str):
        return re.sub(r"://([^:@\s]+):(.+)@([^/@\s]+)", r"://\1:***@\3", source)
    return str(source)

class VideoReader:
    """Iterator đọc frame tuần tự từ một nguồn video."""

    def __init__(self, source, empty_frame_tolerance=None):
        resolved = config.resolve_source(source)
        self.source = resolved if resolved is not None else config.DEFAULT_SOURCE
        if not self.source and self.source != 0:
            raise VideoSourceError(
                "Chưa cấu hình nguồn video. Đặt biến môi trường VIDEO_SOURCE "
                "hoặc truyền --source."
            )
        self.empty_frame_tolerance = (
            empty_frame_tolerance
            if empty_frame_tolerance is not None
            else config.EMPTY_FRAME_TOLERANCE
        )
        self._cap = None
        src_str = str(self.source).strip().lower()
        stream_protocols = ("rtsp://", "rtsps://", "http://", "https://", "rtmp://", "rtmps://", "rtsp:")
        self._is_stream = (
            isinstance(self.source, int)
            or src_str.startswith(stream_protocols)
            or ".m3u8" in src_str
        )
        self._opened = False
        self._frames_read = 0
        self._empty_streak = 0

    # ---------- mở / đóng ----------

    def open(self):
        if self._opened:
            return self
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            self._cap.release()
            raise VideoSourceError(
                f"Không mở được nguồn video: {mask_source(self.source)!r}. "
                "Kiểm tra đường dẫn file, quyền truy cập hoặc URL RTSP/index webcam."
            )
        self._opened = True
        return self

    def release(self):
        if self._cap is not None:
            self._cap.release()
        self._cap = None
        self._opened = False

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc, tb):
        self.release()

    # ---------- thông tin nguồn ----------

    @property
    def fps(self):
        fps = self._cap.get(cv2.CAP_PROP_FPS) if self._opened else 0.0
        return fps if fps and fps > 0 else config.FALLBACK_FPS

    @property
    def frame_count(self):
        if not self._opened or self._is_stream:
            return None
        n = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return n if n > 0 else None

    @property
    def size(self):
        if not self._opened:
            return None
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)

    # ---------- đọc frame ----------

    def read(self):
        """Trả về (frame_number, frame); frame=None khi hết video."""
        if not self._opened:
            raise VideoSourceError("Chưa mở nguồn video — gọi open() hoặc dùng with.")

        while True:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                self._empty_streak += 1
                if not self._is_stream or self._empty_streak > self.empty_frame_tolerance:
                    return None  # hết file, hoặc stream mất tín hiệu quá lâu
                continue  # stream: bỏ qua frame rỗng thoáng qua
            self._empty_streak = 0
            self._frames_read += 1
            return self._frames_read, frame

    def __iter__(self):
        return self

    def __next__(self):
        item = self.read()
        if item is None:
            raise StopIteration
        return item


def read_and_report(source, show_every=100):
    """Chạy thử VI-01: đọc hết nguồn, in thông tin và tốc độ đọc thực tế."""
    with VideoReader(source) as reader:
        print(f"Nguồn       : {mask_source(reader.source)}")
        print(f"Kích thước  : {reader.size[0]}x{reader.size[1]}" if reader.size else "Kích thước  : N/A")
        print(f"FPS khai báo: {reader.fps:.2f} | Tổng frame (file): {reader.frame_count}")
        t0 = time.time()
        last = None
        for num, frame in reader:
            last = num
            if show_every and num % show_every == 0:
                print(f"  frame {num:>6} — {frame.shape[1]}x{frame.shape[0]}")
        elapsed = time.time() - t0
        n = last or 0
        print(f"Đọc xong    : {n} frame trong {elapsed:.2f}s "
              f"({n / elapsed:.1f} frame/s thực tế)" if n else "Không đọc được frame nào.")
