"""VI-02: Detector — tải model YOLO, chạy inference trên 1 frame.

Đổi model bằng cách sửa MODEL_PATH trong config/env; pipeline không thay đổi.
"""
import torch

from ultralytics import YOLO

from app import config
from app.detection.adapter import normalize_results


class DetectorError(Exception):
    """Lỗi tải model hoặc chạy inference."""


def auto_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Detector:
    def __init__(self, model_path=None, imgsz=None, conf=None, device=None, track=True):
        self.model_path = model_path or config.MODEL_PATH
        self.imgsz = imgsz or config.MODEL_IMGSZ
        self.conf = conf if conf is not None else config.CONFIDENCE_THRESHOLD
        self.device = device or config.DEVICE or auto_device()
        self.track = track
        self._model = None

    def load(self):
        try:
            self._model = YOLO(self.model_path)
        except Exception as e:
            raise DetectorError(f"Không tải được model '{self.model_path}': {e}")
        return self

    def detect(self, frame):
        """Chạy model trên 1 frame, trả về list[Detection] theo tọa độ frame gốc.

        Khi self.track=True dùng ByteTrack (có sẵn trong ultralytics) và
        persist=True để track_id được giữ ổn định giữa các lần gọi.
        """
        if self._model is None:
            self.load()
        kwargs = dict(imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)
        if self.track:
            kwargs["persist"] = True
            results = self._model.track(frame, **kwargs)
        else:
            results = self._model.predict(frame, **kwargs)
        h, w = frame.shape[:2]
        return normalize_results(results[0], w, h)
