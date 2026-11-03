"""VI-02 / VI-06: Scene Cut Detector — phát hiện chuyển cảnh (shot transition).

Trong camera giám sát thực tế (CCTV / RTSP), camera là cố định nên không có chuyển cảnh.
Tuy nhiên với các video kiểm thử (clip dựng, video truyền thông, video demo), các cú cắt cảnh
(scene cut) khiến tracker (ByteTrack/BoT-SORT) gán nhầm track_id của người ở cảnh trước cho
người ở cảnh sau.

Phát hiện chuyển cảnh giúp:
1. Reset tracker (detector.reset_tracker()) để cấp ID mới cho người ở cảnh mới.
2. Reset chuỗi xác nhận vi phạm (confirmation.reset()) để tránh cộng dồn thời gian vi phạm
   vắt ngang qua cú chuyển cảnh, ngăn chặn triệt để tình trạng ảnh đối chứng và ảnh vi phạm
   chụp ở hai cảnh quay khác nhau.
"""
import cv2
import numpy as np


class SceneCutDetector:
    """Bộ phát hiện chuyển cảnh siêu nhẹ dựa trên độ biến thiên cường độ xám."""

    def __init__(self, threshold: float = 35.0, sample_size=(160, 90)):
        self.threshold = float(threshold)
        self.sample_size = sample_size
        self._prev_small_gray = None

    def check(self, frame: np.ndarray) -> bool:
        """Kiểm tra frame hiện tại có phải là điểm bắt đầu của một cảnh quay mới không.

        Thực thi siêu nhẹ (< 0.2ms) do downsample về độ phân giải thấp (160x90).
        Bảo vệ phòng thủ chống frame rỗng, kích thước bất thường hoặc lỗi định dạng kênh màu.
        """
        if frame is None or getattr(frame, "size", 0) == 0 or len(getattr(frame, "shape", ())) < 2:
            return False

        try:
            small = cv2.resize(frame, self.sample_size, interpolation=cv2.INTER_NEAREST)
            if len(small.shape) == 3 and small.shape[2] >= 3:
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            elif len(small.shape) == 2:
                gray = small
            else:
                return False

            is_cut = False
            if self._prev_small_gray is not None:
                diff = float(np.mean(np.abs(gray.astype(np.float32) - self._prev_small_gray.astype(np.float32))))
                if diff >= self.threshold:
                    is_cut = True
            self._prev_small_gray = gray
            return is_cut
        except Exception:
            return False

    def reset(self):
        """Reset trạng thái cảnh quay."""
        self._prev_small_gray = None

