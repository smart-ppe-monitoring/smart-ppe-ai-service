import numpy as np
import pytest

from app.video.scene_cut import SceneCutDetector
from app.rules.confirmation import ViolationConfirmation
from app.detection.detector import Detector


def test_scene_cut_detector_continuous_and_cut():
    detector = SceneCutDetector(threshold=35.0)

    # Khung đen liên tục -> không có chuyển cảnh
    f_black_1 = np.zeros((100, 100, 3), dtype=np.uint8)
    f_black_2 = np.zeros((100, 100, 3), dtype=np.uint8)
    assert not detector.check(f_black_1)
    assert not detector.check(f_black_2)

    # Chuyển đột ngột sang khung trắng -> chuyển cảnh
    f_white = np.full((100, 100, 3), 255, dtype=np.uint8)
    assert detector.check(f_white)

    # Khung trắng tiếp theo -> cùng cảnh mới, không chuyển cảnh
    f_white_2 = np.full((100, 100, 3), 250, dtype=np.uint8)
    assert not detector.check(f_white_2)


def test_confirmation_reset_clears_streaks():
    conf = ViolationConfirmation(duration_s=2.0)
    conf._streaks[(1, "no_helmet")] = (0.0, 1.5, 1.5, True)
    conf._start_frames[(1, "no_helmet")] = ("dummy_frame", (10, 10, 50, 50))
    conf._compliant_frames[1] = ("dummy_compliant", None)
    conf._emitted.add((1, "no_helmet"))

    conf.reset()

    assert len(conf._streaks) == 0
    assert len(conf._start_frames) == 0
    assert len(conf._compliant_frames) == 0
    assert len(conf._emitted) == 0


def test_detector_reset_tracker_safe_call():
    det = Detector()
    # Chưa load model hoặc không có predictor -> gọi không được văng lỗi
    det.reset_tracker()


def test_scene_cut_detector_defensive_frame_handling():
    detector = SceneCutDetector()

    # None frame
    assert detector.check(None) is False

    # Empty frame (size 0)
    assert detector.check(np.zeros((0, 0, 3), dtype=np.uint8)) is False
    assert detector.check(np.array([])) is False

    # 1D array
    assert detector.check(np.array([1, 2, 3], dtype=np.uint8)) is False

    # 2D grayscale frame (should be accepted without crashing)
    f_gray_1 = np.zeros((100, 100), dtype=np.uint8)
    assert detector.check(f_gray_1) is False

    f_gray_cut = np.full((100, 100), 255, dtype=np.uint8)
    assert detector.check(f_gray_cut) is True


def test_video_reader_is_stream_property():
    from app.video.video_reader import VideoReader

    # Video file path -> is_stream is False
    reader_file = VideoReader("sample.mp4")
    assert reader_file.is_stream is False

    # RTSP url -> is_stream is True
    reader_rtsp = VideoReader("rtsp://admin:pass@192.168.1.100:554/live")
    assert reader_rtsp.is_stream is True


def test_run_pipeline_frame_loop_executes_each_frame(tmp_path):
    from unittest.mock import MagicMock, patch
    from app.main import run_pipeline

    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frames = [(1, dummy_frame), (2, dummy_frame), (3, dummy_frame)]

    mock_reader = MagicMock()
    mock_reader.__enter__.return_value = mock_reader
    mock_reader.__exit__.return_value = None
    mock_reader.__iter__.return_value = iter(frames)
    mock_reader.fps = 25.0
    mock_reader.size = (100, 100)
    mock_reader.is_stream = False

    mock_detector = MagicMock()
    mock_detector.load.return_value = mock_detector
    mock_detector.model_path = "mock.pt"
    mock_detector.device = "cpu"
    mock_detector.conf = 0.25
    mock_detector.detect.return_value = []

    with patch("app.main.VideoReader", return_value=mock_reader), \
         patch("app.main.Detector", return_value=mock_detector), \
         patch("app.main.ViolationConfirmation") as mock_conf_cls, \
         patch("app.main.flush_producer"), \
         patch("app.main.flush_offline_buffer"):

        mock_conf_instance = MagicMock()
        mock_conf_instance.update.return_value = []
        mock_conf_cls.return_value = mock_conf_instance

        ret = run_pipeline(source="test.mp4", out_dir=str(tmp_path))

        assert ret == 0
        # Verify detect and confirmation.update were called for EVERY frame in the loop
        assert mock_detector.detect.call_count == 3
        assert mock_conf_instance.update.call_count == 3




