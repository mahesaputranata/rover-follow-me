# rover_ws/src/rover_bridge/rover_bridge/camera_service.py
import cv2
import time
import threading
import socket
import pickle
import struct
import signal


class CameraService:
    def __init__(self, camera_index: int, width: int, height: int, socket_port: int = 6001):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.socket_port = socket_port

        self._lock = threading.Lock()
        self._latest_frame = None
        self._running = False
        self._recording = False
        self._video_writer = None

    def start(self):
        self._running = True
        threading.Thread(target=self._capture_loop, daemon=True).start()
        threading.Thread(target=self._serve_frames, daemon=True).start()
        print(f"[CameraService] started on port {self.socket_port}")

    def _capture_loop(self):
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        while self._running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            with self._lock:
                self._latest_frame = frame
                if self._recording and self._video_writer:
                    self._video_writer.write(frame)
        cap.release()

    def get_latest_frame(self):
        with self._lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def start_recording(self, output_path: str, fps: int = 15):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        with self._lock:
            self._video_writer = cv2.VideoWriter(output_path, fourcc, fps, (self.width, self.height))
            self._recording = True

    def stop_recording(self):
        with self._lock:
            self._recording = False
            if self._video_writer:
                self._video_writer.release()
                self._video_writer = None

    def _serve_frames(self):
        """Simple local TCP server so other processes (VisionLoop, snapshot requests) can pull frames."""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", self.socket_port))
        server.listen(5)
        while self._running:
            conn, _ = server.accept()
            try:
                frame = self.get_latest_frame()
                if frame is not None:
                    data = pickle.dumps(frame)
                    conn.sendall(struct.pack(">I", len(data)) + data)
            finally:
                conn.close()

if __name__ == "__main__":
    from rover_bridge.vision.config.settings import Settings
    settings = Settings()
    svc = CameraService(
        camera_index=settings.CAMERA_INDEX,
        width=settings.FRAME_WIDTH,
        height=settings.FRAME_HEIGHT,
    )
    svc.start()

    def _shutdown(signum, frame):
        print("[CameraService] shutting down...")
        svc._running = False
        time.sleep(0.3)  # let the capture loop notice and release the camera cleanly
        exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    while True:
        time.sleep(1)