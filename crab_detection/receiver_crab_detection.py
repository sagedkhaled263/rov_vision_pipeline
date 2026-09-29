import cv2, socket, struct, numpy as np, threading, time
from ultralytics import YOLO

MODEL_PATH      = "crab_detection.pt"
TARGET_CLASS    = "European green crab"
CONF_THRESHOLD  = 0.8
SENDER_HOST     = "192.168.1.100"
PORT            = 9002
RECONNECT_DELAY = 3
FRAME_W, FRAME_H = 640, 480

COLOR_INVASIVE = (0, 255,   0)
COLOR_HUD_BG   = (0,   0,   0)
COLOR_HUD_TEXT = (0, 255,   0)

_frame_lock   = threading.Lock()
_latest_frame = None
_stop_event   = threading.Event()


def recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed")
        buf.extend(chunk)
    return bytes(buf)


def receiver_thread(sock):
    global _latest_frame
    while not _stop_event.is_set():
        try:
            header      = recv_exact(sock, 8)
            payload_len = struct.unpack("Q", header)[0]
            payload     = recv_exact(sock, payload_len)
        except ConnectionError:
            break
        arr   = np.frombuffer(payload, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is not None:
            with _frame_lock:
                _latest_frame = frame


def run_detection(frame, model):
    # No imgsz override — use native 320x240; no .plot() — draw with OpenCV
    results = model.predict(frame, conf=CONF_THRESHOLD,
                            classes=[0], verbose=False)

    invasive_count = 0
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            invasive_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_INVASIVE, 2)
            cv2.putText(frame, f"INVASIVE {conf:.2f}", (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_INVASIVE, 1)

    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 36), COLOR_HUD_BG, -1)
    cv2.putText(frame, f"Green Crab Count: {invasive_count}", (8, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_HUD_TEXT, 1)

    return frame, invasive_count


def main():
    global _latest_frame
    print("[Receiver] Loading YOLO model …")
    model = YOLO(MODEL_PATH)
    print("[Receiver] Model loaded.")
    print(model.overrides)

    while True:
        print(f"[Receiver] Connecting to {SENDER_HOST}:{PORT} …")
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((SENDER_HOST, PORT))
        except OSError as e:
            print(f"[Receiver] Failed: {e}  — retry in {RECONNECT_DELAY}s")
            time.sleep(RECONNECT_DELAY)
            continue

        print("[Receiver] Connected!")
        _stop_event.clear()
        with _frame_lock:
            _latest_frame = None

        t = threading.Thread(target=receiver_thread, args=(sock,), daemon=True)
        t.start()

        try:
            while True:
                with _frame_lock:
                    frame = _latest_frame
                    _latest_frame = None

                if frame is None:
                    time.sleep(0.005)
                    continue

                annotated, _ = run_detection(frame, model)

                cv2.imshow("MATE ROV - Crab Detector", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    _stop_event.set()
                    sock.close()
                    cv2.destroyAllWindows()
                    return
        except KeyboardInterrupt:
            pass
        finally:
            _stop_event.set()
            sock.close()
            t.join(timeout=2)

        print(f"[Receiver] Reconnecting in {RECONNECT_DELAY}s …")
        time.sleep(RECONNECT_DELAY)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
