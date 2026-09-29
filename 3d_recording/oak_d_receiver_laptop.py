#!/usr/bin/env python3
"""
OAK-D RGB Receiver — Connects to Pi at 192.168.1.100:5001
Records 1080p video for 3D reconstruction.

Controls:
  SPACE — Start/Stop recording
  Q     — Quit
"""

import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import socket
import struct
import numpy as np
import time


def recv_exact(sock, size):
    data = b''
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            return None
        data += chunk
    return data


def main():
    PI_IP = "192.168.1.100"
    PORT = 5001

    recording = False
    writer = None
    start_time = 0
    video_count = 0
    save_dir = os.path.join(os.path.expanduser("~"), "3d_video")
    os.makedirs(save_dir, exist_ok=True)

    print(f"Connecting to {PI_IP}:{PORT}...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((PI_IP, PORT))
    print("Connected!")
    print("SPACE = Start/Stop recording | Q = Quit")
    print(f"Videos save to: {save_dir}")

    cv2.namedWindow("OAK-D RGB", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("OAK-D RGB", 1280, 720)

    while True:
        size_data = recv_exact(sock, 4)
        if size_data is None:
            print("Connection lost.")
            break

        frame_size = struct.unpack('>I', size_data)[0]
        frame_data = recv_exact(sock, frame_size)
        if frame_data is None:
            print("Connection lost.")
            break

        frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            continue

        h, w = frame.shape[:2]
        display = frame.copy()

        if recording:
            elapsed = time.time() - start_time
            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            ms = int((elapsed % 1) * 100)

            scale = max(w / 640, 1.0)
            thick = max(int(2 * scale), 2)
            radius = max(int(10 * scale), 10)

            cv2.circle(display, (int(30 * scale), int(30 * scale)), radius, (0, 0, 255), -1)
            cv2.putText(display, f"REC {mins:02d}:{secs:02d}.{ms:02d}",
                        (int(50 * scale), int(38 * scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale, (0, 0, 255), thick)

            writer.write(frame)
        else:
            scale = max(w / 640, 1.0)
            thick = max(int(2 * scale), 2)
            cv2.putText(display, "SPACE to record",
                        (int(10 * scale), int(38 * scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale, (0, 255, 0), thick)

        cv2.imshow("OAK-D RGB", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            if not recording:
                filename = os.path.join(save_dir, f"video_{video_count}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(filename, fourcc, 30, (w, h))
                start_time = time.time()
                recording = True
                print(f"\n🔴 Recording: {filename}")
                print(f"   Resolution: {w}x{h}")
            else:
                elapsed = time.time() - start_time
                writer.release()
                writer = None
                recording = False
                video_count += 1
                print(f"⏹  Stopped — {elapsed:.2f}s")
                print(f"   Saved: {filename}")

        elif key == ord('q'):
            if recording and writer:
                elapsed = time.time() - start_time
                writer.release()
                print(f"\n⏹  Stopped — {elapsed:.2f}s")
            break

    sock.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
