#!/usr/bin/env python3
"""
OAK-D RGB Sender — Runs on Raspberry Pi (inside AUV)
Streams 1080p RGB over TCP on port 5001.
"""

import depthai as dai
import socket
import struct
import cv2

pipeline = dai.Pipeline()

camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
camRgb.setFps(30)

xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutRgb.setStreamName("rgb")
camRgb.video.link(xoutRgb.input)

HOST = '0.0.0.0'
PORT = 5001

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((HOST, PORT))
server.listen(1)
print(f"Waiting for receiver on port {PORT}...")

conn, addr = server.accept()
print(f"Connected: {addr}")

with dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.SUPER) as device:
    queue = device.getOutputQueue("rgb", maxSize=4, blocking=False)
    print("Streaming 1080p...")
    while True:
        frame = queue.get().getCvFrame()
        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        data = jpeg.tobytes()
        size = struct.pack('>I', len(data))
        try:
            conn.sendall(size + data)
        except:
            print("Receiver disconnected.")
            break

conn.close()
server.close()
