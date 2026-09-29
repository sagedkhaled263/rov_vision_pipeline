import cv2
import depthai as dai
import numpy as np
import math
import time
import threading
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

# --- CONFIGURATION ---
UNDERWATER_MODE = True  # True = Flashed EEPROM, False = Factory EEPROM
PORT = 1211
JPEG_QUALITY = 70
# ---------------------

# --- WATER DEPTH CORRECTION ---
# Correction factor for measurements at different submersion depths.
# Calibrated at 1m depth, so that entry = 1.000
# Fill from pool tests: submerge to each depth, measure a KNOWN object
# at the SAME camera-to-object distance, record true/measured.
#
# correction = true_length / measured_length
WATER_DEPTH_CORRECTION = [
    (0.5, 1.000),
    (1.0, 1.000),   # calibration depth — perfect by definition
    (2.0, 0.950),
    (3.0, 0.900),
    (4.0, 0.830),
    (4.5, 0.800),   # MEASURED: 41.5cm reads 52.2, 133.5cm reads 166.1
    (5.0, 0.775),
]

correction_table_lock = threading.Lock()

def get_water_depth_correction(depth_m):
    """Interpolate correction factor based on ROV submersion depth."""
    with correction_table_lock:
        table = list(WATER_DEPTH_CORRECTION)
    depths  = [row[0] for row in table]
    factors = [row[1] for row in table]
    if depth_m <= depths[0]:
        return factors[0]
    if depth_m >= depths[-1]:
        return factors[-1]
    for i in range(len(depths) - 1):
        if depths[i] <= depth_m <= depths[i + 1]:
            t = (depth_m - depths[i]) / (depths[i + 1] - depths[i])
            return factors[i] + t * (factors[i + 1] - factors[i])
    return 1.0

# --- DISTANCE-AWARE EDGE CORRECTION ---
EDGE_ALPHA_TABLE = [
    (0.5, 0.12),
    (1.0, 0.10),
    (2.0, 0.08),
    (3.0, 0.06),
    (4.0, 0.05),
    (5.0, 0.04),
]

def get_edge_alpha(depth_m):
    """Interpolate edge correction alpha based on water depth."""
    depths = [row[0] for row in EDGE_ALPHA_TABLE]
    alphas = [row[1] for row in EDGE_ALPHA_TABLE]
    if depth_m <= depths[0]:
        return alphas[0]
    if depth_m >= depths[-1]:
        return alphas[-1]
    for i in range(len(depths) - 1):
        if depths[i] <= depth_m <= depths[i + 1]:
            t = (depth_m - depths[i]) / (depths[i + 1] - depths[i])
            return alphas[i] + t * (alphas[i + 1] - alphas[i])
    return 0.10

# Global state
system_state = "LIVE"
active_clients = 0
latest_jpeg = None
latest_depth_jpeg = None
latest_uncompressed_frame = None
frozen_uncompressed_frame = None
frozen_depth_color = None
latest_depth_frame = None
depth_buffer = []
fx = fy = cx = cy = 0.0
current_water_depth_m = 1.0
state_lock = threading.Lock()

def colorize_depth(depth_frame, min_range=200, max_range=5000):
    clipped = np.clip(depth_frame.astype(np.float32), min_range, max_range)
    normalized = ((clipped - min_range) / (max_range - min_range) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    colored[depth_frame == 0] = [0, 0, 0]
    return colored

HTML_PAGE = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Triton ROV Web Pilot</title>
    <style>
        body {{ background-color: #0a0a0a; color: #0f0; font-family: monospace; text-align: center; margin-top: 20px; }}
        #streams-row {{ display: flex; justify-content: center; gap: 10px; flex-wrap: wrap; }}
        .stream-wrapper {{ position: relative; display: inline-block; border: 2px solid #0f0; box-shadow: 0px 0px 20px rgba(0,255,0,0.3); cursor: crosshair; }}
        .stream-wrapper.depth {{ border-color: #f90; box-shadow: 0px 0px 20px rgba(255,153,0,0.3); }}
        .stream-label {{ position: absolute; top: 4px; left: 8px; font-size: 11px; color: #0f0; background: rgba(0,0,0,0.6); padding: 2px 6px; border-radius: 3px; z-index: 15; pointer-events: none; }}
        .stream-wrapper.depth .stream-label {{ color: #f90; }}
        .stream-img {{ display: block; width: 640px; height: 360px; }}
        .target-dot {{ position: absolute; width: 8px; height: 8px; background: red; border-radius: 50%; transform: translate(-50%, -50%); pointer-events: none; z-index: 10; box-shadow: 0 0 5px red; }}
        .target-line {{ position: absolute; pointer-events: none; z-index: 9; top: 0; left: 0; width: 100%; height: 100%; }}
        #panel {{ margin-top: 20px; padding: 15px; border: 1px solid #333; display: inline-block; background: #111; min-width: 550px; }}
        .readout {{ font-size: 28px; color: #0ff; margin: 10px 0; font-weight: bold; }}
        .btn {{ background: #0f0; color: #000; border: none; padding: 10px 20px; cursor: pointer; font-weight: bold; font-family: monospace; margin: 5px; }}
        .btn:hover {{ background: #fff; }}
        .btn-freeze {{ background: #ff0; }}
        .btn-mode {{ background: #0af; color: #fff; }}
        .btn-capture {{ background: #fff; color: #000; }}
        .btn-depth {{ background: #0af; color: #fff; }}
        #depth-panel {{ margin-top: 10px; border-top: 1px solid #333; padding-top: 10px; }}
        #depth-input {{ width: 60px; background: #222; color: #0ff; border: 1px solid #0ff; text-align: center; font-family: monospace; font-size: 16px; padding: 4px; }}
        #debug-info {{ margin-top: 8px; font-size: 11px; color: #555; }}
        #magnifier {{ position: absolute; border: 2px solid #0ff; border-radius: 50%; box-shadow: 0 0 15px #0ff; pointer-events: none; display: none; z-index: 20; background-color: #000; }}
        #correction-panel {{ margin-top: 15px; border-top: 1px solid #333; padding-top: 12px; text-align: left; }}
        #correction-panel h3 {{ color: #0ff; text-align: center; margin-bottom: 8px; cursor: pointer; user-select: none; }}
        #correction-table-body {{ display: none; }}
        #correction-table-body.open {{ display: block; }}
        #corr-table {{ width: 100%; border-collapse: collapse; margin-bottom: 8px; }}
        #corr-table th {{ color: #0ff; border-bottom: 1px solid #333; padding: 4px 8px; font-size: 12px; text-align: center; }}
        #corr-table td {{ padding: 3px 4px; text-align: center; }}
        #corr-table input {{ width: 70px; background: #1a1a1a; color: #0f0; border: 1px solid #333; text-align: center; font-family: monospace; font-size: 13px; padding: 3px; }}
        .btn-small {{ background: #333; color: #f55; border: 1px solid #555; padding: 2px 8px; cursor: pointer; font-family: monospace; font-size: 11px; }}
        .btn-add {{ background: #1b5e20; color: #0f0; border: 1px solid #4caf50; padding: 4px 12px; cursor: pointer; font-family: monospace; font-size: 12px; margin-right: 5px; }}
        .btn-save {{ background: #0d47a1; color: #0ff; border: 1px solid #2196f3; padding: 4px 12px; cursor: pointer; font-family: monospace; font-size: 12px; }}
        #corr-status {{ font-size: 11px; color: #555; margin-top: 6px; text-align: center; }}
        .add-row {{ display: flex; gap: 6px; align-items: center; justify-content: center; margin-top: 6px; }}
        .add-row input {{ width: 70px; background: #1a1a1a; color: #ff0; border: 1px solid #555; text-align: center; font-family: monospace; font-size: 13px; padding: 3px; }}
        .add-row label {{ color: #888; font-size: 11px; }}
    </style>
</head>
<body>
    <h2>TRITON ROV // TACTICAL WEB HUD</h2>
    <div id="streams-row">
        <div class="stream-wrapper" id="hud-wrapper">
            <span class="stream-label">RGB</span>
            <img id="video-stream" class="stream-img" src="/stream" draggable="false" />
            <canvas id="magnifier" width="120" height="120"></canvas>
            <svg class="target-line" id="line-rgb"></svg>
        </div>
        <div class="stream-wrapper depth" id="depth-wrapper">
            <span class="stream-label">DEPTH HEATMAP</span>
            <img id="depth-stream" class="stream-img" src="/depth_stream" draggable="false" />
            <svg class="target-line" id="line-depth"></svg>
        </div>
    </div>
    <br>
    <div id="panel">
        <div id="status" style="color:#aaa;">Live Stream. Press SPACE to Freeze.</div>
        <div id="result" class="readout">00.0 cm &plusmn; 0.0 cm</div>
        <button id="freeze-btn" class="btn btn-freeze" onclick="toggleFreeze()">FREEZE (SPACE)</button>
        <button id="mode-btn" class="btn btn-mode" onclick="toggleMode()">MODE: DIRECT 3D</button>
        <button class="btn" onclick="clearPoints()">CLEAR CLICKS</button>
        <button id="capture-btn" class="btn btn-capture" onclick="takeSnapshot()">&#128248; SAVE TO PC</button>
        <div id="depth-panel">
            <label style="color: #aaa;">ROV Water Depth: </label>
            <input id="depth-input" type="number" value="1.0" step="0.5" min="0" max="50" />
            <span style="color: #aaa;"> meters</span>
            <button class="btn btn-depth" onclick="setWaterDepth()">SET DEPTH</button>
            <span id="depth-status" style="color: #555; margin-left: 10px;"></span>
        </div>
        <div id="debug-info"></div>
        <div id="correction-panel">
            <h3 onclick="toggleCorrPanel()">&#9881; WATER DEPTH CORRECTION TABLE &#9660;</h3>
            <div id="correction-table-body">
                <table id="corr-table">
                    <thead><tr><th>Depth (m)</th><th>Factor</th><th></th></tr></thead>
                    <tbody id="corr-tbody"></tbody>
                </table>
                <div class="add-row">
                    <label>Depth:</label>
                    <input id="new-depth" type="number" step="0.1" min="0" placeholder="0.0" />
                    <label>Factor:</label>
                    <input id="new-factor" type="number" step="0.001" min="0" placeholder="1.000" />
                    <button class="btn-add" onclick="addCorrRow()">+ ADD</button>
                    <button class="btn-save" onclick="saveCorrTable()">&#10003; SAVE ALL</button>
                </div>
                <div id="corr-status"></div>
            </div>
        </div>
    </div>
    <script>
        const rgbWrapper = document.getElementById('hud-wrapper');
        const depthWrapper = document.getElementById('depth-wrapper');
        const stream = document.getElementById('video-stream');
        const resultDiv = document.getElementById('result');
        const statusDiv = document.getElementById('status');
        const freezeBtn = document.getElementById('freeze-btn');
        const modeBtn = document.getElementById('mode-btn');
        const debugDiv = document.getElementById('debug-info');
        const lineRgb = document.getElementById('line-rgb');
        const lineDepth = document.getElementById('line-depth');
        const mag = document.getElementById('magnifier');
        const ctx = mag.getContext('2d');
        const MAG_ZOOM = 3;
        const MAG_SIZE = 120;
        const CAM_W = 640;
        const CAM_H = 360;
        let clicks = [];
        let isFrozen = false;
        let measureMode = 'DIRECT';
        let scaleFactor = 1.0;

        function addDot(wrapper, x, y) {{
            const dot = document.createElement('div');
            dot.className = 'target-dot';
            dot.style.left = ((x / CAM_W) * 100) + '%';
            dot.style.top = ((y / CAM_H) * 100) + '%';
            wrapper.appendChild(dot);
        }}

        function drawLine(svgEl, p1, p2) {{
            svgEl.innerHTML = '';
            const x1 = (p1[0] / CAM_W) * 100, y1 = (p1[1] / CAM_H) * 100;
            const x2 = (p2[0] / CAM_W) * 100, y2 = (p2[1] / CAM_H) * 100;
            svgEl.innerHTML = '<line x1="' + x1 + '%" y1="' + y1 + '%" x2="' + x2 + '%" y2="' + y2 + '%" stroke="red" stroke-width="2" stroke-opacity="0.7" />';
        }}

        function clearOverlays() {{
            document.querySelectorAll('.target-dot').forEach(e => e.remove());
            lineRgb.innerHTML = '';
            lineDepth.innerHTML = '';
        }}

        let corrTableData = [];

        function toggleCorrPanel() {{
            const body = document.getElementById('correction-table-body');
            body.classList.toggle('open');
            if (body.classList.contains('open')) loadCorrTable();
        }}

        function loadCorrTable() {{
            fetch('/correction_table').then(r => r.json()).then(data => {{
                corrTableData = data.table;
                renderCorrTable();
            }});
        }}

        function renderCorrTable() {{
            const tbody = document.getElementById('corr-tbody');
            tbody.innerHTML = '';
            corrTableData.forEach((row, i) => {{
                const tr = document.createElement('tr');
                tr.innerHTML =
                    '<td><input type="number" step="0.1" min="0" value="' + row.depth + '" onchange="corrTableData[' + i + '].depth=parseFloat(this.value)" /></td>' +
                    '<td><input type="number" step="0.001" min="0" value="' + row.factor + '" onchange="corrTableData[' + i + '].factor=parseFloat(this.value)" /></td>' +
                    '<td><button class="btn-small" onclick="removeCorrRow(' + i + ')">&times;</button></td>';
                tbody.appendChild(tr);
            }});
        }}

        function addCorrRow() {{
            const d = parseFloat(document.getElementById('new-depth').value);
            const f = parseFloat(document.getElementById('new-factor').value);
            if (isNaN(d) || isNaN(f) || d < 0 || f <= 0) {{ alert('Enter valid depth (>=0) and factor (>0).'); return; }}
            corrTableData.push({{ depth: d, factor: f }});
            corrTableData.sort((a, b) => a.depth - b.depth);
            renderCorrTable();
            document.getElementById('new-depth').value = '';
            document.getElementById('new-factor').value = '';
            setCorrStatus('Row added (unsaved)', '#ff0');
        }}

        function removeCorrRow(index) {{
            corrTableData.splice(index, 1);
            renderCorrTable();
            setCorrStatus('Row removed (unsaved)', '#ff0');
        }}

        function saveCorrTable() {{
            for (let r of corrTableData) {{
                if (isNaN(r.depth) || isNaN(r.factor) || r.depth < 0 || r.factor <= 0) {{ alert('Fix invalid values before saving.'); return; }}
            }}
            corrTableData.sort((a, b) => a.depth - b.depth);
            fetch('/correction_table', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{ table: corrTableData }})
            }}).then(r => r.json()).then(data => {{
                if (data.status === 'ok') {{
                    corrTableData = data.table;
                    renderCorrTable();
                    setCorrStatus('Saved (' + data.count + ' entries). Current correction at ' + data.current_depth + 'm = ' + data.current_correction + 'x', '#0f0');
                }} else {{ setCorrStatus('Error: ' + (data.error || 'unknown'), '#f55'); }}
            }}).catch(() => setCorrStatus('Network error', '#f55'));
        }}

        function setCorrStatus(msg, color) {{
            const el = document.getElementById('corr-status');
            el.innerText = msg; el.style.color = color;
            setTimeout(() => {{ el.style.color = '#555'; }}, 5000);
        }}

        function setWaterDepth() {{
            let d = parseFloat(document.getElementById('depth-input').value);
            if (isNaN(d) || d < 0) {{ alert("Invalid depth value."); return; }}
            fetch('/set_depth', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{ water_depth_m: d }})
            }}).then(res => res.json()).then(data => {{
                document.getElementById('depth-status').innerText = 'Set to ' + data.depth + 'm (correction: ' + data.correction + 'x)';
                document.getElementById('depth-status').style.color = '#0ff';
                setTimeout(() => {{ document.getElementById('depth-status').style.color = '#555'; }}, 3000);
            }});
        }}

        function toggleMode() {{
            if (measureMode === 'DIRECT') {{
                measureMode = 'REF_1';
                modeBtn.innerText = 'MODE: SET REFERENCE';
                modeBtn.style.background = '#f90';
            }} else {{
                measureMode = 'DIRECT';
                scaleFactor = 1.0;
                modeBtn.innerText = 'MODE: DIRECT 3D';
                modeBtn.style.background = '#0af';
            }}
            clearPoints();
        }}

        function toggleFreeze() {{
            fetch('/toggle', {{ method: 'POST' }}).then(res => res.json()).then(data => {{
                isFrozen = (data.state === 'FROZEN');
                if (isFrozen) {{
                    freezeBtn.innerText = 'RESUME (SPACE)';
                    statusDiv.style.color = '#0ff';
                    statusDiv.innerText = measureMode === 'DIRECT'
                        ? 'System Frozen. Hover to magnify, click 2 points.'
                        : 'System Frozen. Click 2 points on KNOWN reference.';
                }} else {{
                    freezeBtn.innerText = 'FREEZE (SPACE)';
                    statusDiv.innerText = 'Live Stream. Press SPACE to Freeze.';
                    statusDiv.style.color = '#aaa';
                    mag.style.display = 'none';
                    debugDiv.innerText = '';
                    clearPoints();
                }}
            }});
        }}

        function takeSnapshot() {{ window.open('/snapshot', '_blank'); }}

        document.addEventListener('keydown', (e) => {{
            if (e.code === 'Space') {{ e.preventDefault(); toggleFreeze(); }}
        }});

        function clearPoints() {{
            clicks = [];
            clearOverlays();
            resultDiv.innerHTML = "00.0 cm &plusmn; 0.0 cm";
            debugDiv.innerText = '';
        }}

        rgbWrapper.addEventListener('mousemove', function(e) {{
            if (!isFrozen) return;
            const rect = stream.getBoundingClientRect();
            const mouseX = e.clientX - rect.left, mouseY = e.clientY - rect.top;
            if (mouseX < 0 || mouseY < 0 || mouseX > rect.width || mouseY > rect.height) {{ mag.style.display = 'none'; return; }}
            mag.style.display = 'block';
            mag.style.left = (mouseX + 15) + 'px';
            mag.style.top = (mouseY - MAG_SIZE - 15) + 'px';
            const camX = mouseX * (CAM_W / rect.width), camY = mouseY * (CAM_H / rect.height);
            ctx.clearRect(0, 0, MAG_SIZE, MAG_SIZE);
            ctx.drawImage(stream, camX - (MAG_SIZE / 2 / MAG_ZOOM), camY - (MAG_SIZE / 2 / MAG_ZOOM), MAG_SIZE / MAG_ZOOM, MAG_SIZE / MAG_ZOOM, 0, 0, MAG_SIZE, MAG_SIZE);
            ctx.strokeStyle = 'rgba(255,0,0,0.8)'; ctx.lineWidth = 1; ctx.beginPath();
            ctx.moveTo(MAG_SIZE / 2, 0); ctx.lineTo(MAG_SIZE / 2, MAG_SIZE);
            ctx.moveTo(0, MAG_SIZE / 2); ctx.lineTo(MAG_SIZE, MAG_SIZE / 2);
            ctx.stroke();
            ctx.fillStyle = 'red'; ctx.beginPath(); ctx.arc(MAG_SIZE/2, MAG_SIZE/2, 2, 0, 2*Math.PI); ctx.fill();
        }});

        rgbWrapper.addEventListener('mouseleave', () => {{ mag.style.display = 'none'; }});

        function handleStreamClick(e, imgEl) {{
            if (!isFrozen) {{ alert("Please freeze the feed (Spacebar) before measuring."); return; }}
            const rect = imgEl.getBoundingClientRect();
            const x = Math.round((e.clientX - rect.left) * (CAM_W / rect.width));
            const y = Math.round((e.clientY - rect.top) * (CAM_H / rect.height));
            if (clicks.length >= 2) {{ clicks = []; clearOverlays(); debugDiv.innerText = ''; }}
            clicks.push([x, y]);
            addDot(rgbWrapper, x, y); addDot(depthWrapper, x, y);
            if (clicks.length === 1) {{
                statusDiv.innerText = "Point 1 locked. Click Point 2.";
            }} else if (clicks.length === 2) {{
                drawLine(lineRgb, clicks[0], clicks[1]);
                drawLine(lineDepth, clicks[0], clicks[1]);
                statusDiv.innerText = "Calculating on ROV VPU & CPU...";
                resultDiv.innerText = "---";
                fetch('/measure', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ p1: clicks[0], p2: clicks[1], mode: measureMode }})
                }}).then(response => response.json()).then(data => {{
                    if (data.status === 'error') {{
                        statusDiv.innerText = "Error: " + data.result;
                        resultDiv.innerHTML = "ERR";
                        return;
                    }}
                    if (measureMode === 'DIRECT') {{
                        statusDiv.innerText = "Measurement Complete.";
                        resultDiv.innerHTML = data.result;
                        if (data.water_depth_m !== undefined) {{
                            debugDiv.innerText = 'Z_avg=' + data.avg_cam_dist_m + 'm | Water=' + data.water_depth_m + 'm | Correction=' + data.water_correction + 'x | Alpha=' + data.edge_alpha + ' | Raw=' + data.raw_dist_cm + 'cm';
                        }}
                    }} else if (measureMode === 'REF_1') {{
                        let actual = prompt(`The camera calculated ${{data.dist_cm}} cm.\\n\\nEnter the ACTUAL length of this reference object in cm:`);
                        if (actual && !isNaN(actual) && parseFloat(actual) > 0) {{
                            scaleFactor = parseFloat(actual) / data.dist_cm;
                            measureMode = 'REF_2';
                            statusDiv.innerText = `Scale Locked (${{scaleFactor.toFixed(2)}}x). Now click 2 points on the UNKNOWN target.`;
                            resultDiv.innerHTML = "Scale Set";
                            modeBtn.innerText = "MODE: MEASURING TARGET";
                            modeBtn.style.background = "#f0f";
                            clicks = []; clearOverlays();
                        }} else {{ alert("Invalid input. Reference scaling canceled."); clearPoints(); }}
                    }} else if (measureMode === 'REF_2') {{
                        let finalDist = data.dist_cm * scaleFactor;
                        let finalUnc = data.uncertainty_cm * scaleFactor;
                        statusDiv.innerText = "Target Measurement Complete (Corrected).";
                        resultDiv.innerHTML = `${{finalDist.toFixed(1)}} cm &plusmn; ${{finalUnc.toFixed(1)}} cm`;
                    }}
                }}).catch(() => {{ statusDiv.innerText = "Network Error."; }});
            }}
        }}

        document.getElementById('video-stream').addEventListener('mousedown', function(e) {{ handleStreamClick(e, this); }});
        document.getElementById('depth-stream').addEventListener('mousedown', function(e) {{ handleStreamClick(e, this); }});
    </script>
</body>
</html>
"""
HTML_PAGE_BYTES = HTML_PAGE.encode('utf-8')


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    pass


class ROVWebHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        global active_clients

        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_PAGE_BYTES)

        elif self.path == '/stream':
            with state_lock:
                active_clients += 1
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            last_sent_jpeg = None
            try:
                while True:
                    frame = latest_jpeg
                    if frame is not None and frame != last_sent_jpeg:
                        self.wfile.write(b'--frame\r\n')
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Content-Length', len(frame))
                        self.end_headers()
                        self.wfile.write(frame)
                        self.wfile.write(b'\r\n')
                        last_sent_jpeg = frame
                    time.sleep(0.05)
            except:
                pass
            finally:
                with state_lock:
                    active_clients -= 1

        elif self.path == '/depth_stream':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            last_sent = None
            try:
                while True:
                    frame = latest_depth_jpeg
                    if frame is not None and frame != last_sent:
                        self.wfile.write(b'--frame\r\n')
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Content-Length', len(frame))
                        self.end_headers()
                        self.wfile.write(frame)
                        self.wfile.write(b'\r\n')
                        last_sent = frame
                    time.sleep(0.05)
            except:
                pass

        elif self.path == '/snapshot':
            frame_to_capture = None
            with state_lock:
                if system_state == "FROZEN" and frozen_uncompressed_frame is not None:
                    frame_to_capture = frozen_uncompressed_frame.copy()
                elif system_state == "LIVE" and latest_uncompressed_frame is not None:
                    frame_to_capture = latest_uncompressed_frame.copy()
            if frame_to_capture is not None:
                _, buffer = cv2.imencode('.jpg', frame_to_capture, [cv2.IMWRITE_JPEG_QUALITY, 100])
                self.send_response(200)
                self.send_header('Content-Type', 'image/jpeg')
                timestamp = int(time.time())
                self.send_header('Content-Disposition', f'attachment; filename="triton_capture_{timestamp}.jpg"')
                self.end_headers()
                self.wfile.write(buffer.tobytes())
            else:
                self.send_response(400)
                self.end_headers()

        elif self.path == '/calibration':
            cal_info = {
                'water_depth_correction_table': WATER_DEPTH_CORRECTION,
                'edge_alpha_table': EDGE_ALPHA_TABLE,
                'current_water_depth_m': current_water_depth_m,
                'current_correction': round(get_water_depth_correction(current_water_depth_m), 4),
                'current_edge_alpha': round(get_edge_alpha(current_water_depth_m), 4),
                'intrinsics': {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy},
            }
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(cal_info).encode('utf-8'))

        elif self.path == '/correction_table':
            with correction_table_lock:
                table = [{'depth': d, 'factor': round(f, 4)} for d, f in WATER_DEPTH_CORRECTION]
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'table': table}).encode('utf-8'))

    def do_POST(self):
        global system_state, latest_depth_frame, depth_buffer
        global frozen_uncompressed_frame, current_water_depth_m
        global WATER_DEPTH_CORRECTION, frozen_depth_color, latest_depth_jpeg

        if self.path == '/toggle':
            with state_lock:
                if system_state == "LIVE":
                    system_state = "FROZEN"
                    if latest_uncompressed_frame is not None:
                        frozen_uncompressed_frame = latest_uncompressed_frame.copy()
                    if len(depth_buffer) > 0:
                        latest_depth_frame = np.median(depth_buffer, axis=0).astype(np.uint16)
                        frozen_depth_color = colorize_depth(latest_depth_frame)
                        _, dbuf = cv2.imencode('.jpg', frozen_depth_color, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                        latest_depth_jpeg = dbuf.tobytes()
                else:
                    system_state = "LIVE"
                    depth_buffer = []
                    frozen_uncompressed_frame = None
                    frozen_depth_color = None
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'state': system_state}).encode('utf-8'))

        elif self.path == '/set_depth':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            current_water_depth_m = float(data.get('water_depth_m', 1.0))
            correction = get_water_depth_correction(current_water_depth_m)
            alpha = get_edge_alpha(current_water_depth_m)
            print(f"[DEPTH] Water depth set to {current_water_depth_m}m (correction={correction:.4f}, alpha={alpha:.4f})")
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'depth': current_water_depth_m, 'correction': round(correction, 4), 'edge_alpha': round(alpha, 4)}).encode('utf-8'))

        elif self.path == '/correction_table':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            new_table_raw = data.get('table', [])
            new_table = []
            for row in new_table_raw:
                d = float(row.get('depth', -1))
                f = float(row.get('factor', -1))
                if d < 0 or f <= 0:
                    self.send_response(400)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'status': 'error', 'error': f'Invalid row: depth={d}, factor={f}'}).encode('utf-8'))
                    return
                new_table.append((d, round(f, 4)))
            new_table.sort(key=lambda x: x[0])
            if len(new_table) < 1:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'error', 'error': 'Table must have at least 1 entry'}).encode('utf-8'))
                return
            with correction_table_lock:
                WATER_DEPTH_CORRECTION = new_table
            correction_now = get_water_depth_correction(current_water_depth_m)
            print(f"[TABLE] Correction table updated: {len(new_table)} entries. Current at {current_water_depth_m}m = {correction_now:.4f}")
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                'status': 'ok', 'count': len(new_table),
                'table': [{'depth': d, 'factor': f} for d, f in new_table],
                'current_depth': current_water_depth_m,
                'current_correction': round(correction_now, 4),
            }).encode('utf-8'))

        elif self.path == '/measure':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            p1, p2 = data['p1'], data['p2']
            measure_mode = data.get('mode', 'DIRECT')
            response_data = {'status': 'error', 'result': 'Unknown Error'}

            if latest_depth_frame is not None:
                pad = 10
                x1_min, x1_max = max(0, p1[0]-pad), min(640, p1[0]+pad)
                y1_min, y1_max = max(0, p1[1]-pad), min(360, p1[1]+pad)
                x2_min, x2_max = max(0, p2[0]-pad), min(640, p2[0]+pad)
                y2_min, y2_max = max(0, p2[1]-pad), min(360, p2[1]+pad)

                roi1 = latest_depth_frame[y1_min:y1_max, x1_min:x1_max]
                roi2 = latest_depth_frame[y2_min:y2_max, x2_min:x2_max]
                valid1 = roi1[(roi1 > 0)]
                valid2 = roi2[(roi2 > 0)]

                if len(valid1) > 10 and len(valid2) > 10:
                    z1, z2 = np.percentile(valid1, 30), np.percentile(valid2, 30)
                    avg_z = (z1 + z2) / 2.0

                    x1_raw = (p1[0] - cx) * z1 / fx
                    y1_raw = (p1[1] - cy) * z1 / fy
                    x2_raw = (p2[0] - cx) * z2 / fx
                    y2_raw = (p2[1] - cy) * z2 / fy

                    alpha_used = 0.0
                    if measure_mode == 'DIRECT':
                        alpha_used = get_edge_alpha(current_water_depth_m)
                        edge_ratio_x1 = abs(p1[0] - cx) / cx
                        edge_ratio_x2 = abs(p2[0] - cx) / cx
                        x1_3d = x1_raw * (1.0 - (alpha_used * (edge_ratio_x1 ** 2)))
                        x2_3d = x2_raw * (1.0 - (alpha_used * (edge_ratio_x2 ** 2)))
                    else:
                        x1_3d = x1_raw
                        x2_3d = x2_raw

                    y1_3d = y1_raw
                    y2_3d = y2_raw

                    dist_mm_raw = math.sqrt((x2_3d - x1_3d)**2 + (y2_3d - y1_3d)**2 + (z2 - z1)**2)
                    std1, std2 = np.std(valid1), np.std(valid2)
                    uncertainty_mm_raw = math.sqrt(std1**2 + std2**2)

                    water_correction = get_water_depth_correction(current_water_depth_m)
                    dist_mm = dist_mm_raw * water_correction
                    uncertainty_mm = uncertainty_mm_raw * water_correction

                    dist_cm = round(dist_mm / 10.0, 2)
                    unc_cm = round(uncertainty_mm / 10.0, 2)
                    raw_dist_cm = round(dist_mm_raw / 10.0, 2)

                    response_data = {
                        'status': 'success',
                        'result': f"{dist_cm:.1f} cm &plusmn; {unc_cm:.1f} cm",
                        'dist_cm': dist_cm,
                        'uncertainty_cm': unc_cm,
                        'raw_dist_cm': raw_dist_cm,
                        'water_correction': round(water_correction, 4),
                        'water_depth_m': current_water_depth_m,
                        'edge_alpha': round(alpha_used, 4),
                        'avg_cam_dist_m': round(avg_z / 1000.0, 2),
                    }
                else:
                    response_data['result'] = "ERR: Bad Depth at Clicks"
            else:
                response_data['result'] = "ERR: Buffer Empty"

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode('utf-8'))


def main():
    global latest_jpeg, latest_uncompressed_frame, depth_buffer
    global system_state, fx, fy, cx, cy, active_clients, latest_depth_jpeg

    print("\n--- TRITON ROV BOOT SEQUENCE ---")

    with dai.Device() as temp_device:
        if UNDERWATER_MODE:
            print("🌊 UNDERWATER MODE: Fetching Flashed Custom Calibration from EEPROM...")
            calibData = temp_device.readCalibration()
        else:
            print("🏢 DESK MODE: Fetching Original Factory Calibration from EEPROM...")
            calibData = temp_device.readFactoryCalibration()

    intrinsics = calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, 640, 360)
    fx, fy = intrinsics[0][0], intrinsics[1][1]
    cx, cy = intrinsics[0][2], intrinsics[1][2]

    pipeline = dai.Pipeline()
    pipeline.setCalibrationData(calibData)

    camRgb = pipeline.create(dai.node.ColorCamera)
    camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    camRgb.setPreviewSize(640, 360)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    camLeft = pipeline.create(dai.node.MonoCamera)
    camLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    camLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)

    camRight = pipeline.create(dai.node.MonoCamera)
    camRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    camRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)

    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(640, 360)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)
    stereo.initialConfig.setConfidenceThreshold(200)
    stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)

    config = stereo.initialConfig.get()
    config.postProcessing.spatialFilter.enable = True
    config.postProcessing.spatialFilter.holeFillingRadius = 2
    config.postProcessing.spatialFilter.numIterations = 1
    config.postProcessing.temporalFilter.enable = True
    config.postProcessing.temporalFilter.alpha = 0.4
    config.postProcessing.temporalFilter.delta = 20
    config.postProcessing.thresholdFilter.minRange = 200
    config.postProcessing.thresholdFilter.maxRange = 5000
    stereo.initialConfig.set(config)

    camLeft.out.link(stereo.left)
    camRight.out.link(stereo.right)

    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutRgb.setStreamName("rgb")
    camRgb.preview.link(xoutRgb.input)

    xoutDepth = pipeline.create(dai.node.XLinkOut)
    xoutDepth.setStreamName("depth")
    stereo.depth.link(xoutDepth.input)

    server = ThreadedHTTPServer(('0.0.0.0', PORT), ROVWebHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    print(f"✅ SYSTEM ONLINE (HEADLESS + WATER DEPTH CORRECTION)")
    print(f"🌐 Connect any laptop browser to: http://<ROV_IP>:{PORT}")
    print(f"📊 Calibration info at: http://<ROV_IP>:{PORT}/calibration\n")

    with dai.Device(pipeline) as device:
        qRgb = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
        qDepth = device.getOutputQueue(name="depth", maxSize=1, blocking=False)

        while True:
            inRgb = qRgb.get()
            inDepth = qDepth.tryGet()

            if system_state == "LIVE":
                if inDepth is not None:
                    depth_buffer.append(inDepth.getFrame())
                    if len(depth_buffer) > 5:
                        depth_buffer.pop(0)

                with state_lock:
                    client_count = active_clients

                if client_count > 0:
                    latest_uncompressed_frame = inRgb.getCvFrame()
                    stream_frame = latest_uncompressed_frame.copy()
                    cv2.line(stream_frame, (310, 180), (330, 180), (0, 255, 0), 1)
                    cv2.line(stream_frame, (320, 170), (320, 190), (0, 255, 0), 1)
                    _, buffer = cv2.imencode('.jpg', stream_frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                    latest_jpeg = buffer.tobytes()

                    if len(depth_buffer) > 0:
                        depth_color = colorize_depth(depth_buffer[-1])
                        _, dbuf = cv2.imencode('.jpg', depth_color, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                        latest_depth_jpeg = dbuf.tobytes()


if __name__ == "__main__":
    main()
