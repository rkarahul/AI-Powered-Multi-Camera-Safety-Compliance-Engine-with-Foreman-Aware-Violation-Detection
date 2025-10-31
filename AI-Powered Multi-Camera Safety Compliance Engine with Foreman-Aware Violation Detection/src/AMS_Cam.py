

#--------------------------------------------------------------------------------------------------------------------------------------

import os
import io
import gc
import json
import time
import base64
import traceback
from queue import Queue
from threading import Thread, Lock
import cv2
import numpy as np
import requests
from Offence_AMS_batch import score_img, paramreset
from new_cap import cap_check

# === Camera configurations ===
camera_configs = [
    {'path': 'videos/1.mp4'},
    {'path': 'videos/2.mp4'},
    {'path': 'videos/3.mp4'}
]

# camera_configs = [
#     {'ip': '192.168.1.21', 'username': 'admin', 'password': 'Add12345'},
#     {'ip': '192.168.1.31', 'username': 'admin', 'password': 'AddInn2025'},
#     {'ip': '192.168.1.32', 'username': 'admin', 'password': 'AddInn2025'}
# ]
camera_caps = []
latest_frames = {}
output_queue = Queue()
processing_complete = True
Flag = True
Flag_lock = Lock()   # protect shared variable access
has_offense = False   # <<-- add near top (global scope) before starting api_trigger_thread

# === API URLs ===
API_URL = "http://192.168.1.55:5052/api/CheckAlertSignal"
API_URL2 = "http://192.168.1.55:5052/api/send"


# === Initialize one camera or video ===
def init_camera(cfg):
    # url = f"rtsp://{cfg['username']}:{cfg['password']}@{cfg['ip']}/Streaming/Channels/1"
    url = cfg['path']
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_FPS, 5)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # if not cap.isOpened():
    #     print(f"❌ Failed to open: {cfg['ip']}")
    # else:
    #     print(f"✅ Connected: {cfg['ip']}")
    return cap


# === Thread: continuously capture frames ===
def camera_thread(index, cfg):
    global latest_frames
    cap = init_camera(cfg)

    while True:
        if not cap.isOpened():
            print(f"Reconnecting source {index}...")
            time.sleep(0.2)
            cap.release()
            cap = init_camera(cfg)
            continue

        ret, frame = cap.read()
        if not ret or frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop video
            continue

        frame = cv2.resize(frame, (640, 640))
        cv2.putText(frame, f"Cam {index+1}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        latest_frames[index] = frame


# === Start all camera/video threads ===
def start_cameras():
    for i, cfg in enumerate(camera_configs):
        t = Thread(target=camera_thread, args=(i, cfg), daemon=True)
        t.start()
        time.sleep(0.3)


# === Capture snapshot of all current frames ===
def capture_frames_snapshot():
    return [latest_frames.get(i, None) for i in range(len(camera_configs))]


# === Frame processing function ===
def frames_Process(output_queue):
    try:
        frame_list = capture_frames_snapshot()
        frame_list, finalresults = score_img(frame_list)
        # print("finalresults:", finalresults)
        output_queue.put((frame_list, finalresults))
    except Exception:
        traceback.print_exc()
        dummy = {(91, 64, 313, 384): {'count': 0, 'start_time': 0, 'offense': False}}
        output_queue.put((frame_list, dummy))

# === API trigger thread ===
def api_trigger_thread():
    global Flag
    while True:
        try:
            # Check signal
            response = requests.post(f"{API_URL}?has_offense={str(has_offense).lower()}", timeout=1)
            if response.ok:
                new_flag = response.json()
                with Flag_lock:
                    Flag = bool(new_flag)
                #print("new_flag result",new_flag)
        except Exception as e:
            print("⚠️ API not reachable:", e)
        time.sleep(0.5)


# === Combine frames in a grid ===
def combine_frames_grid(frames, grid_cols=3):
    frames = [f if f is not None else np.zeros((640, 640, 3), np.uint8) for f in frames]
    rows = [frames[i:i + grid_cols] for i in range(0, len(frames), grid_cols)]
    for row in rows:
        while len(row) < grid_cols:
            row.append(np.zeros_like(row[0]))
    full_rows = [np.hstack(row) for row in rows]
    return np.vstack(full_rows)


# === Main Execution ===
print("🚀 Starting video feeds...")
start_cameras()
counter = 0
# Start API trigger thread (parallel)
api_thread = Thread(target=api_trigger_thread, daemon=True)
api_thread.start()

output_queue = Queue()
processing_complete = True
temp = 0

# Detection ROI (optional)
# dectparam = {
#     ((149, 267), (364, 259), (484, 264), (638, 284), (639, 256),
#      (639, 218), (789, 214), (1013, 227), (1278, 281), (1280, 354),
#      (1377, 327), (1596, 276), (1764, 244), (1796, 393), (1658, 440),
#      (1458, 486), (1279, 498), (1279, 420), (1009, 410), (809, 386),
#      (639, 349), (639, 423), (371, 442), (129, 433)): 2
# }

# dectparam = {
#     ((1764,244), (1796,393),(1658,440),
#      (1458, 486),(1279,498),(1279,420),(1009,410),(809,386),(639,349),(639,423),(371,442),(129,433)): 2
# }
dectparam = {
    ((1815,244), (1870,360),(1796,393),(1658,440),
     (1458, 486),(1279,498),(1279,420),(1009,410),(809,386),(639,349),(639,423),(371,442),(60,428)): 2
}

# === Main Loop ===
while True:
    time.sleep(0.2)

    # Use thread-safe flag read
    with Flag_lock:
        local_flag = Flag

    if processing_complete and local_flag:
        processing_complete = False
        Thread(target=frames_Process, args=(output_queue,), daemon=True).start()

    if not output_queue.empty():
        frame_list, offenceresults = output_queue.get()
        # print("Offense results:", offenceresults)

        # Compute offense
        has_offense = any(d['offense'] for d in offenceresults.values())
        print("Offense results:", has_offense)
        # Send to API (parallel and non-blocking)
        Thread(target=lambda: requests.post(
            f"{API_URL2}?has_offense={str(has_offense).lower()}",
            timeout=1), daemon=True).start()

        processing_complete = True
        combined = combine_frames_grid(frame_list, grid_cols=3)
        _, mask_combine = cap_check(combined)

        for roi in dectparam:
            pts = np.array([list(p) for p in roi], np.int32).reshape((-1, 1, 2))
            cv2.polylines(combined, [pts], isClosed=False, color=(0, 255, 255), thickness=2)

        counter += 1
        #cv2.imwrite(f'CombinedImage{counter}.bmp', combined)
        cv2.imshow('Processed Feed', cv2.resize(combined, (1200, 440)))
        cv2.imshow('Mask Feed', cv2.resize(mask_combine, (1200, 440)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()


#-------------------------------------------------------------------------------------------------------------------------------------

