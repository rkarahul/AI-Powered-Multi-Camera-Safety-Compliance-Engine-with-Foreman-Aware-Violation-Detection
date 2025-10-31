# AI-Powered-Multi-Camera-Safety-Compliance-Engine-with-Foreman-Aware-Violation-Detection
**Real-time safety-violation monitoring for industrial zones using 3 overlapping cameras, YOLOv5 person detection, red-cap (foreman) identification, and smart overlap handling.**

---
## Overview  

| Component | Purpose |
|-----------|---------|
| **3 Fixed IP Cameras** (640×640 each) | Cover a 1920×640 panoramic view of the work-floor |
| **YOLOv5 custom model** (`best.pt`) | Detects `person` (class 0) |
| **Red-cap module** (`new_cap.py`) | Counts red pixels in the upper half of a person → **Foreman** |
| **Overlap-aware grouping** | Merges duplicate detections across cameras |
| **Cap-status propagation** | If a foreman is only partially visible in one camera, the cap detected in the adjacent camera is propagated |
| **ROI size filter** | Ignores tiny detections (≤70 px width **or** ≤120 px height) |
| **Polygon baseline** | Offence = bottom-center of a **non-foreman** outside the safe polygon |
| **Visual output** | Green = safe, Red = foreman, Pink = offender |

---

## How the Overlap Problem is Solved  

### 1. **Geometric propagation (no masking)**  
* A person on the **right edge** of Cam-0 is matched with the **left edge** of Cam-1 (and Cam-1 → Cam-2).  
* Matching criteria:  
  * >50 % vertical overlap  
  * Height difference < 20 %  
  * Horizontal gap < 50 px after global offset  
* If the **full view** (Cam-1) sees a red cap, the **partial view** (Cam-0) inherits `is_cap = True`.  

### 2. **Global-coordinate grouping**  
* Every detection is projected to a **1920 px** canvas (`x + cam_idx*640`).  
* Greedy IoU > 0.3 merges detections of the same person.  
* Foreman flag is **OR-ed** across the group.

---

## Camera Layout  

| Camera | Physical Position | Overlap Zone |
|--------|-------------------|--------------|
| **Cam-0** | Left side | Right 100 px overlap with Cam-1 |
| **Cam-1** | Center | Left 100 px with Cam-0, Right 100 px with Cam-2 |
| **Cam-2** | Right side | Left 100 px overlap with Cam-1 |

*All cameras stream **640×640** @ 15 fps (RTSP/HTTP).*

---

## Installation  

```bash
git clone https://github.com/<your-user>/Denso-AMS-MultiCam-Offense-Detection.git
cd Denso-AMS-MultiCam-Offense-Detection

# 1. Create venv
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. YOLOv5 submodule (or copy your trained weights)
git submodule update --init --recursive
# place your custom `best.pt` in ./yolov5/
