# Outline Extractor

A local web application that converts architectural photos into clean line art / CAD-ready outlines.

It supports multiple extraction methods:

- **Canny Edge**
- **Adaptive Threshold**
- **LSD (Line Segment Detector)**
- **PiDiNet**
- **ComfyUI + ControlNet** (highest quality)

The application provides a side-by-side comparison slider, adjustable parameters, invert option, and export to PNG or DXF.

---

## Features

- Upload any photo (optimized for buildings / architecture)
- Multiple outline extraction modes
- Real-time parameter adjustment
- Before/After comparison slider
- Invert colors (black lines on white / white lines on black)
- Export as PNG or DXF
- Integration with ComfyUI for high-quality generative line art

---

## Screenshots

*(Add your screenshots here later)*

---

## Requirements

### Backend
- Python 3.10+ (3.13 is better if you also running ComfyUI)
- FastAPI + Uvicorn
- OpenCV
- (Optional) ComfyUI running locally for the best quality mode

### Frontend
- Node.js 18+
- React + Vite

### For ComfyUI mode
- ComfyUI installed and running on `http://127.0.0.1:8188`
- ControlNet models:
  - `control_v11p_sd15_lineart.pth`
- Checkpoint: any SD 1.5 model (e.g. `v1-5-pruned.ckpt`)

---

## Installation

### Clone the repository

bash
git clone https://github.com/yourusername/outline-extractor.git
cd outline-extractor

### Backend
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt

### Frontend
cd frontend
npm install

### (Optional) ComfyUI
Install ComfyUI desktop or install manually
Place the required ControlNet and checkpoint models. e.g. control_v11p_sd15_lineart.pth and control_v11p_sd15_softedge.pth which could be found in Hugging Face.  

### ComfyUI workflow
open the ComfyUI, either desktop or python main.py --listen 127.0.0.1 --port 8188
open the web page http://127.0.0.1:8188
create the workflow
Put your exported workflow API JSON into backend/workflows/outline_api.json

## Running the Application

### Terminal 1 – Backend
cd backend
venv\Scripts\activate
uvicorn main:app --reload --host 127.0.0.1 --port 8000

### Terminal 2 – Frontend
cd frontend
npm run dev
Open in browser: http://localhost:5173

## Usage
Upload an architectural photo
Select a mode:
Canny / Adaptive / LSD / PiDiNet → classical methods (fast)
ComfyUI → highest quality (requires ComfyUI running)

Adjust parameters
Click Extract
Use the slider to compare Original vs Outline
Download as PNG or DXF (excluding ControlNet and ComfyUI at the moment I create this.)

### Recommended Settings - Classical Modes

| Mode  | Key Parameters | Notes |
| ------------- | ------------- |------------- |
| Canny  | "Threshold1: 50–100, Threshold2: 150–200"  | Good general purpose  |
| Adaptive  | "Block Size: 11–21, C: 2–5" | Better with uneven lighting |
| LSD  | "Dilate: 0–2"  | Excellent for straight lines  |
| PiDiNet  | "Safe mode ON, Resolution 512–768" | Cleaner than Canny |


### Recommended Settings - ComfyUI Mode (Best Quality)

ControlNet Strength: 0.7 – 0.85
Steps: 25
CFG: 7.5 – 8.0
Sampler: euler_ancestral

Positive Prompt (CAD style):
clean technical architectural line drawing, pure white background, only thin black structural outlines, precise building edges, windows, floors, sharp lines, CAD style, no shading, no shadows, no filled areas, high contrast, professional drafting

Negative Prompt:
shading, shadow, gradient, gray, texture, soft edges, filled black areas, photorealistic, 3d, noise, blurry, hatching, crosshatching, messy lines

## Notes

DXF export works best with clean binary line images (Canny / PiDiNet / well-processed ComfyUI results).
ComfyUI mode is significantly slower on CPU. GPU (especially NVIDIA) is recommended for acceptable speed.
The application is designed for local use.
