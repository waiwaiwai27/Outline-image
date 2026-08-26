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
- Python 3.10+
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

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/outline-extractor.git
cd outline-extractor
