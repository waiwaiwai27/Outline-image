from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import cv2
import numpy as np
import ezdxf
from io import BytesIO
from PIL import Image

import torch
import requests
import time
import uuid
import json
from pathlib import Path

app = FastAPI(title="Outline Extractor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====================== Comyfi UI ======================

COMFYUI_URL = "http://127.0.0.1:8188"
WORKFLOW_PATH = Path("workflows/outline_test.json")

def queue_prompt(workflow: dict):
    payload = {
        "prompt": workflow,
        "client_id": str(uuid.uuid4())
    }
    res = requests.post(f"{COMFYUI_URL}/prompt", json=payload, timeout=30)
    res.raise_for_status()
    return res.json()

def get_history(prompt_id: str):
    res = requests.get(f"{COMFYUI_URL}/history/{prompt_id}", timeout=30)
    res.raise_for_status()
    return res.json()

def get_image(filename: str, subfolder: str = "", folder_type: str = "output"):
    params = {
        "filename": filename,
        "subfolder": subfolder,
        "type": folder_type
    }
    res = requests.get(f"{COMFYUI_URL}/view", params=params, timeout=60)
    res.raise_for_status()
    return res.content

def run_comfyui_outline(
    image_bytes: bytes,
    positive_prompt: str = None,
    negative_prompt: str = None,
    strength: float = 0.8,
    steps: int = 20,
    cfg: float = 8.0,
    seed: int = -1,
) -> bytes:
    """
    Send image to ComfyUI using your exported workflow and return the result PNG.
    """
    # Load the workflow template
    with open(WORKFLOW_PATH, "r", encoding="utf-8") as f:
        workflow = json.load(f)

    # 1. Upload image to ComfyUI
    files = {
        "image": ("input.png", image_bytes, "image/png")
    }
    upload_res = requests.post(f"{COMFYUI_URL}/upload/image", files=files, timeout=60)
    upload_res.raise_for_status()
    uploaded_name = upload_res.json()["name"]

    # 2. Inject uploaded image into LoadImage node (node "14")
    workflow["14"]["inputs"]["image"] = uploaded_name

    # 3. Override prompts if provided
    if positive_prompt and positive_prompt.strip():
        workflow["16"]["inputs"]["text"] = positive_prompt.strip()

    if negative_prompt and negative_prompt.strip():
        workflow["17"]["inputs"]["text"] = negative_prompt.strip()

    # 4. Override ControlNet strength
    workflow["23"]["inputs"]["strength"] = float(strength)

    # 5. Override sampler settings
    workflow["18"]["inputs"]["steps"] = int(steps)
    workflow["18"]["inputs"]["cfg"] = float(cfg)

    if seed >= 0:
        workflow["18"]["inputs"]["seed"] = int(seed)
    else:
        # random seed
        workflow["18"]["inputs"]["seed"] = int(time.time() * 1000) % (2**32)

    # 6. Queue the prompt
    queued = queue_prompt(workflow)
    prompt_id = queued["prompt_id"]
    print(f"ComfyUI prompt queued: {prompt_id}")

    # 7. Wait for completion
    while True:
        history = get_history(prompt_id)
        if prompt_id in history:
            break
        time.sleep(1.5)

    # 8. Get the output image from SaveImage node ("20")
    outputs = history[prompt_id]["outputs"]
    if "20" not in outputs or "images" not in outputs["20"]:
        raise RuntimeError("No image found in ComfyUI output")

    image_info = outputs["20"]["images"][0]
    image_data = get_image(
        filename=image_info["filename"],
        subfolder=image_info.get("subfolder", ""),
        folder_type=image_info.get("type", "output")
    )

    return image_data



# ====================== Global models ======================
_pidinet = None
_controlnet_pipe = None
_device = "cuda" if torch.cuda.is_available() else "cpu"

def get_pidinet():
    global _pidinet
    if _pidinet is None:
        from controlnet_aux import PidiNetDetector
        print("Loading PiDiNet...")
        _pidinet = PidiNetDetector.from_pretrained("lllyasviel/Annotators")
        _pidinet.to(_device)
        print("PiDiNet ready")
    return _pidinet

def get_controlnet_pipe():
    """Load SD 1.5 + ControlNet Lineart (lazy)"""
    global _controlnet_pipe
    if _controlnet_pipe is None:
        from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
        from diffusers.utils import load_image

        print("Loading ControlNet Lineart pipeline (this may take a while the first time)...")
        
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_lineart",
            torch_dtype=torch.float16 if _device == "cuda" else torch.float32
        )
        
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float16 if _device == "cuda" else torch.float32,
            safety_checker=None
        )
        
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        
        if _device == "cuda":
            pipe.enable_model_cpu_offload()  # saves VRAM
            # pipe.enable_xformers_memory_efficient_attention()  # uncomment if xformers installed
        else:
            pipe.to("cpu")
        
        _controlnet_pipe = pipe
        print("ControlNet pipeline ready")
    return _controlnet_pipe

# ====================== Helper functions ======================

def process_pidinet(img_bgr, safe=True, detect_resolution=512, dilate_iter=0):
    pidi = get_pidinet()
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    result = pidi(pil_img, detect_resolution=detect_resolution, safe=safe, output_type="np")
    edges = result[:, :, 0] if result.ndim == 3 else result
    edges = np.clip(edges, 0, 255).astype(np.uint8)
    _, edges = cv2.threshold(edges, 25, 255, cv2.THRESH_BINARY)
    if dilate_iter > 0:
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=dilate_iter)
    return edges

def process_lsd(gray, dilate_iter=1):
    fld = cv2.ximgproc.createFastLineDetector()
    lines = fld.detect(gray)
    edges = np.zeros_like(gray)
    if lines is not None:
        for line in lines:
            x0, y0, x1, y1 = map(int, line[0][:4])
            cv2.line(edges, (x0, y0), (x1, y1), 255, 1, cv2.LINE_AA)
    if dilate_iter > 0:
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=dilate_iter)
    return edges

def create_dxf_from_contours(binary_img):
    contours, _ = cv2.findContours(binary_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    scale = 0.1
    for cnt in contours:
        if len(cnt) < 2:
            continue
        points = [(float(p[0][0]) * scale, float(-p[0][1]) * scale) for p in cnt]
        if len(points) > 2:
            msp.add_lwpolyline(points, close=True)
    stream = BytesIO()
    doc.write(stream)
    return stream.getvalue()

def generate_lineart_controlnet(
    img_bgr: np.ndarray,
    prompt: str = None,
    negative_prompt: str = None,
    conditioning_scale: float = 0.95,
    guidance_scale: float = 7.5,
    steps: int = 25,
    seed: int = -1
) -> np.ndarray:
    pipe = get_controlnet_pipe()

    if prompt is None or prompt.strip() == "":
        prompt = (
            "clean technical architectural line drawing, pure white background, "
            "only thin black lines, precise building outlines, windows, fire escapes, "
            "no filled areas, no solid black regions, CAD style, sharp vector-like lines, "
            "professional drafting, pure white sky"
        )

    if negative_prompt is None or negative_prompt.strip() == "":
        negative_prompt = (
            "black background, solid black, filled black areas, large white shapes, "
            "white triangles, white blobs, shading, shadow, gradient, texture, color, "
            "photorealistic, 3d, realistic photo, noise, messy lines, broken lines, "
            "hatching, crosshatching, dark sky, inverted"
        )

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)

    # Resize to multiple of 8
    w, h = pil_image.size
    new_w = (w // 8) * 8
    new_h = (h // 8) * 8
    if new_w != w or new_h != h:
        pil_image = pil_image.resize((new_w, new_h), Image.LANCZOS)

    generator = None
    if seed >= 0:
        generator = torch.Generator(device=_device).manual_seed(seed)

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=pil_image,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=conditioning_scale,
        generator=generator,
        height=new_h,
        width=new_w
    ).images[0]

    result_np = np.array(result)
    gray = cv2.cvtColor(result_np, cv2.COLOR_RGB2GRAY)

    # --- Aggressive cleanup for CAD-ready result ---
    # 1. Force pure black & white
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # 2. Remove large solid black regions (sky etc.)
    # Invert temporarily so black becomes white for contour analysis
    inv = cv2.bitwise_not(binary)
    contours, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = binary.shape
    min_area = (h * w) * 0.08   # ignore regions larger than 8% of image

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            cv2.drawContours(binary, [cnt], -1, 255, -1)  # fill large black regions with white

    # 3. Optional: light morphological cleanup
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return binary

# ====================== API Endpoint ======================

@app.post("/extract-outline")
async def extract_outline(
    file: UploadFile = File(...),
    mode: str = Form("canny"),                    # canny | adaptive | lsd | pidinet | controlnet
    threshold1: int = Form(50),
    threshold2: int = Form(150),
    blur_ksize: int = Form(5),
    dilate_iter: int = Form(1),
    invert: bool = Form(False),
    block_size: int = Form(11),
    c_value: int = Form(2),
    safe: bool = Form(True),
    detect_resolution: int = Form(512),
    # ControlNet specific
    conditioning_scale: float = Form(0.9),
    guidance_scale: float = Form(7.5),
    steps: int = Form(20),
    seed: int = Form(-1),
    prompt: str = Form(""),
    negative_prompt: str = Form(""),
    output_format: str = Form("png"),
):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Could not decode image"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ksize = max(1, blur_ksize)
    if ksize % 2 == 0:
        ksize += 1
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)

    if mode == "controlnet":
        edges = generate_lineart_controlnet(
            img,
            prompt=prompt if prompt.strip() else None,
            negative_prompt=negative_prompt if negative_prompt.strip() else None,
            conditioning_scale=conditioning_scale,
            guidance_scale=guidance_scale,
            steps=steps,
            seed=seed
        )
    elif mode == "lsd":
        edges = process_lsd(blurred, dilate_iter)
    elif mode == "pidinet":
        edges = process_pidinet(img, safe=safe, detect_resolution=detect_resolution, dilate_iter=dilate_iter)
    elif mode == "adaptive":
        block = max(3, block_size)
        if block % 2 == 0:
            block += 1
        edges = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block, c_value
        )
        if dilate_iter > 0:
            kernel = np.ones((2, 2), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=dilate_iter)

    elif mode == "comfyui":
        try:
            result_bytes = run_comfyui_outline(
                image_bytes=contents,
                positive_prompt=prompt if prompt.strip() else None,
                negative_prompt=negative_prompt if negative_prompt.strip() else None,
                strength=conditioning_scale,   # reuse the existing slider
                steps=steps,
                cfg=guidance_scale,
                seed=seed
            )
            return Response(content=result_bytes, media_type="image/png")
        except Exception as e:
            return {"error": f"ComfyUI failed: {str(e)}"}

    else:  # canny
        edges = cv2.Canny(blurred, threshold1, threshold2)
        if dilate_iter > 0:
            kernel = np.ones((2, 2), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=dilate_iter)

    # Default polarity: black lines on white
    if not invert:
        edges = cv2.bitwise_not(edges)

    if output_format == "dxf":
        dxf_img = cv2.bitwise_not(edges) if not invert else edges
        dxf_bytes = create_dxf_from_contours(dxf_img)
        return Response(
            content=dxf_bytes,
            media_type="application/dxf",
            headers={"Content-Disposition": "attachment; filename=outline.dxf"}
        )
    else:
        success, encoded = cv2.imencode(".png", edges)
        if not success:
            return {"error": "Failed to encode image"}
        return Response(content=encoded.tobytes(), media_type="image/png")

@app.get("/")
def root():
    return {
        "status": "Outline Extractor API is running",
        "device": _device,
        "modes": ["canny", "adaptive", "lsd", "pidinet", "controlnet"]
    }
