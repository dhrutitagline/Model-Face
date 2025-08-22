import torchvision.transforms.functional as F
import sys
sys.modules['torchvision.transforms.functional_tensor'] = F


import os
import cv2
import numpy as np
import mediapipe as mp
import gradio as gr
from gfpgan import GFPGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# ==============================
# 📂 Auto-download weights if missing
# ==============================
os.makedirs("weights", exist_ok=True)

# Download GFPGANv1.4.pth if it doesn't exist
if not os.path.exists("weights/GFPGANv1.4.pth"):
    os.system("curl -L -o weights/GFPGANv1.4.pth https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth")

# Download RealESRGAN_x2plus.pth if it doesn't exist
if not os.path.exists("weights/RealESRGAN_x2plus.pth"):
    os.system("curl -L -o weights/RealESRGAN_x2plus.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth")

# ==============================
# 📌 Load GFPGAN (CPU)
# ==============================
gfpganer = GFPGANer(
    model_path="weights/GFPGANv1.4.pth",
    upscale=1,
    arch="clean",
    channel_multiplier=2,
    device="cpu"
)

# ==============================
# 📌 Load RealESRGAN (CPU)
# ==============================
model = RRDBNet(
    num_in_ch=3, num_out_ch=3, num_feat=64,
    num_block=23, num_grow_ch=32, scale=2
)

upscaler = RealESRGANer(
    scale=2,
    model_path="weights/RealESRGAN_x2plus.pth",
    model=model,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=False,
    device="cpu"
)

# ==============================
# 📌 Face Beautification
# ==============================
def beautify(image):
    input_h, input_w = image.shape[:2]

    # Step 0: Resize very large inputs (for CPU speed)
    max_size = 1000
    if max(input_h, input_w) > max_size:
        scale = max_size / max(input_h, input_w)
        image = cv2.resize(image, (int(input_w * scale), int(input_h * scale)))

    # Step 1: Face restore with GFPGAN
    _, _, restored_img = gfpganer.enhance(
        image,
        has_aligned=False,
        only_center_face=False,
        paste_back=True
    )

    # Step 2: Super-resolution with RealESRGAN
    restored_img, _ = upscaler.enhance(restored_img, outscale=2)

    # Step 3: Skin smoothing + iris brightening (MediaPipe)
    mp_face = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        refine_landmarks=True
    )
    results = mp_face.process(cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB))

    if results.multi_face_landmarks:
        mask = np.zeros(restored_img.shape[:2], dtype=np.uint8)

        for face_landmarks in results.multi_face_landmarks:
            points = [(int(l.x * restored_img.shape[1]),
                       int(l.y * restored_img.shape[0]))
                      for l in face_landmarks.landmark]
            hull = cv2.convexHull(np.array(points))
            cv2.fillConvexPoly(mask, hull, 255)

        # Smooth skin
        smooth = cv2.bilateralFilter(restored_img, 9, 50, 50)
        restored_img = np.where(mask[..., None] == 255, smooth, restored_img)

        # Brighten iris
        for face_landmarks in results.multi_face_landmarks:
            for i in [468, 473]:
                x, y = int(face_landmarks.landmark[i].x * restored_img.shape[1]), \
                       int(face_landmarks.landmark[i].y * restored_img.shape[0])
                x1, y1, x2, y2 = x - 6, y - 6, x + 6, y + 6
                if x1 >= 0 and y1 >= 0 and x2 < restored_img.shape[1] and y2 < restored_img.shape[0]:
                    iris_region = restored_img[y1:y2, x1:x2]
                    iris_region = cv2.convertScaleAbs(iris_region, alpha=1.2, beta=15)
                    restored_img[y1:y2, x1:x2] = iris_region

    return restored_img

# ==============================
# 📌 Gradio Interface
# ==============================
demo = gr.Interface(
    fn=beautify,
    inputs=gr.Image(type="numpy", label="Upload a Face"),
    outputs=gr.Image(type="numpy", label="Enhanced Face"),
    title="✨ Model Face Enhancer (Male/Female)",
    description="Enhances faces using GFPGAN + RealESRGAN + MediaPipe for a model-like look."
)

if __name__ == "__main__":
    demo.launch(debug=True,share=True)
