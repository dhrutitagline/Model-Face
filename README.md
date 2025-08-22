# ✨ Model Face (Male/Female)

This project enhances user face images to give them a **model-like look** ✨.  
It uses a combination of:
- [GFPGAN](https://github.com/TencentARC/GFPGAN) → Face restoration  
- [RealESRGAN](https://github.com/xinntao/Real-ESRGAN) → Super resolution  
- [MediaPipe FaceMesh](https://github.com/google/mediapipe) → Beauty pass (skin smoothing + iris brightening)  

Built with **Gradio UI** for easy usage.  

---

## 🚀 Features
- Works for **both Male & Female faces**  
- Restores and enhances low-quality faces  
- Super-resolution upscaling (x2)  
- Automatic **skin smoothing + iris brightening**  
- Runs on **CPU** (slow) or **GPU** (fast)  

---

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/dhrutitagline/Model-Face.git
cd Model-Face
```

### 2️⃣ Create virtual environment
```bash
python3 -m venv env
source env/bin/activate   # Mac/Linux
env\Scripts\activate      # Windows
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
### 🖼️ Usage
#### Run on CPU
```bash
python face_enhancer_cpu.py
```

#### Run on GPU
```bash
python face_enhancer_gpu.py
```

#### Once started, Gradio will launch a web UI:

Running on http://127.0.0.1:7860

Upload an image → Get enhanced face.


### 📂 Project Structure
```bash
FaceModel-Enhancer/
│── Model_face_cpu.py   # CPU version
│── Model_face_gpu.py   # GPU version
│── requirements.txt       # Dependencies
│── README.md              # Project documentation
│── .gitignore             # Ignore unnecessary files
│── weights/               # Auto-downloaded model weights
```

