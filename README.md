<div align="center">

# ✋ ISL Translator  
### Real-Time Indian Sign Language to Text & Speech



![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red?logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-green)
</div>

---

# ISL Translator

A lightweight system that converts Indian Sign Language (ISL) gestures into **text** and **speech** in real time.  
The project uses MediaPipe for hand-landmark detection, OpenCV for video processing, and a custom classifier for gesture recognition.

---

## Credits
This project is originally based on the work by **Pranav Prince (@princeo741)**.  
I have extended, modernized, and optimized multiple components to improve usability, accuracy, and overall functionality.

---

## My Additions & Enhancements
- Added full support for all **26 alphabet gestures** and **numeric signs**.
- Upgraded the UI with a modern overlay, dashed bounding boxes, smoother landmark markers, sidebar information, FPS display, and top-3 probability bars.
- Added convenient keyboard shortcuts:  
  - `s` → add space  
  - `c` → clear word  
  - `l` → switch language  
  - `q` → quit  
- Implemented a smoother **word-builder** with debouncing and auto-clear after inactivity.
- Improved the training pipeline with dynamic label handling, extended dataset support, and proper model/encoder saving.
- Added quick CSV verification for dataset balance (see `insight.py`).

---

## Features
- Real-time ISL gesture recognition  
- Text and speech output (English, Hindi, Kannada)  
- Record new gestures and extend the dataset  
- Train a custom model for improved accuracy  
- Clean UI with visual feedback and real-time probability bars  

---

## How to Use
```bash
git clone https://github.com/sidk44/ISL_Translator.git
cd ISL_Translator
pip install -r requirements.txt
python app.py
