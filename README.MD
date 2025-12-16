# 🖼️ Mosaic DIY

A high-performance Python application that transforms any image into a stunning mosaic composed of hundreds of tiny tile images. The app uses **multiprocessing** for high-speed generation and features a modern, stable Tkinter GUI.

---

## 🚀 How to Use

1.  **Prepare your Tiles**: Create a folder containing many small images (JPG, JPEG, or PNG). A larger variety of colors in your tile collection will result in a more accurate mosaic.
2.  **Select Tiles Directory**: Click **"📁 Select Tiles Dir"** and choose your folder of small images.
3.  **Select Target Image**: Click **"🖼️ Select Target"** and pick the main photo you want to recreate.
4.  **Adjust Settings**: Set your Resolution and Overlay visibility (see [Settings](#-settings) below).
5.  **Generate**: Click **"✨ Generate"**. The real-time progress bar will show you the status of each CPU process.
6.  **Save**: Once the preview appears, click **"💾 Save"** to export your final high-resolution masterpiece.

---

## ⚙️ Settings Explained

| Setting | Meaning | Recommendation |
| :--- | :--- | :--- |
| **Resolution Ratio** | Controls the final scale and tile density. A higher ratio means more tiles and a much larger final image file. | **5–10** for general use; **15+** for large-scale prints. |
| **Original Image Overlay** | Blends the original "target" photo over the tiles. This acts as a filter to help define details that tiles alone might miss. | **0.15 - 0.30 (15-30%)** provides a professional look without hiding the tiles. |

---

## 🧠 How it Works

The generator follows a sophisticated image processing pipeline:
1. **Grid Analysis**: The target image is divided into a grid based on the selected resolution.
2. **Color Matching**: For every cell in the grid, the app calculates the **Euclidean Distance** between the target segment and every available tile image.
3. **Multiprocessing**: The task is split across all available CPU cores, allowing thousands of comparisons to happen simultaneously.
4. **Weighted Blending**: The final tile assembly is blended with the original photo using `cv2.addWeighted` for the final aesthetic touch.



---

## 🛠️ Technical Setup

### Prerequisites
* Python 3.8 or higher
* **Libraries**: OpenCV, NumPy, Pillow, SciPy, Tqdm

### Installation
Install the required dependencies via pip:
```bash
pip install opencv-python numpy pillow scipy tqdm