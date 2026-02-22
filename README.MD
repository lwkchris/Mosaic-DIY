# 🖼️ Mosaic DIY

A high-performance Python application that transforms any image into a stunning mosaic composed of hundreds of tiny tile images. The app leverages **multiprocessing** to utilize every available CPU core for high-speed generation and features a modular, stable architecture.

[**Download Mosaic DIY.exe**](https://drive.google.com/file/d/1ame1GF9V3R4-0ogZXNDtBFmad9kgYe6Y/view?usp=sharing)

---

## 🚀 How to Use

1.  **Prepare your Tiles**: Create a folder containing many small images (JPG, PNG). A larger variety of colors results in higher accuracy.
2.  **Select Tiles Directory**: Click **"📁 Select Tiles Dir"** to load your tile library.
3.  **Select Target Image**: Click **"🖼️ Select Target"** to pick the photo you want to recreate.
4.  **Adjust Settings**: Set your **Resolution Ratio** (density) and **Overlay Transparency**.
5.  **Generate**: Click **"✨ Generate"**. The real-time progress bar tracks the status of all CPU worker processes.
6.  **Save**: Click **"💾 Save"** to export your final high-resolution masterpiece.

---

## 🧠 How it Works

The generator utilizes a sophisticated image processing pipeline designed for speed and memory efficiency:



1. **Grid Analysis**: The target image is divided into a grid. Each cell is analyzed for its average color profile.
2. **Parallel Color Matching**: Using the `multiprocessing` module, the workload is distributed across your CPU cores. Each core calculates the **Euclidean Distance** between the target segment and every available tile image.
3. **Shared Memory**: To avoid memory bottlenecks, tiles are stored in a `RawArray` shared across processes, ensuring lightning-fast access without duplicating data in RAM.
4. **Weighted Blending**: The assembly is blended with the original photo using `cv2.addWeighted` to ensure facial details and sharp edges are preserved.

---

## ⚙️ Settings Explained

| Setting | Meaning | Recommendation |
| :--- | :--- | :--- |
| **Resolution Ratio** | Multiplies the output size and tile density. | **5–10** for general use; **15+** for large prints. |
| **Overlay Transparency** | Blends the original photo over the tiles to reclaim lost detail. | **0.15 - 0.30** for a professional, "authentic" look. |

---

## 🛠️ Technical Structure

The project is refactored into a modular architecture for better maintainability and performance:

* **`logic.py`**: The engine. Handles OpenCV processing, Euclidean distance math, and multiprocessing pools.
* **`gui_components.py`**: The UI library. Contains styling constants (`BG_COLOR`, `ACCENT_BLUE`), custom widgets, and theme configurations.
* **`app.py`**: The controller. Manages the main window, thread synchronization, and user events.

### Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install opencv-python numpy pillow scipy

## ⏳ Performance & Processing Time
The time required to generate your mosaic is primarily determined by the computational "heavy lifting" involved in color matching. Below are the key factors that affect speed:

1. **Target Image & Resolution Ratio**
* The Resolution Ratio is the most significant factor.

* How it works: This setting determines how many "cells" the target image is divided into.

* Impact: Increasing the ratio increases the workload exponentially. For example, moving from a ratio of 10 to 20 doesn't just double the work—it quadruples the number of tiles that need to be calculated and placed.

2. **Tile Library Depth**
* Because the algorithm calculates the Euclidean Distance for every single cell against every available tile, the size of your "Tiles Dir" matters:

* Large Libraries (5,000+ images): Provide much higher color accuracy and less tile repetition but require the CPU to perform millions of additional comparisons.

* Small Libraries (<500 images): Result in very fast generation but may lead to a "blocky" look with many repeating images.

3. **Image Dimensions (Input vs. Output)**
* Target Image Size: Larger source files (like 4K photos) require more memory (RAM) during the initial grid analysis.

* Tile File Size: Using high-resolution, uncompressed files (like large PNGs or TIFFs) as your "material" tiles will slow down the initial loading phase. For maximum efficiency, use small, compressed JPGs for your tile collection.
