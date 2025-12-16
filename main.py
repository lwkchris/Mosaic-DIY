import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import threading
import os
import re
import numpy as np
import cv2
import multiprocessing as mp
from multiprocessing.sharedctypes import RawArray
from scipy.spatial.distance import euclidean
from tqdm import tqdm
import ctypes

# --- Constants for GUI ---
PREVIEW_SIZE_INPUT = 250
PREVIEW_SIZE_OUTPUT = 400


# --- Core Mosaic Generation Functions ---

def resize(im, tile_row, tile_col):
    shape_row, shape_col = im.shape[0], im.shape[1]
    shrink_ratio = min(shape_row / tile_row, shape_col / tile_col)
    resized = cv2.resize(im, (int(shape_col / shrink_ratio) + 1, int(shape_row / shrink_ratio) + 1),
                         interpolation=cv2.INTER_CUBIC)
    return resized[:tile_row, :tile_col, :]


def img_distance(im1, im2):
    return euclidean(im1.flatten(), im2.flatten())


def load_all_images(img_dir, tile_row, tile_col):
    filenames = os.listdir(img_dir)
    result = []
    for filename in tqdm(filenames):
        if not re.search(r"\.(jpg|jpeg|png)$", filename, re.I):
            continue
        filepath = os.path.join(img_dir, filename)
        try:
            im = cv2.imread(filepath)
            if im is None: continue
            result.append(np.array(resize(im, tile_row, tile_col)))
        except:
            continue
    return np.array(result, dtype=np.uint8)


def find_closest_image(q, shared_tile_images, tile_images_shape, shared_result, img_shape, tile_row, tile_col,
                       shared_counter):
    tile_images = np.frombuffer(shared_tile_images, dtype=np.uint8).reshape(tile_images_shape)
    while True:
        try:
            task = q.get(timeout=0.1)
            row, col, im_roi = task
            min_dist = float("inf")
            min_img = None
            for im in tile_images:
                dist = img_distance(im_roi, im)
                if dist < min_dist:
                    min_dist, min_img = dist, im
            im_res = np.frombuffer(shared_result, dtype=np.uint8).reshape(img_shape)
            if min_img is not None:
                im_res[row:row + tile_row, col:col + tile_col, :] = min_img
            q.task_done()
            with shared_counter.get_lock():
                shared_counter.value += 1
        except:
            break


def get_tile_row_col(shape):
    return [120, 90] if shape[0] >= shape[1] else [90, 120]


def generate_mosaic_core(infile, img_dir, ratio, num_processes, shared_counter):
    img = cv2.imread(infile)
    if img is None: raise FileNotFoundError(f"Could not read input file: {infile}")
    tile_row, tile_col = get_tile_row_col(img.shape)
    img_shape = [int(img.shape[0] / tile_row) * tile_row * ratio, int(img.shape[1] / tile_col) * tile_col * ratio, 3]
    img_resized = cv2.resize(img, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_CUBIC)

    tile_images = load_all_images(img_dir, tile_row, tile_col)
    shared_tile_images = RawArray(ctypes.c_ubyte, len(tile_images.flatten()))
    np.copyto(np.frombuffer(shared_tile_images, dtype=np.uint8).reshape(tile_images.shape), tile_images)

    im_res = np.zeros(img_shape, np.uint8)
    shared_result = RawArray(ctypes.c_ubyte, len(im_res.flatten()))

    q = mp.JoinableQueue()
    processes = [mp.Process(target=find_closest_image, args=(
    q, shared_tile_images, tile_images.shape, shared_result, img_shape, tile_row, tile_col, shared_counter),
                            daemon=True) for _ in range(num_processes)]
    for p in processes: p.start()

    total_tiles = 0
    for row in range(0, img_shape[0], tile_row):
        for col in range(0, img_shape[1], tile_col):
            q.put([row, col, img_resized[row:row + tile_row, col:col + tile_col, :]])
            total_tiles += 1
    yield total_tiles
    q.join()
    for p in processes: p.terminate()

    # Final yield is the raw tile mosaic
    yield np.frombuffer(shared_result, dtype=np.uint8).reshape(img_shape).copy()


# --- GUI Application Class ---

class MosaicGeneratorApp:
    def __init__(self, master):
        self.master = master
        master.title("Mosaic DIY")

        self.output_array = None
        self.generation_running = False
        self.total_tasks = 0
        self.shared_counter = mp.Value('i', 0)
        self.img_dir, self.input_file = tk.StringVar(), tk.StringVar()
        self.ratio = tk.IntVar(value=10)
        self.overlay_alpha = tk.DoubleVar(value=0.2)  # 20% default visibility

        main_frame = tk.Frame(master)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)

        # --- Settings ---
        input_settings_frame = tk.LabelFrame(main_frame, text="Settings", padx=10, pady=10)
        input_settings_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        tk.Button(input_settings_frame, text="📁 Select Tiles Dir", command=self.browse_dir).grid(row=0, column=0,
                                                                                                 sticky="ew", pady=2)
        tk.Entry(input_settings_frame, textvariable=self.img_dir, state='readonly').grid(row=0, column=1, sticky="ew",
                                                                                         padx=5)

        tk.Button(input_settings_frame, text="🖼️ Select Target", command=self.browse_file).grid(row=1, column=0,
                                                                                                sticky="ew", pady=2)
        tk.Entry(input_settings_frame, textvariable=self.input_file, state='readonly').grid(row=1, column=1,
                                                                                            sticky="ew", padx=5)
        self.input_file.trace_add('write', self.show_input_preview)

        tk.Scale(input_settings_frame, from_=1, to=20, orient=tk.HORIZONTAL, variable=self.ratio,
                 label="Resolution Ratio").grid(row=2, column=0, columnspan=2, sticky="ew")

        # New Visibility Control
        tk.Scale(input_settings_frame, from_=0.0, to=1.0, resolution=0.05, orient=tk.HORIZONTAL,
                 variable=self.overlay_alpha, label="Original Image Overlay (Visibility)").grid(row=3, column=0,
                                                                                                columnspan=2,
                                                                                                sticky="ew")

        self.generate_button = tk.Button(input_settings_frame, text="✨ Generate", command=self.start_generation_thread,
                                         bg="#4CAF50", fg="white", font=('bold'))
        self.generate_button.grid(row=4, column=0, columnspan=2, sticky="ew", pady=10)

        self.progress_bar = ttk.Progressbar(input_settings_frame, orient='horizontal', mode='determinate')
        self.progress_bar.grid(row=5, column=0, columnspan=2, sticky="ew")
        self.status_text = tk.StringVar(value="Ready.")
        tk.Label(input_settings_frame, textvariable=self.status_text, wraplength=250).grid(row=6, column=0,
                                                                                           columnspan=2, sticky="w")

        # --- Input Preview (LOCKED SIZE) ---
        self.input_preview_frame = tk.LabelFrame(main_frame, text="Input Preview", width=PREVIEW_SIZE_INPUT + 20,
                                                 height=PREVIEW_SIZE_INPUT + 20)
        self.input_preview_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.input_preview_frame.grid_propagate(False)
        self.input_label = tk.Label(self.input_preview_frame, text="No Image")
        self.input_label.place(relx=0.5, rely=0.5, anchor='center')

        # --- Output Preview (LOCKED SIZE) ---
        self.output_preview_frame = tk.LabelFrame(main_frame, text="Mosaic Preview", width=PREVIEW_SIZE_OUTPUT + 20,
                                                  height=PREVIEW_SIZE_OUTPUT + 60)
        self.output_preview_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        self.output_preview_frame.grid_propagate(False)
        self.output_label = tk.Label(self.output_preview_frame, text="Result will appear here.")
        self.output_label.place(relx=0.5, rely=0.45, anchor='center')

        ctrl_frame = tk.Frame(self.output_preview_frame)
        ctrl_frame.place(relx=0.5, rely=0.9, anchor='center', relwidth=0.9)
        self.save_button = tk.Button(ctrl_frame, text="💾 Save", command=self.save_output_mosaic, state=tk.DISABLED,
                                     bg="#007bff", fg="white")
        self.save_button.pack(side="left", expand=True, fill="x", padx=2)
        self.discard_button = tk.Button(ctrl_frame, text="❌ Clear", command=self.clear_output_preview,
                                        state=tk.DISABLED)
        self.discard_button.pack(side="right", expand=True, fill="x", padx=2)

    def browse_dir(self):
        d = filedialog.askdirectory()
        if d: self.img_dir.set(d)

    def browse_file(self):
        f = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if f: self.input_file.set(f)

    def show_input_preview(self, *args):
        fp = self.input_file.get()
        if not fp: return
        try:
            img = Image.open(fp)
            img.thumbnail((PREVIEW_SIZE_INPUT, PREVIEW_SIZE_INPUT))
            self.input_photo = ImageTk.PhotoImage(img)
            self.input_label.config(image=self.input_photo, text="")
        except:
            pass

    def start_generation_thread(self):
        if not self.img_dir.get() or not self.input_file.get():
            return messagebox.showerror("Error", "Select both folder and image.")
        self.shared_counter.value, self.total_tasks, self.generation_running = 0, 0, True
        self.progress_bar['value'] = 0
        self.generate_button.config(state=tk.DISABLED)
        threading.Thread(target=self.run_generation,
                         args=(self.input_file.get(), self.img_dir.get(), self.ratio.get())).start()
        self.update_progress()

    def update_progress(self):
        if not self.generation_running: return
        if self.total_tasks > 0:
            val = self.shared_counter.value
            self.progress_bar['value'] = (val / self.total_tasks) * 100
            self.status_text.set(f"Processing: {val}/{self.total_tasks}")
        self.master.after(200, self.update_progress)

    def run_generation(self, infile, img_dir, ratio):
        try:
            gen = generate_mosaic_core(infile, img_dir, ratio, mp.cpu_count(), self.shared_counter)
            self.total_tasks = next(gen)
            raw_mosaic = next(gen)

            # --- BLENDING LOGIC ---
            alpha = self.overlay_alpha.get()
            if alpha > 0:
                original = cv2.imread(infile)
                # Resize original to match mosaic resolution exactly
                original_resized = cv2.resize(original, (raw_mosaic.shape[1], raw_mosaic.shape[0]))
                # Blend: Result = Mosaic*(1-alpha) + Original*alpha
                self.output_array = cv2.addWeighted(raw_mosaic, 1 - alpha, original_resized, alpha, 0)
            else:
                self.output_array = raw_mosaic

            self.master.after(0, self.on_success)
        except Exception as e:
            self.master.after(0, lambda: messagebox.showerror("Error", str(e)))

    def on_success(self):
        self.generation_running = False
        self.status_text.set("Done!")
        self.generate_button.config(state=tk.NORMAL)
        self.save_button.config(state=tk.NORMAL)
        self.discard_button.config(state=tk.NORMAL)
        img = Image.fromarray(cv2.cvtColor(self.output_array, cv2.COLOR_BGR2RGB))
        img.thumbnail((PREVIEW_SIZE_OUTPUT, PREVIEW_SIZE_OUTPUT))
        self.output_photo = ImageTk.PhotoImage(img)
        self.output_label.config(image=self.output_photo, text="")

    def save_output_mosaic(self):
        fp = filedialog.asksaveasfilename(defaultextension=".jpg")
        if fp: cv2.imwrite(fp, self.output_array)

    def clear_output_preview(self):
        self.output_label.config(image='', text="Result will appear here.")
        self.save_button.config(state=tk.DISABLED)
        self.discard_button.config(state=tk.DISABLED)
        self.output_array = None


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    mp.set_start_method('spawn', force=True)

    root = tk.Tk()
    app = MosaicGeneratorApp(root)
    root.mainloop()