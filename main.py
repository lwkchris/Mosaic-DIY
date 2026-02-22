
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import threading
from core_logic import *
from gui_components import *


class MosaicGeneratorApp:
    def __init__(self, master):
        self.master = master
        master.title("Mosaic Generator Pro")
        master.configure(bg=BG_COLOR)

        screen_w = master.winfo_screenwidth()
        screen_h = master.winfo_screenheight()
        self.window_w = int(screen_w * 0.55)
        self.window_h = int(screen_h * 0.88)
        master.geometry(f"{self.window_w}x{self.window_h}")
        master.resizable(False, False)

        top_h = int(self.window_h * 0.42)
        bottom_h = int(self.window_h * 0.50)
        settings_w = int(self.window_w * 0.45)

        self.base_mosaic = None
        self.original_resized = None
        self.display_array = None
        self.generation_running = False
        self.total_tasks = 0
        self.shared_counter = mp.Value('i', 0)
        self.img_dir, self.input_file = tk.StringVar(), tk.StringVar()
        self.ratio = tk.IntVar(value=3)
        self.overlay_alpha = tk.DoubleVar(value=0.5)

        self.overlay_alpha.trace_add("write", lambda *args: self.apply_overlay_filter())

        top_frame = tk.Frame(master, bg=BG_COLOR)
        top_frame.pack(fill='x', padx=10, pady=10)

        self.settings_frame = tk.LabelFrame(top_frame, text=" Settings ", bg=BG_COLOR, relief='groove', borderwidth=2,
                                            width=settings_w, height=top_h)
        self.settings_frame.pack_propagate(False)
        self.settings_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))

        def create_input(parent, label, var, cmd):
            row = tk.Frame(parent, bg=BG_COLOR)
            row.pack(fill='x', pady=5, padx=10)
            tk.Button(row, text=f"{label}", fg="white", bg=ACCENT_GREEN, command=cmd, width=15).pack(side='left')
            tk.Entry(row, textvariable=var, bg="white").pack(side='left', fill='x', expand=True, padx=(5, 0))

        create_input(self.settings_frame, "Select Tiles Dir", self.img_dir, self.browse_dir)
        create_input(self.settings_frame, "Select Target", self.input_file, self.browse_file)

        tk.Scale(
            self.settings_frame,
            from_=1, to=20,
            resolution=1,
            orient='horizontal',
            variable=self.ratio,
            bg=BG_COLOR,
            highlightthickness=0,
            label="Resolution Ratio",
            showvalue=True,  # This puts the number on the handle
        ).pack(fill='x', padx=20)

        tk.Scale(
            self.settings_frame,
            from_=0, to=1,
            resolution=0.01,
            orient='horizontal',
            variable=self.overlay_alpha,
            bg=BG_COLOR,
            highlightthickness=0,
            label="Overlay Transparency",  # This puts text above the slider
            showvalue=True,  # This puts the number on the handle
            font=('Segoe UI', 9)
        ).pack(fill='x', padx=20, pady=(5, 10))

        self.gen_btn = tk.Button(self.settings_frame, text="✨ Generate", bg=ACCENT_GREEN, fg="white",
                                 font=('Segoe UI', 10, 'bold'), relief='raised', command=self.start_generation_thread)
        self.gen_btn.pack(fill='x', padx=15, pady=10)

        self.prog = ttk.Progressbar(self.settings_frame, orient='horizontal', mode='determinate')
        self.prog.pack(fill='x', padx=15)
        self.status_lbl = tk.Label(self.settings_frame, text="Ready", bg=BG_COLOR, anchor='w')
        self.status_lbl.pack(fill='x', padx=15, pady=2)

        self.input_preview_frame = tk.LabelFrame(top_frame, text=" Input Preview ", bg=BG_COLOR, relief='groove',
                                                 borderwidth=2, width=self.window_w - settings_w - 30, height=top_h)
        self.input_preview_frame.pack(side='left', fill='both', padx=(5, 0))
        self.input_preview_frame.pack_propagate(False)
        self.input_lbl = tk.Label(self.input_preview_frame, bg=BG_COLOR)
        self.input_lbl.pack(expand=True, fill='both', padx=5, pady=5)

        bottom_frame = tk.Frame(master, bg=BG_COLOR)
        bottom_frame.pack(fill='both', expand=True, padx=10, pady=(0, 10))

        # Inside __init__, find the Mosaic Preview section:
        self.mosaic_frame = tk.LabelFrame(bottom_frame, text=" Mosaic Preview ", bg=BG_COLOR, relief='groove',
                                          borderwidth=2, height=bottom_h)
        self.mosaic_frame.pack_propagate(False)
        self.mosaic_frame.pack(fill='both', expand=True)

        self.output_lbl = tk.Label(self.mosaic_frame, bg=BG_COLOR)
        self.output_lbl.pack(expand=True, fill='both', padx=5)

        # Pack buttons FIRST (at the bottom)
        btn_row = tk.Frame(self.mosaic_frame, bg=BG_COLOR)
        btn_row.pack(fill='x', side='bottom', pady=5)  # Added small pady

        self.save_btn = tk.Button(btn_row, text="💾 Save", bg=ACCENT_BLUE, fg="white", font=('Segoe UI', 9, 'bold'),
                                  command=self.save_output_mosaic, state='disabled', height=2)
        self.save_btn.pack(side='left', expand=True, fill='x', padx=5, pady=5)

        self.clear_btn = tk.Button(btn_row, text="❌ Clear", bg="#FF3333", font=('Segoe UI', 9), command=self.clear_all,
                                   height=2)
        self.clear_btn.pack(side='left', expand=True, fill='x', padx=5, pady=5)

        master.protocol("WM_DELETE_WINDOW", self.on_closing)

    def apply_overlay_filter(self):
        # 1. Get the value from the slider
        val = self.overlay_alpha.get()

        # 2. Check if the mosaic has actually been generated yet
        if self.base_mosaic is not None:
            # We combine the layers for the preview
            self.display_array = cv2.addWeighted(
                self.base_mosaic, 1.0,
                self.original_resized, val,
                0
            )
            # Update the preview label
            self.render_preview(self.display_array, self.output_lbl)

    def render_preview(self, cv_img, label_widget):
        # 1. Force a geometry update to get accurate container dimensions
        self.master.update_idletasks()

        # 2. Identify the container (the LabelFrame)
        parent = label_widget.master  # This is self.mosaic_frame

        # 3. Calculate strict maximums
        # We subtract ~80-100 pixels to account for:
        # - The 'Mosaic Preview' text header
        # - The Save/Clear button row at the bottom
        # - Internal padding
        max_w = parent.winfo_width() - 30
        max_h = parent.winfo_height() - 90

        # Fallback for initialization
        if max_w < 50: max_w, max_h = 400, 300

        # 4. Prepare the image
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        # 5. Fit to the 'Safe Zone'
        img_w, img_h = img_pil.size
        scale = min(max_w / img_w, max_h / img_h)
        new_size = (int(img_w * scale), int(img_h * scale))

        # 6. Final rendering
        img_pil = img_pil.resize(new_size, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img_pil)

        # We use anchor='center' to keep it tidy if the image is small
        label_widget.config(image=photo, anchor='center')
        label_widget.image = photo

    def browse_dir(self):
        d = filedialog.askdirectory()
        if d: self.img_dir.set(d)

    def browse_file(self):
        f = filedialog.askopenfilename()
        if f:
            img = safe_imread(f)  # FIXED
            if img is not None:
                self.input_file.set(f)
                self.render_preview(img, self.input_lbl)
            else:
                messagebox.showerror("Error", "Selected file is not a valid image.")

    def clear_all(self):
        self.output_lbl.config(image='')
        self.save_btn.config(state='disabled')
        self.base_mosaic = None
        self.status_lbl.config(text="Ready")

    def start_generation_thread(self):
        if not self.img_dir.get() or not self.input_file.get(): return
        self.shared_counter.value = 0
        self.generation_running = True
        self.gen_btn.config(state='disabled')
        threading.Thread(target=self.run_generation, daemon=True).start()
        self.update_progress_loop()

    def update_progress_loop(self):
        if not self.generation_running: return
        if self.total_tasks > 0:
            val = self.shared_counter.value
            self.prog['value'] = (val / self.total_tasks) * 100
            self.status_lbl.config(text=f"Processing... {val}/{self.total_tasks}")
        self.master.after(100, self.update_progress_loop)

    def run_generation(self):
        try:
            gen = generate_mosaic_core(self.input_file.get(), self.img_dir.get(), self.ratio.get(), mp.cpu_count(),
                                       self.shared_counter)
            self.total_tasks = next(gen)
            self.base_mosaic = next(gen)
            orig = safe_imread(self.input_file.get())  # FIXED
            self.original_resized = cv2.resize(orig, (self.base_mosaic.shape[1], self.base_mosaic.shape[0]))
            self.master.after(0, self.on_success)
        except Exception as e:
            self.master.after(0, lambda: messagebox.showerror("Error", f"Generation failed: {e}"))
            self.generation_running = False
            self.master.after(0, lambda: self.gen_btn.config(state='normal'))

    def on_success(self):
        self.generation_running = False
        self.gen_btn.config(state='normal')
        self.save_btn.config(state='normal')
        self.status_lbl.config(text="Done!")
        self.apply_overlay_filter()

    def save_output_mosaic(self):
        if self.display_array is None: return

        fp = filedialog.asksaveasfilename(defaultextension=".jpg",
                                          filetypes=[("JPEG Image", "*.jpg"), ("PNG Image", "*.png")])
        if fp:
            # The display_array is already the 'combined' version
            # We use the safe_imwrite logic for Unicode paths
            ext = os.path.splitext(fp)[1]
            success, nparr = cv2.imencode(ext, self.display_array)
            if success:
                nparr.tofile(fp)
                messagebox.showinfo("Success", "Mosaic saved successfully!")

    def on_closing(self):
        self.generation_running = False
        active_children = mp.active_children()
        for p in active_children:
            p.terminate()
            p.join(timeout=0.1)
        self.master.destroy()
        os._exit(0)  # type: ignore


if __name__ == "__main__":
    mp.freeze_support()
    try:
        mp.set_start_method('spawn', force=True)
    except:
        pass
    root = tk.Tk()
    app = MosaicGeneratorApp(root)
    root.mainloop()