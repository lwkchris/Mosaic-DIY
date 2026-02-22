import tkinter as tk

# --- Styling Constants ---
BG_COLOR = "#f0f0f0"
ACCENT_BLUE = "#0078d7"
ACCENT_GREEN = "#4caf50"
ACCENT_RED = "#f44336"

def create_input_row(parent, label, var, cmd):
    row = tk.Frame(parent, bg=BG_COLOR)
    row.pack(fill='x', pady=5, padx=10)
    tk.Button(row, text=f"{label}", fg="white", bg=ACCENT_GREEN, command=cmd, width=15).pack(side='left')
    tk.Entry(row, textvariable=var, bg="white").pack(side='left', fill='x', expand=True, padx=(5, 0))