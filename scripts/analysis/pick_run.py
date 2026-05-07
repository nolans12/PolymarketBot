"""
pick_run.py - Shared helper: show a Tkinter listbox of data/ run folders,
return the chosen Path. Import and call pick_run_folder() from any analysis script.

Falls back to the most-recent folder automatically if --run is passed on the CLI,
or if Tkinter is unavailable (headless environments).
"""

import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _all_run_folders() -> list[Path]:
    """Return all subdirs of data/, sorted newest-first by name."""
    if not DATA_DIR.exists():
        return []
    folders = sorted(
        [p for p in DATA_DIR.iterdir() if p.is_dir()],
        reverse=True,
    )
    return folders


def pick_run_folder(cli_arg: str | None = None,
                    title: str = "Select a run folder") -> Path:
    """
    Return a Path to the chosen data/<run> folder.

    Priority:
      1. cli_arg if provided (--run data/2026-05-07_00-15-00_BTC_ETH)
      2. Tkinter popup listing all folders in data/
      3. Most-recent folder if Tk unavailable

    Exits with an error message if no folders exist.
    """
    folders = _all_run_folders()
    if not folders:
        sys.exit(f"ERROR: no run folders found in {DATA_DIR}. "
                 "Run the bot first to generate data.")

    if cli_arg:
        p = Path(cli_arg)
        if not p.is_absolute():
            p = Path(__file__).resolve().parents[2] / p
        if not p.exists():
            sys.exit(f"ERROR: run folder not found: {p}")
        return p

    # Try Tkinter popup
    try:
        import tkinter as tk
        from tkinter import font as tkfont

        selected = [None]

        root = tk.Tk()
        root.title(title)
        root.resizable(False, False)

        tk.Label(root, text="Select a run folder to analyse:",
                 font=("Segoe UI", 11), pady=8).pack()

        frame = tk.Frame(root)
        frame.pack(padx=16, pady=(0, 8), fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL)
        lb = tk.Listbox(frame, yscrollcommand=scrollbar.set,
                        font=("Consolas", 10), width=55, height=min(20, len(folders)),
                        selectmode=tk.SINGLE, activestyle="dotbox")
        scrollbar.config(command=lb.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        for f in folders:
            lb.insert(tk.END, f.name)
        lb.selection_set(0)   # default to most recent

        def _ok():
            idxs = lb.curselection()
            if idxs:
                selected[0] = folders[idxs[0]]
            root.destroy()

        def _on_double(event):
            _ok()

        lb.bind("<Double-Button-1>", _on_double)
        tk.Button(root, text="Open", command=_ok,
                  font=("Segoe UI", 10), width=12).pack(pady=(0, 12))

        root.bind("<Return>", lambda e: _ok())
        root.bind("<Escape>", lambda e: root.destroy())
        root.mainloop()

        if selected[0] is None:
            sys.exit("No run folder selected.")
        print(f"Run folder: {selected[0]}", flush=True)
        return selected[0]

    except Exception:
        # Headless fallback
        print(f"Tkinter unavailable; using most recent run: {folders[0].name}",
              flush=True)
        return folders[0]
