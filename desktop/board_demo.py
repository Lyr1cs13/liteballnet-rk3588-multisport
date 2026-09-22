#!/usr/bin/env python3
"""Offline replay viewer for RK3588 deployment evidence.

The viewer plays a board-generated result video and presents the matching
RKNN result metadata and the current board identity in one desktop window.
It intentionally labels the view as a replay, not a live inference session.
"""

import argparse
import base64
import csv
import json
from pathlib import Path
import time
import tkinter as tk
from tkinter import ttk

import cv2


SPORTS = {
    "tennis": {
        "label": "网球",
        "video": "tennis_reference_fp.mp4",
        "json": "tennis_reference_fp.json",
        "csv": "tennis_reference_fp.csv",
        "command": "python3 edge/run_reference_pipeline.py --sport tennis --root .",
    },
    "table_tennis": {
        "label": "乒乓球",
        "video": "table_tennis_reference_fp.mp4",
        "json": "table_tennis_reference_fp.json",
        "csv": "table_tennis_reference_fp.csv",
        "command": "python3 edge/run_reference_pipeline.py --sport table_tennis --root .",
    },
    "badminton": {
        "label": "羽毛球",
        "video": "badminton_reference_fp.mp4",
        "json": "badminton_reference_fp.json",
        "csv": "badminton_reference_fp.csv",
        "command": "python3 edge/run_reference_pipeline.py --sport badminton --root .",
    },
}


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace").replace("\x00", "").strip()
    except OSError:
        return "Unavailable"


def count_csv_rows(path: Path) -> int:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            return max(sum(1 for _ in csv.reader(handle)) - 1, 0)
    except OSError:
        return 0


class BoardDemo:
    def __init__(self, root: Path, initial_sport: str):
        self.root_path = root
        self.output_path = root / "outputs"
        self.window = tk.Tk()
        self.window.title("RK3588 Board Deployment Verification")
        self.window.configure(background="#121820")
        self.window.geometry("1180x820")
        self.window.minsize(900, 650)

        self.sport = tk.StringVar(value=initial_sport)
        self.video = None
        self.video_fps = 30.0
        self.last_frame_at = time.monotonic()
        self.frame_image = None
        self.info = tk.StringVar()
        self.npu_load = tk.StringVar()

        self._build_ui()
        self.load_sport(initial_sport)
        self.window.after(0, self.next_frame)
        self.window.after(0, self.refresh_npu_load)

    def _build_ui(self):
        header = tk.Frame(self.window, background="#182431", padx=22, pady=14)
        header.pack(fill="x")
        tk.Label(
            header,
            text="RK3588端侧部署结果回放",
            font=("Noto Sans CJK SC", 20, "bold"),
            fg="#F4F8FB",
            bg="#182431",
        ).pack(side="left")
        tk.Label(
            header,
            text="已生成板端结果，非实时推理画面",
            font=("Noto Sans CJK SC", 11),
            fg="#A9C7E6",
            bg="#182431",
        ).pack(side="right", padx=(20, 0))

        body = tk.Frame(self.window, background="#121820", padx=22, pady=16)
        body.pack(fill="both", expand=True)
        self.video_label = tk.Label(body, background="#05080B")
        self.video_label.pack(fill="both", expand=True)

        controls = tk.Frame(body, background="#121820", pady=10)
        controls.pack(fill="x")
        for key, entry in SPORTS.items():
            tk.Radiobutton(
                controls,
                text=entry["label"],
                value=key,
                variable=self.sport,
                command=lambda sport=key: self.load_sport(sport),
                font=("Noto Sans CJK SC", 11),
                fg="#E5EDF5",
                bg="#121820",
                selectcolor="#274865",
                activebackground="#121820",
                activeforeground="#FFFFFF",
            ).pack(side="left", padx=(0, 16))

        console = tk.Text(
            body,
            height=12,
            background="#07100C",
            foreground="#CBEFD1",
            insertbackground="#CBEFD1",
            relief="flat",
            padx=16,
            pady=12,
            font=("DejaVu Sans Mono", 10),
            wrap="word",
            state="disabled",
        )
        console.pack(fill="x")
        self.console = console

        footer = tk.Label(
            self.window,
            textvariable=self.npu_load,
            anchor="w",
            font=("DejaVu Sans Mono", 9),
            fg="#90A8B9",
            bg="#121820",
            padx=22,
            pady=8,
        )
        footer.pack(fill="x")

    def load_sport(self, sport: str):
        if self.video is not None:
            self.video.release()

        self.sport.set(sport)
        entry = SPORTS[sport]
        video_path = self.output_path / entry["video"]
        self.video = cv2.VideoCapture(str(video_path))
        self.video_fps = self.video.get(cv2.CAP_PROP_FPS) or 30.0
        self.last_frame_at = time.monotonic()

        try:
            data = json.loads((self.output_path / entry["json"]).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        model = data.get("model", "FP RKNN")
        fps = data.get("fps", "Unavailable")
        samples = data.get("samples", "Unavailable")
        rows = count_csv_rows(self.output_path / entry["csv"])
        device = read_text(Path("/proc/device-tree/model"))

        content = (
            "$ rk3588-board-demo --replay\n"
            f"[DEVICE] {device}\n"
            "[BACKEND] RKNNLite2 / RKNPU v2 / three Runtime workers\n"
            f"[SPORT] {entry['label']}\n"
            f"[MODEL] {model}\n"
            f"[RESULT] {entry['video']}\n"
            f"[MEASURED] average pure inference: {fps} FPS | windows: {samples}\n"
            f"[RECORD] {entry['csv']} | rows: {rows}\n"
            f"[COMMAND] {entry['command']}\n"
            "[NOTE] playback uses a board-generated MP4; the FPS inside the video is a rolling inference value."
        )
        self.console.configure(state="normal")
        self.console.delete("1.0", "end")
        self.console.insert("1.0", content)
        self.console.configure(state="disabled")

    def next_frame(self):
        if self.video is not None:
            ok, frame = self.video.read()
            if not ok:
                self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.video.read()
            if ok:
                width = max(self.video_label.winfo_width(), 640)
                height = max(self.video_label.winfo_height(), 360)
                source_h, source_w = frame.shape[:2]
                scale = min(width / source_w, height / source_h)
                target = (max(int(source_w * scale), 1), max(int(source_h * scale), 1))
                frame = cv2.resize(frame, target, interpolation=cv2.INTER_AREA)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ppm = f"P6 {frame.shape[1]} {frame.shape[0]} 255\n".encode() + frame.tobytes()
                self.frame_image = tk.PhotoImage(data=base64.b64encode(ppm))
                self.video_label.configure(image=self.frame_image)

        elapsed = max(time.monotonic() - self.last_frame_at, 0.001)
        self.last_frame_at = time.monotonic()
        delay = max(int(1000.0 / self.video_fps - elapsed * 1000), 1)
        self.window.after(delay, self.next_frame)

    def refresh_npu_load(self):
        load_path = Path("/sys/kernel/debug/rknpu/load")
        current_load = read_text(load_path)
        if current_load == "Unavailable":
            current_load = "unavailable (requires kernel debug node permission)"
        self.npu_load.set(f"Current RKNPU load: {current_load}")
        self.window.after(1000, self.refresh_npu_load)

    def run(self):
        self.window.mainloop()
        if self.video is not None:
            self.video.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--sport", choices=SPORTS, default="tennis")
    args = parser.parse_args()
    BoardDemo(args.root.resolve(), args.sport).run()


if __name__ == "__main__":
    main()
