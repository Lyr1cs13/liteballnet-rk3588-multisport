#!/usr/bin/env python3
"""QiuWu AI styled RK3588 multi-sport deployment showcase.

This display is intentionally a replay console: it plays board-generated
videos and reads the matching JSON/CSV records instead of claiming to run
the model live. It is designed for the RK3588 touch display.
"""

import argparse
import csv
import json
from pathlib import Path
import time
import tkinter as tk

import cv2
from PIL import Image, ImageTk


SPORTS = {
    "tennis": {
        "label": "网球追踪",
        "tagline": "TrackNet时序热图 | 三帧球路追踪",
        "color": "#ccff00",
        "video": "tennis_reference_fp.mp4",
        "json": "tennis_reference_fp.json",
        "csv": "tennis_reference_fp.csv",
        "command": "python3 edge/run_reference_pipeline.py --sport tennis --root .",
    },
    "table_tennis": {
        "label": "乒乓球追踪",
        "tagline": "TrackNet时序热图 | 高速小目标追踪",
        "color": "#58a6ff",
        "video": "table_tennis_reference_fp.mp4",
        "json": "table_tennis_reference_fp.json",
        "csv": "table_tennis_reference_fp.csv",
        "command": "python3 edge/run_reference_pipeline.py --sport table_tennis --root .",
    },
    "badminton": {
        "label": "羽毛球追踪",
        "tagline": "TrackNet时序热图 | 空间轨迹可视化",
        "color": "#f0a050",
        "video": "badminton_reference_fp.mp4",
        "json": "badminton_reference_fp.json",
        "csv": "badminton_reference_fp.csv",
        "command": "python3 edge/run_reference_pipeline.py --sport badminton --root .",
    },
}


BG = "#0a0e14"
PANEL = "#11161f"
PANEL_DARK = "#0d1219"
BORDER = "#2a3040"
TEXT = "#c9d1d9"
MUTED = "#8b949e"
BLUE = "#58a6ff"
SUCCESS = "#3fb950"


def read_text(path: Path, default: str = "Unavailable") -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace").replace("\x00", "").strip()
    except OSError:
        return default


def csv_rows(path: Path) -> int:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            return max(sum(1 for _ in csv.reader(handle)) - 1, 0)
    except OSError:
        return 0


class Showcase:
    def __init__(self, project_root: Path, sport: str):
        self.project_root = project_root
        self.output_root = project_root / "outputs"
        self.sport = sport
        self.capture = None
        self.video_fps = 30.0
        self.paused = False
        self.image_ref = None
        self.sport_buttons = {}

        self.root = tk.Tk()
        self.root.title("QiuWu AI | 球悟AI 多球种端侧验证")
        self.root.configure(background=BG)
        self.root.geometry("1024x600")
        self.root.minsize(900, 560)
        self.root.attributes("-fullscreen", True)
        self.root.bind("<Escape>", lambda _event: self.root.destroy())
        self.root.bind("<F11>", self.toggle_fullscreen)

        self._build_ui()
        self.select_sport(sport)
        self.root.after(0, self.next_frame)
        self.root.after(0, self.refresh_board_status)

    def _label(self, parent, text, size=10, color=TEXT, bold=False, **kwargs):
        return tk.Label(
            parent,
            text=text,
            foreground=color,
            background=kwargs.pop("background", PANEL),
            font=("Noto Sans CJK SC", size, "bold" if bold else "normal"),
            **kwargs,
        )

    def _card(self, parent, title, row, column, rowspan=1):
        frame = tk.Frame(parent, background=PANEL, highlightbackground=BORDER,
                         highlightthickness=1, padx=10, pady=8)
        frame.grid(row=row, column=column, rowspan=rowspan, sticky="nsew", pady=(0, 7))
        self._label(frame, title.upper(), size=8, color="#ccff00", bold=True).pack(anchor="w")
        return frame

    def _build_ui(self):
        top = tk.Frame(self.root, background="#161b22", height=42, padx=14)
        top.pack(fill="x")
        top.pack_propagate(False)
        self._label(top, "QiuWu AI | 球悟AI", size=13, color="#ccff00", bold=True,
                    background="#161b22").pack(side="left")
        self.mode_label = self._label(top, "多球种端侧视觉追踪验证", size=10, color=BLUE,
                                      background="#161b22")
        self.mode_label.pack(side="left", padx=16)
        self.top_npu = self._label(top, "NPU: --", size=9, color=MUTED, background="#161b22")
        self.top_npu.pack(side="right", padx=8)
        close = tk.Button(top, text="退出", command=self.root.destroy, foreground="#ffffff",
                          background="#a02020", activebackground="#d73a49", activeforeground="#ffffff",
                          relief="flat", bd=0, padx=12, pady=3,
                          font=("Noto Sans CJK SC", 9, "bold"))
        close.pack(side="right", padx=(8, 0))

        body = tk.Frame(self.root, background=BG, padx=8, pady=8)
        body.pack(fill="both", expand=True)
        body.grid_columnconfigure(0, minsize=205)
        body.grid_columnconfigure(1, weight=1)
        body.grid_columnconfigure(2, minsize=230)
        body.grid_rowconfigure(0, weight=1)

        left = tk.Frame(body, background=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 7))
        self._label(left, "运行场景", size=9, color=MUTED, bold=True, background=BG).pack(anchor="w", pady=(0, 5))
        for key, item in SPORTS.items():
            button = tk.Button(
                left, text=item["label"] + "\n" + item["tagline"],
                command=lambda value=key: self.select_sport(value),
                justify="left", anchor="w", wraplength=180,
                foreground=TEXT, background=PANEL, activebackground="#182030",
                activeforeground="#ffffff", relief="flat", bd=0,
                highlightthickness=1, highlightbackground=BORDER,
                padx=10, pady=10, font=("Noto Sans CJK SC", 9),
            )
            button.pack(fill="x", pady=(0, 6))
            self.sport_buttons[key] = button

        source_card = tk.Frame(left, background=PANEL, highlightbackground=BORDER,
                               highlightthickness=1, padx=10, pady=9)
        source_card.pack(fill="x", pady=(5, 0))
        self._label(source_card, "验证方式", size=8, color="#ccff00", bold=True).pack(anchor="w")
        self._label(source_card, "板端FP模型已生成结果回放\n非实时推理展示", size=9, color=MUTED,
                    justify="left", wraplength=175).pack(anchor="w", pady=(6, 0))

        center = tk.Frame(body, background=BG)
        center.grid(row=0, column=1, sticky="nsew", padx=(0, 7))
        video_box = tk.Frame(center, background="#000000", highlightbackground=BORDER,
                             highlightthickness=1)
        video_box.pack(fill="both", expand=True)
        self.video_label = tk.Label(video_box, background="#000000", foreground=MUTED,
                                    text="正在加载板端结果视频...",
                                    font=("Noto Sans CJK SC", 12))
        self.video_label.pack(fill="both", expand=True)
        self.video_label.bind("<Button-1>", lambda _event: self.toggle_pause())

        controls = tk.Frame(center, background=BG, pady=7)
        controls.pack(fill="x")
        self.play_button = tk.Button(controls, text="暂停回放", command=self.toggle_pause,
                                     foreground=BG, background="#a3cc00", activebackground="#bbe020",
                                     relief="flat", bd=0, padx=14, pady=5,
                                     font=("Noto Sans CJK SC", 10, "bold"))
        self.play_button.pack(side="left")
        tk.Button(controls, text="从头播放", command=self.restart_video, foreground=TEXT,
                  background="#1e2430", activebackground="#2a3545", activeforeground="#ffffff",
                  relief="flat", bd=0, padx=14, pady=5,
                  font=("Noto Sans CJK SC", 10)).pack(side="left", padx=7)
        self.video_state = self._label(controls, "板端结果回放中", size=9, color=SUCCESS, background=BG)
        self.video_state.pack(side="right")

        right = tk.Frame(body, background=BG)
        right.grid(row=0, column=2, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        metric_card = self._card(right, "板端推理指标", 0, 0)
        self.metric_title = self._label(metric_card, "--", size=11, color=TEXT, bold=True)
        self.metric_title.pack(anchor="w", pady=(7, 2))
        self.metric_fps = self._label(metric_card, "-- FPS", size=23, color="#ccff00", bold=True)
        self.metric_fps.pack(anchor="w")
        self.metric_note = self._label(metric_card, "纯模型推理，不含前后处理", size=8, color=MUTED)
        self.metric_note.pack(anchor="w", pady=(2, 0))

        backend_card = self._card(right, "部署配置", 1, 0)
        self.backend_info = self._label(backend_card, "", size=9, color=TEXT, justify="left", wraplength=205)
        self.backend_info.pack(anchor="w", pady=(7, 0))

        record_card = self._card(right, "结果资产", 2, 0)
        self.record_info = self._label(record_card, "", size=9, color=TEXT, justify="left", wraplength=205)
        self.record_info.pack(anchor="w", pady=(7, 0))

        console_wrap = tk.Frame(self.root, background=BG, padx=8, pady=0)
        console_wrap.pack(fill="x", pady=(0, 8))
        self.console = tk.Text(console_wrap, height=6, background="#07100c", foreground="#cbeed1",
                               insertbackground="#cbeed1", relief="flat", bd=0, padx=10, pady=7,
                               font=("DejaVu Sans Mono", 8), wrap="word", state="disabled")
        self.console.pack(fill="x")

    def toggle_fullscreen(self, _event=None):
        self.root.attributes("-fullscreen", not self.root.attributes("-fullscreen"))

    def select_sport(self, sport: str):
        self.sport = sport
        entry = SPORTS[sport]
        for key, button in self.sport_buttons.items():
            selected = key == sport
            button.configure(
                background="#1a2810" if selected else PANEL,
                foreground=entry["color"] if selected else TEXT,
                highlightbackground=entry["color"] if selected else BORDER,
            )
        if self.capture is not None:
            self.capture.release()
        self.capture = cv2.VideoCapture(str(self.output_root / entry["video"]))
        self.video_fps = self.capture.get(cv2.CAP_PROP_FPS) or 30.0
        self.paused = False
        self.play_button.configure(text="暂停回放")
        self.video_state.configure(text="板端结果回放中", foreground=SUCCESS)

        try:
            result = json.loads((self.output_root / entry["json"]).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            result = {}
        fps = float(result.get("pure_inference_fps", 0.0))
        windows = result.get("windows", "--")
        precision = result.get("precision", "FP")
        rows = csv_rows(self.output_root / entry["csv"])

        self.metric_title.configure(text=entry["label"])
        self.metric_fps.configure(text=f"{fps:.2f} FPS")
        self.backend_info.configure(
            text=f"平台: RK3588\n后端: RKNNLite2 / RKNPU v2\n模型: {precision} RKNN\n并行: NPU0 / NPU1 / NPU2"
        )
        self.record_info.configure(
            text=f"视频: {entry['video']}\n坐标记录: {rows}帧\n测试窗口: {windows}\nJSON/CSV已生成"
        )
        model = read_text(Path("/proc/device-tree/model"))
        self._set_console(
            f"$ qiuwu-ai edge-showcase --replay\n"
            f"[DEVICE] {model}\n"
            f"[PIPELINE] TrackNet时序热图 -> FP RKNN -> NPU0/1/2并行 -> 参考后处理\n"
            f"[RESULT] {entry['video']} | average pure inference: {fps:.2f} FPS | CSV rows: {rows}\n"
            f"[COMMAND] {entry['command']}\n"
            "[STATUS] 当前页面播放的是已由RK3588生成的结果视频。"
        )

    def _set_console(self, content: str):
        self.console.configure(state="normal")
        self.console.delete("1.0", "end")
        self.console.insert("1.0", content)
        self.console.configure(state="disabled")

    def toggle_pause(self):
        self.paused = not self.paused
        if self.paused:
            self.play_button.configure(text="继续回放")
            self.video_state.configure(text="回放已暂停", foreground="#d29922")
        else:
            self.play_button.configure(text="暂停回放")
            self.video_state.configure(text="板端结果回放中", foreground=SUCCESS)

    def restart_video(self):
        if self.capture is not None:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.paused = False
        self.play_button.configure(text="暂停回放")
        self.video_state.configure(text="板端结果回放中", foreground=SUCCESS)

    def next_frame(self):
        if self.capture is not None and not self.paused:
            ok, frame = self.capture.read()
            if not ok:
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.capture.read()
            if ok:
                width = max(self.video_label.winfo_width() - 6, 320)
                height = max(self.video_label.winfo_height() - 6, 180)
                source_h, source_w = frame.shape[:2]
                scale = min(width / source_w, height / source_h)
                target_w = max(1, int(source_w * scale))
                target_h = max(1, int(source_h * scale))
                image = cv2.cvtColor(cv2.resize(frame, (target_w, target_h)), cv2.COLOR_BGR2RGB)
                self.image_ref = ImageTk.PhotoImage(Image.fromarray(image))
                self.video_label.configure(image=self.image_ref, text="")
        self.root.after(max(1, int(1000 / self.video_fps)), self.next_frame)

    def refresh_board_status(self):
        load = read_text(Path("/sys/kernel/debug/rknpu/load"), "unavailable")
        self.top_npu.configure(text=f"当前RKNPU负载: {load}")
        self.root.after(1000, self.refresh_board_status)

    def run(self):
        self.root.mainloop()
        if self.capture is not None:
            self.capture.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--sport", choices=SPORTS, default="tennis")
    args = parser.parse_args()
    Showcase(args.root.resolve(), args.sport).run()


if __name__ == "__main__":
    main()
