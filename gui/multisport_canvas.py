#!/usr/bin/env python3
"""Touch-friendly QiuWu AI styled board verification console."""

import argparse
import csv
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk

import cv2
from PIL import Image, ImageTk


SPORTS = {
    "tennis": ("网球追踪", "TrackNet时序热图 | 三帧球路追踪", "#ccff00", 27.20),
    "table_tennis": ("乒乓球追踪", "TrackNet时序热图 | 高速小目标追踪", "#58a6ff", 27.78),
    "badminton": ("羽毛球追踪", "TrackNet时序热图 | 空间轨迹可视化", "#f0a050", 27.10),
}
BG, PANEL, PANEL_DARK, BORDER, TEXT, MUTED = "#0a0e14", "#11161f", "#0d1219", "#2a3040", "#c9d1d9", "#8b949e"


def read_text(path, default="Unavailable"):
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").replace("\x00", "").strip()
    except OSError:
        return default


def csv_rows(path):
    try:
        with open(path, encoding="utf-8", newline="") as handle:
            return max(sum(1 for _ in csv.reader(handle)) - 1, 0)
    except OSError:
        return 0


class Showcase:
    def __init__(self, root_path, sport):
        self.root_path = Path(root_path)
        self.outputs = self.root_path / "outputs"
        self.sport = sport
        self.capture = None
        self.video_fps = 30.0
        self.paused = False
        self.image_ref = None
        self.video_item = None
        self.result_roots = {}
        self.run_queue = queue.Queue()
        self.process = None
        self.progress = 0
        self.progress_text = ""
        self.logs = []

        self.root = tk.Tk()
        self.root.title("QiuWu AI | 球悟AI 多球种端侧验证")
        self.root.geometry("1024x600")
        self.root.attributes("-fullscreen", True)
        self.root.configure(background=BG)
        self.root.bind("<Escape>", lambda _event: self.root.destroy())
        self.canvas = tk.Canvas(self.root, background=BG, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _event: self.redraw())
        self.canvas.bind("<Button-1>", self.on_click)

        self.select_sport(sport)
        self.root.after(0, self.next_frame)
        self.root.after(0, self.poll_runner)
        self.root.after(0, self.refresh_status)

    def result_root(self):
        return self.result_roots.get(self.sport, self.outputs)

    def select_sport(self, sport):
        self.sport = sport
        if self.capture is not None:
            self.capture.release()
        self.capture = cv2.VideoCapture(str(self.result_root() / f"{sport}_reference_fp.mp4"))
        self.video_fps = self.capture.get(cv2.CAP_PROP_FPS) or 30.0
        self.paused = False
        self.refresh_result_log()
        self.redraw()

    def result_data(self):
        root = self.result_root()
        try:
            data = json.loads((root / f"{self.sport}_reference_fp.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        return data, csv_rows(root / f"{self.sport}_reference_fp.csv")

    def refresh_result_log(self):
        data, rows = self.result_data()
        label, _, _, _ = SPORTS[self.sport]
        fps = float(data.get("pure_inference_fps", 0.0))
        self.logs = [
            "[DEVICE] " + read_text("/proc/device-tree/model"),
            "[BACKEND] RKNNLite2 / RKNPU v2 / NPU0 NPU1 NPU2",
            f"[RESULT] {label} | average pure inference: {fps:.2f} FPS | CSV rows: {rows}",
            f"[COMMAND] python3 edge/run_reference_pipeline.py --sport {self.sport} --root .",
        ]

    def add_log(self, text):
        if text:
            self.logs.append(text)
            self.logs = self.logs[-5:]
            self.redraw()

    def draw_text(self, x, y, text, size=10, color=TEXT, bold=False, anchor="nw", width=0):
        font = ("Noto Sans CJK SC", size, "bold" if bold else "normal")
        kwargs = {"anchor": anchor, "fill": color, "font": font}
        if width:
            kwargs["width"] = width
        self.canvas.create_text(x, y, text=text, tags="ui", **kwargs)

    def card(self, x1, y1, x2, y2, title):
        self.canvas.create_rectangle(x1, y1, x2, y2, fill=PANEL, outline=BORDER, width=1)
        self.draw_text(x1 + 10, y1 + 9, title, size=8, color="#ccff00", bold=True)

    def redraw(self):
        if not hasattr(self, "canvas"):
            return
        c = self.canvas
        c.delete("ui")
        width, height = max(c.winfo_width(), 1024), max(c.winfo_height(), 600)
        sx, sy = width / 1024.0, height / 600.0
        c.create_rectangle(0, 0, width, height, fill=BG, outline="", tags="ui")
        c.create_rectangle(0, 0, width, 43 * sy, fill="#161b22", outline="#2a3040", tags="ui")
        self.draw_text(14 * sx, 13 * sy, "QiuWu AI | 球悟AI", 13, "#ccff00", True)
        self.draw_text(170 * sx, 15 * sy, "多球种端侧视觉追踪验证", 10, "#58a6ff")
        for index, core in enumerate(("C0", "C1", "C2")):
            x = width - (184 - index * 38) * sx
            c.create_rectangle(x, 11 * sy, x + 31 * sx, 32 * sy, fill="#1a7f37", outline="", tags="ui")
            self.draw_text(x + 15 * sx, 21 * sy, core, 8, "#ffffff", True, "center")
        _, _, _, rolling = SPORTS[self.sport]
        self.draw_text(width - 58 * sx, 21 * sy, f"FPS:{rolling:.2f}", 8, TEXT, True, "center")

        left_x, center_x, right_x = 8 * sx, 220 * sx, width - 233 * sx
        self.card(left_x, 53 * sy, 210 * sx, 313 * sy, "场景模式选择")
        for index, (key, (label, subtitle, color, _)) in enumerate(SPORTS.items()):
            y1, y2 = (80 + index * 72) * sy, (145 + index * 72) * sy
            selected = key == self.sport
            c.create_rectangle(18 * sx, y1, 200 * sx, y2, fill="#1a2810" if selected else "#161b22",
                               outline=color if selected else "#30363d", width=2 if selected else 1, tags="ui")
            self.draw_text(29 * sx, y1 + 12 * sy, label, 10, color if selected else TEXT, True)
            self.draw_text(29 * sx, y1 + 34 * sy, subtitle, 8, MUTED, width=155 * sx)

        self.card(left_x, 322 * sy, 210 * sx, 392 * sy, "统一设备控制")
        self.draw_text(18 * sx, 350 * sy, "输入源: 演示视频\n后端: FP RKNN模型", 9, MUTED)
        self.card(left_x, 401 * sy, 210 * sx, 510 * sy, "任务控制")
        c.create_rectangle(18 * sx, 430 * sy, 140 * sx, 462 * sy, fill="#ccff00", outline="", tags="ui")
        self.draw_text(79 * sx, 446 * sy, "▶ 启动运行(RUN)", 9, BG, True, "center")
        c.create_rectangle(146 * sx, 430 * sy, 200 * sx, 462 * sy, fill="#a02020", outline="", tags="ui")
        self.draw_text(173 * sx, 446 * sy, "■ 暂停", 8, "#ffffff", True, "center")
        task = "● 运行中: 板端FP RKNN推理" if self.process else "就绪 — 选择球类后点击RUN"
        self.draw_text(18 * sx, 475 * sy, task, 8, "#ccff00" if self.process else MUTED, width=175 * sx)

        video_x1, video_y1, video_x2, video_y2 = center_x, 53 * sy, right_x - 8 * sx, 375 * sy
        c.create_rectangle(video_x1, video_y1, video_x2, video_y2, fill="#000000", outline=BORDER, tags="ui")
        if self.video_item is None:
            self.draw_text((video_x1 + video_x2) / 2, (video_y1 + video_y2) / 2, "加载板端结果视频...", 11, MUTED, anchor="center")
        c.create_rectangle(center_x, 385 * sy, center_x + 105 * sx, 418 * sy, fill="#a3cc00", outline="", tags="ui")
        self.draw_text(center_x + 52 * sx, 401 * sy, "继续回放" if self.paused else "暂停回放", 9, BG, True, "center")
        c.create_rectangle(center_x + 112 * sx, 385 * sy, center_x + 202 * sx, 418 * sy, fill="#1e2430", outline=BORDER, tags="ui")
        self.draw_text(center_x + 157 * sx, 401 * sy, "从头播放", 9, TEXT, False, "center")
        status = "回放已暂停" if self.paused else "回放中"
        self.draw_text(video_x2, 401 * sy, status, 9, "#d29922" if self.paused else "#3fb950", anchor="e")
        c.create_rectangle(center_x, 429 * sy, video_x2, 436 * sy, fill="#141b26", outline="", tags="ui")
        c.create_rectangle(center_x, 429 * sy, center_x + (video_x2 - center_x) * self.progress / 100.0, 436 * sy,
                           fill="#ccff00", outline="", tags="ui")
        self.draw_text(center_x, 442 * sy, self.progress_text, 8, MUTED)

        self.card(right_x, 53 * sy, width - 8 * sx, 183 * sy, "端侧推理指标")
        label, _, _, rolling = SPORTS[self.sport]
        data, rows = self.result_data()
        average = float(data.get("pure_inference_fps", 0.0))
        self.draw_text(right_x + 10 * sx, 81 * sy, f"{label} | 滑动纯推理FPS", 8, TEXT, True, width=205 * sx)
        self.draw_text(right_x + 10 * sx, 105 * sy, f"{rolling:.2f} FPS", 21, "#ccff00", True)
        self.draw_text(right_x + 10 * sx, 143 * sy, f"24窗口滑动值 | 全片平均{average:.2f}FPS", 8, MUTED, width=205 * sx)
        self.card(right_x, 191 * sy, width - 8 * sx, 295 * sy, "端侧部署配置")
        self.draw_text(right_x + 10 * sx, 221 * sy, "平台: RK3588\n后端: RKNNLite2 / RKNPU v2\n模型: FP16 RKNN\n并行: NPU0 / NPU1 / NPU2", 9, TEXT)
        self.card(right_x, 303 * sy, width - 8 * sx, 407 * sy, "验证结果资产")
        self.draw_text(right_x + 10 * sx, 333 * sy, f"视频: {self.sport}_reference_fp.mp4\n坐标记录: {rows}帧\n测试窗口: {data.get('windows', '--')}\nJSON/CSV已生成", 9, TEXT, width=205 * sx)

        c.create_rectangle(8 * sx, 520 * sy, width - 8 * sx, height - 8 * sy, fill="#07100c", outline=BORDER, tags="ui")
        for index, line in enumerate(self.logs[-4:]):
            self.draw_text(18 * sx, (532 + index * 15) * sy, line, 8, "#cbeed1", width=(width - 40 * sx))
        if self.video_item is not None:
            c.tag_raise(self.video_item)

    def on_click(self, event):
        x, y = event.x, event.y
        if 18 <= x <= 200:
            for index, key in enumerate(SPORTS):
                if 80 + index * 72 <= y <= 145 + index * 72:
                    self.select_sport(key)
                    return
        if 18 <= x <= 140 and 430 <= y <= 462:
            self.start_run()
        elif 146 <= x <= 200 and 430 <= y <= 462:
            self.toggle_pause()
        elif 220 <= x <= 325 and 385 <= y <= 418:
            self.toggle_pause()
        elif 332 <= x <= 422 and 385 <= y <= 418:
            self.restart_video()

    def toggle_pause(self):
        self.paused = not self.paused
        self.redraw()

    def restart_video(self):
        if self.capture is not None:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.paused = False
        self.redraw()

    def start_run(self):
        if self.process and self.process.poll() is None:
            return
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = self.outputs / "verification_runs" / self.sport / stamp
        output_dir.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(self.root_path / "edge" / "run_reference_pipeline.py"),
                   "--sport", self.sport, "--root", str(self.root_path),
                   "--output-dir", str(output_dir), "--progress"]
        self.progress, self.progress_text = 2, "运行：准备模型与视频"
        self.add_log("[RUN] " + " ".join(command))
        self.process = subprocess.Popen(command, cwd=self.root_path, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, bufsize=1)

        def collect():
            for line in self.process.stdout:
                self.run_queue.put(("line", line.rstrip()))
            self.run_queue.put(("finished", self.process.wait(), str(output_dir), self.sport))

        threading.Thread(target=collect, daemon=True).start()
        self.redraw()

    def poll_runner(self):
        try:
            while True:
                kind, *data = self.run_queue.get_nowait()
                if kind == "line":
                    line = data[0]
                    if line.startswith("PROGRESS|"):
                        _, percent, message = line.split("|", 2)
                        self.progress, self.progress_text = int(percent), "运行：" + message
                    elif line:
                        self.add_log(line)
                else:
                    code, output_dir, sport = data
                    self.process = None
                    if code == 0:
                        self.result_roots[sport] = Path(output_dir)
                        self.progress, self.progress_text = 100, "运行完成：新结果已生成"
                        self.add_log("[OK] 新板端结果已写入 " + output_dir)
                        self.select_sport(sport)
                    else:
                        self.progress, self.progress_text = 0, f"板端运行失败，返回码{code}"
                        self.add_log(f"[ERR] 运行失败，返回码{code}")
        except queue.Empty:
            pass
        self.root.after(100, self.poll_runner)

    def next_frame(self):
        if self.capture is not None and not self.paused:
            ok, frame = self.capture.read()
            if not ok:
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.capture.read()
            if ok:
                height, width = frame.shape[:2]
                scale = min(550 / width, 310 / height)
                image = cv2.cvtColor(cv2.resize(frame, (int(width * scale), int(height * scale))), cv2.COLOR_BGR2RGB)
                self.image_ref = ImageTk.PhotoImage(Image.fromarray(image))
                if self.video_item is None:
                    self.video_item = self.canvas.create_image(500, 214, image=self.image_ref, anchor="center")
                else:
                    self.canvas.itemconfig(self.video_item, image=self.image_ref)
        self.root.after(max(1, int(1000 / self.video_fps)), self.next_frame)

    def refresh_status(self):
        self.root.after(1000, self.refresh_status)

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
