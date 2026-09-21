#!/usr/bin/env python3
"""Re-render an existing LiteBallNet result without re-running the NPU."""
import argparse
import csv
from collections import deque
from pathlib import Path

import cv2

from run_liteballnet import draw_overlay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--csv', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--fps', type=float, default=0.0, help='0 keeps the source FPS')
    args = parser.parse_args()

    records = []
    with Path(args.csv).open(newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            point = None if not row['x'] else (float(row['x']), float(row['y']))
            records.append((point, None if not row['pure_inference_fps'] else float(row['pure_inference_fps'])))

    cap = cv2.VideoCapture(args.video)
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_fps = args.fps or source_fps
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), output_fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError('Cannot open output writer')

    trail = deque(maxlen=10)
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        model_index = index - 2
        if 0 <= model_index < len(records):
            point, fps = records[model_index]
            trail.append(point)
            if point is None:
                trail.clear()
        else:
            fps = None
        draw_overlay(frame, trail, fps)
        writer.write(frame)
        index += 1
    writer.release()
    cap.release()


if __name__ == '__main__':
    main()
