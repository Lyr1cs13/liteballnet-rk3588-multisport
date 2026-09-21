"""RK3588 LiteBallNet demo runner for tennis, table tennis and badminton.

The runner preserves the handed-off model's BGR three-frame input semantics and
keeps NPU timing separate from video I/O and postprocessing.  The only on-video
metric is the measured sliding, three-core pure inference throughput.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from rknnlite.api import RKNNLite


SPORTS = {
    'tennis': {'source': 'sources/tennis.mp4', 'model': 'models/tennis_360x640_int8.rknn', 'output_fps': 0.0},
    'table_tennis': {'source': 'sources/table_tennis.mp4', 'model': 'models/table_tennis_360x640_int8.rknn', 'output_fps': 30.0},
    'badminton': {'source': 'sources/badminton.mp4', 'model': 'models/badminton_360x640_int8.rknn', 'output_fps': 30.0},
}


@dataclass
class Candidate:
    x: float
    y: float
    score: float


def make_tensor(frame: np.ndarray, previous: np.ndarray, preprevious: np.ndarray, width: int, height: int) -> np.ndarray:
    # The source model semantic is BGR current/previous/pre-previous (nine channels).
    # RKNN's optimized RK3588 graph exposes that input as NHWC at runtime.
    stack = np.concatenate([
        cv2.resize(frame, (width, height)),
        cv2.resize(previous, (width, height)),
        cv2.resize(preprevious, (width, height)),
    ], axis=2)
    return np.ascontiguousarray(stack[None])


def load_windows(video: Path, width: int, height: int):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open {video}')
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_w, out_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames: deque[np.ndarray] = deque(maxlen=3)
    tensors: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        if len(frames) == 3:
            tensors.append(make_tensor(frames[-1], frames[-2], frames[-3], width, height))
    cap.release()
    if not tensors:
        raise RuntimeError('Video did not yield any three-frame model windows')
    return tensors, out_w, out_h, source_fps


def infer_parallel(model: Path, tensors: list[np.ndarray], warmup: int):
    """Run only the RKNN inference API in three threads pinned to the three NPU cores."""
    engines = []
    for core in range(3):
        engine = RKNNLite(verbose=False)
        if engine.load_rknn(str(model)) != 0 or engine.init_runtime(core_mask=(1 << core)) != 0:
            raise RuntimeError(f'Failed to initialize Runtime on NPU core {core}')
        engines.append(engine)

    outputs: list[np.ndarray | None] = [None] * len(tensors)
    completed = [0.0] * len(tensors)
    barrier = __import__('threading').Barrier(3)
    try:
        def work(core: int):
            engine = engines[core]
            for n in range(warmup):
                engine.inference(inputs=[tensors[n % len(tensors)]], data_format=['nhwc'])
            barrier.wait(timeout=60)
            for index in range(core, len(tensors), 3):
                outputs[index] = engine.inference(inputs=[tensors[index]], data_format=['nhwc'])[0]
                completed[index] = time.perf_counter()

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(work, core) for core in range(3)]
            for future in futures:
                future.result()
    finally:
        for engine in engines:
            engine.release()

    ordered = sorted((done, index) for index, done in enumerate(completed))
    interval: deque[float] = deque(maxlen=24)
    rolling: list[float | None] = [None] * len(tensors)
    for done, index in ordered:
        interval.append(done)
        if len(interval) >= 6:
            rolling[index] = (len(interval) - 1) / (interval[-1] - interval[0])
    elapsed = max(completed) - min(completed)
    return [out for out in outputs if out is not None], rolling, elapsed


def decode_candidates(output: np.ndarray, width: int, height: int, threshold: float = 0.30, top_k: int = 5):
    heatmap = np.squeeze(output).astype(np.float32)
    if heatmap.ndim != 2:
        raise RuntimeError(f'Unexpected output shape {output.shape}')
    work = heatmap.copy()
    candidates = []
    suppress = 7
    for _ in range(top_k):
        _, score, _, loc = cv2.minMaxLoc(work)
        if score < threshold:
            break
        x, y = loc
        # Response-weighted local centroid, matching the source renderer's peak refinement intent.
        x0, x1 = max(0, x - 4), min(work.shape[1], x + 5)
        y0, y1 = max(0, y - 4), min(work.shape[0], y + 5)
        crop = heatmap[y0:y1, x0:x1]
        weights = np.maximum(crop - threshold, 0.0)
        if float(weights.sum()) > 0:
            xs = np.arange(x0, x1, dtype=np.float32)
            ys = np.arange(y0, y1, dtype=np.float32)
            x = float((weights * xs[None, :]).sum() / weights.sum())
            y = float((weights * ys[:, None]).sum() / weights.sum())
        candidates.append(Candidate(x * width / heatmap.shape[1], y * height / heatmap.shape[0], float(score)))
        lx, ly = loc
        work[max(0, ly - suppress):min(work.shape[0], ly + suppress + 1),
             max(0, lx - suppress):min(work.shape[1], lx + suppress + 1)] = -1.0
    return candidates


def track_candidates(rows: list[list[Candidate]], source_fps: float):
    tracks: list[tuple[float, float] | None] = []
    scores: list[float | None] = []
    last = None
    velocity = (0.0, 0.0)
    gap = 0
    scale = 30.0 / max(source_fps, 1.0)
    for candidates in rows:
        point = None
        score = None
        if last is None:
            if candidates and candidates[0].score >= 0.85:
                chosen = candidates[0]
                point, score = (chosen.x, chosen.y), chosen.score
        else:
            expected = (last[0] + velocity[0], last[1] + velocity[1])
            gate = min(220.0 * scale, max(60.0 * scale, math.hypot(*velocity) * 2.2 + 25.0 * scale))
            ranked = [(candidate.score - 0.20 * min(math.hypot(candidate.x - expected[0], candidate.y - expected[1]) / gate, 1.0), candidate)
                      for candidate in candidates if math.hypot(candidate.x - expected[0], candidate.y - expected[1]) <= gate]
            if ranked:
                chosen = max(ranked, key=lambda item: item[0])[1]
                point, score = (chosen.x, chosen.y), chosen.score
        if point is not None:
            if last is not None:
                measured = (point[0] - last[0], point[1] - last[1])
                velocity = (0.65 * measured[0] + 0.35 * velocity[0], 0.65 * measured[1] + 0.35 * velocity[1])
            last, gap = point, 0
        else:
            gap += 1
            if gap > 8:
                last, velocity = None, (0.0, 0.0)
        tracks.append(point)
        scores.append(score)
    return tracks, scores


def draw_overlay(frame: np.ndarray, trail: deque[tuple[float, float] | None], fps: float | None):
    points = [point for point in trail if point is not None]
    for index in range(1, len(points)):
        progress = index / max(1, len(points) - 1)
        # BGR: a transparent-looking, bright-green trace that strengthens toward the live ball.
        color = (int(10 + 30 * progress), int(175 + 70 * progress), int(10 + 35 * progress))
        cv2.line(frame, tuple(map(int, points[index - 1])), tuple(map(int, points[index])), 1 if progress < .70 else 2, cv2.LINE_AA)
    if points:
        cv2.circle(frame, tuple(map(int, points[-1])), 3, (65, 255, 95), -1, cv2.LINE_AA)
    text = f'NPU Inference: {fps:.2f} FPS' if fps is not None else 'NPU Inference: -- FPS'
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, .60, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = frame.shape[1] - tw - 28, 38
    cv2.rectangle(frame, (x - 12, y - th - 10), (frame.shape[1] - 16, y + 8), (8, 18, 12), -1)
    cv2.putText(frame, text, (x, y), font, scale, (220, 255, 228), thickness, cv2.LINE_AA)


def write_video(video: Path, output: Path, tracks, rolling, width, height, source_fps, output_fps):
    cap = cv2.VideoCapture(str(video))
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), output_fps or source_fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f'Cannot open writer {output}')
    trail: deque[tuple[float, float] | None] = deque(maxlen=10)
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        model_index = index - 2
        if 0 <= model_index < len(tracks):
            point = tracks[model_index]
            trail.append(point)
            if point is None:
                trail.clear()
            draw_overlay(frame, trail, rolling[model_index])
        else:
            draw_overlay(frame, trail, None)
        writer.write(frame)
        index += 1
    writer.release()
    cap.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sport', choices=SPORTS, required=True)
    parser.add_argument('--root', default='.')
    parser.add_argument('--model', default=None)
    parser.add_argument('--video', default=None)
    parser.add_argument('--output', default=None)
    parser.add_argument('--warmup', type=int, default=15)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    spec = SPORTS[args.sport]
    model = Path(args.model) if args.model else root / spec['model']
    video = Path(args.video) if args.video else root / spec['source']
    output = Path(args.output) if args.output else root / 'outputs' / f'{args.sport}_rknn.mp4'
    output.parent.mkdir(parents=True, exist_ok=True)

    tensors, width, height, source_fps = load_windows(video, 640, 360)
    outputs, rolling, elapsed = infer_parallel(model, tensors, args.warmup)
    candidates = [decode_candidates(output, width, height) for output in outputs]
    tracks, scores = track_candidates(candidates, source_fps)
    write_video(video, output, tracks, rolling, width, height, source_fps, spec['output_fps'])
    with (output.with_suffix('.csv')).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['frame', 'x', 'y', 'score', 'pure_inference_fps'])
        for index, (point, score, fps) in enumerate(zip(tracks, scores, rolling), start=2):
            writer.writerow([index, '' if point is None else f'{point[0]:.2f}', '' if point is None else f'{point[1]:.2f}',
                             '' if score is None else f'{score:.6f}', '' if fps is None else f'{fps:.2f}'])
    result = {'sport': args.sport, 'model': str(model), 'video': str(video), 'windows': len(outputs),
              'pure_inference_seconds': elapsed, 'pure_inference_fps': len(outputs) / elapsed,
              'scope': 'three RKNN Runtime instances pinned to NPU cores 0/1/2; excludes decode, preprocessing, postprocessing and video writing',
              'output': str(output)}
    output.with_suffix('.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
