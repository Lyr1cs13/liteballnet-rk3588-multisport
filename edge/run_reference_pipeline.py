#!/usr/bin/env python3
"""RK3588 inference backend with the handed-off LiteBallNet post-processing unchanged."""
import argparse
import copy
import csv
import json
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from rknnlite.api import RKNNLite

from reference_postprocess import (
    adjacent_track_stats,
    interpolate_guarded,
    remove_short_fragments,
    remove_temporal_spikes,
    stabilize_v37,
    topk_heatmap_candidates,
)


SPORTS = {
    'tennis': {'model': 'models/tennis_360x640_fp.rknn', 'video': 'sources/tennis.mp4', 'output_fps': 0.0},
    'table_tennis': {'model': 'models/table_tennis_360x640_fp.rknn', 'video': 'sources/table_tennis.mp4', 'output_fps': 30.0},
    'badminton': {'model': 'models/badminton_360x640_fp.rknn', 'video': 'sources/badminton.mp4', 'output_fps': 30.0},
}


def report_progress(enabled, percent, message):
    """Emit machine-readable stage updates for the desktop verification UI."""
    if enabled:
        print(f'PROGRESS|{percent}|{message}', flush=True)


def reference_args():
    """Exact defaults used by the transferred renderer's reference pipeline."""
    return SimpleNamespace(
        threshold=0.30, low_threshold=0.35, high_threshold=0.88, seed_threshold=0.85,
        peak_window=9, top_k=5, suppress_window=15, min_weak_shape=0.05,
        shape_weight=0.05, hough_weight=0.00, enable_hough_quality=False,
        gate_min=60.0, gate_max=220.0, speed_window=5, speed_gate_factor=2.2,
        gate_margin=25.0, strong_gate_factor=1.5, weak_gate_factor=0.85,
        motion_weight=0.20, velocity_alpha=0.65, weak_velocity_alpha=0.35,
        turn_reset_angle=70.0, extreme_turn_angle=120.0, max_accel_ratio=3.0,
        min_speed_for_accel_check=5.0, seed_max_gap=3, weak_reconnect_gap=4,
        max_reconnect_gap=8, fps_adaptive=False, interp_gap=3,
        min_fragment_length=3, isolated_static_max_length=3, isolated_static_radius=8.0,
        static_fragment_max_length=8, static_radius=5.0,
        spike_return_radius=35.0, spike_distance=80.0,
    )


def make_runtime(model_path, core):
    runtime = RKNNLite()
    if runtime.load_rknn(str(model_path)) != 0:
        raise RuntimeError(f'Unable to load {model_path}')
    masks = [RKNNLite.NPU_CORE_0, RKNNLite.NPU_CORE_1, RKNNLite.NPU_CORE_2]
    if runtime.init_runtime(core_mask=masks[core]) != 0:
        raise RuntimeError(f'Unable to initialize NPU core {core}')
    return runtime


def load_temporal_windows(video, input_width=640, input_height=360):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f'Unable to open {video}')
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames, tensors, indices = deque(maxlen=3), [], []
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        if len(frames) == 3:
            current, previous, preprevious = frames[2], frames[1], frames[0]
            stack = np.concatenate([
                cv2.resize(current, (input_width, input_height)),
                cv2.resize(previous, (input_width, input_height)),
                cv2.resize(preprevious, (input_width, input_height)),
            ], axis=2)
            # The optimized RKNN graph exposes the source model's 9-channel tensor as NHWC.
            tensors.append(np.ascontiguousarray(stack[None]))
            indices.append(index)
        index += 1
    cap.release()
    return tensors, indices, index, width, height, fps


def infer(model, tensors, indices, total_frames, warmup):
    engines = [make_runtime(model, core) for core in range(3)]
    outputs, completed = [None] * total_frames, [None] * total_frames

    def worker(core):
        engine = engines[core]
        try:
            for sample in tensors[:min(warmup, len(tensors))]:
                engine.inference(inputs=[sample], data_format=['nhwc'])
            for task in range(core, len(tensors), 3):
                output = engine.inference(inputs=[tensors[task]], data_format=['nhwc'])[0]
                frame_index = indices[task]
                outputs[frame_index] = output
                completed[frame_index] = time.perf_counter()
        finally:
            engine.release()

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(worker, core) for core in range(3)]
        for future in futures:
            future.result()
    elapsed = time.perf_counter() - start

    rolling = [None] * total_frames
    events = sorted((stamp, index) for index, stamp in enumerate(completed) if stamp is not None)
    for position, (stamp, index) in enumerate(events):
        anchor = events[max(0, position - 23)][0]
        count = position - max(0, position - 23) + 1
        if stamp > anchor:
            rolling[index] = count / (stamp - anchor)
    return outputs, rolling, elapsed


def draw_reference_trace(frame, trace, fps):
    """Transferred draw_trace semantics plus the required performance label."""
    for age, point in enumerate(reversed(trace)):
        if point is None:
            continue
        x, y = point
        thickness = max(2, 10 - age)
        cv2.circle(frame, (int(x), int(y)), radius=0, color=(0, 0, 255), thickness=thickness)
    text = 'NPU Inference: -- FPS' if fps is None else f'NPU Inference: {fps:.2f} FPS'
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, .60, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = frame.shape[1] - tw - 28, 38
    cv2.rectangle(frame, (x - 12, y - th - 10), (frame.shape[1] - 16, y + 8), (8, 18, 12), -1)
    cv2.putText(frame, text, (x, y), font, scale, (220, 255, 228), thickness, cv2.LINE_AA)


def write_outputs(video, output, csv_path, stable, rolling, output_fps):
    cap = cv2.VideoCapture(str(video))
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), output_fps or source_fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f'Unable to create {output}')
    trace = deque(maxlen=12)
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        csv_writer = csv.writer(handle)
        csv_writer.writerow(['frame', 'x', 'y', 'score', 'raw_x', 'raw_y', 'raw_score', 'accepted', 'source', 'pure_inference_fps'])
        for index, point in enumerate(stable['tracks']):
            ok, frame = cap.read()
            if not ok:
                break
            trace.append(point)
            draw_reference_trace(frame, trace, rolling[index])
            writer.write(frame)
            raw = stable['raw_tracks'][index]
            csv_writer.writerow([
                index, '' if point is None else f'{point[0]:.2f}', '' if point is None else f'{point[1]:.2f}',
                '' if stable['scores'][index] is None else f"{stable['scores'][index]:.6f}",
                '' if raw is None else f'{raw[0]:.2f}', '' if raw is None else f'{raw[1]:.2f}',
                '' if stable['raw_scores'][index] is None else f"{stable['raw_scores'][index]:.6f}",
                stable['accepted'][index], stable['source'][index],
                '' if rolling[index] is None else f'{rolling[index]:.2f}',
            ])
    writer.release()
    cap.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sport', required=True, choices=SPORTS)
    parser.add_argument('--root', default='.')
    parser.add_argument('--warmup', type=int, default=15)
    parser.add_argument('--output-dir', default='',
                        help='Optional directory for a non-destructive verification result.')
    parser.add_argument('--progress', action='store_true',
                        help='Print stage progress records for a desktop UI.')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    spec = SPORTS[args.sport]
    model, video = root / spec['model'], root / spec['video']
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else root / 'outputs'
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output = output_dir / f'{args.sport}_reference_fp.mp4'
    csv_path = output.with_suffix('.csv')
    output_dir.mkdir(parents=True, exist_ok=True)
    if not model.exists():
        raise FileNotFoundError(model)

    report_progress(args.progress, 8, '读取三帧时序视频')
    tensors, indices, total, width, height, fps = load_temporal_windows(video)
    report_progress(args.progress, 20, '完成输入窗口构建')
    heatmaps, rolling, elapsed = infer(model, tensors, indices, total, args.warmup)
    report_progress(args.progress, 80, '完成三核NPU推理')
    post_args = reference_args()
    scale_x, scale_y = width / 640.0, height / 360.0
    rows = [[] if heatmap is None else topk_heatmap_candidates(np.squeeze(heatmap), post_args, scale_x, scale_y) for heatmap in heatmaps]
    stable = stabilize_v37(rows, width, height, fps, copy.copy(post_args))
    stable = remove_temporal_spikes(stable, post_args)
    stable = interpolate_guarded(stable, post_args.interp_gap, post_args.gate_max)
    stable = remove_short_fragments(stable, post_args)
    report_progress(args.progress, 88, '完成轨迹稳定与清理')
    write_outputs(video, output, csv_path, stable, rolling, spec['output_fps'])
    report_progress(args.progress, 98, '完成结果视频与CSV写入')
    result = {
        'sport': args.sport, 'precision': 'FP16', 'windows': len(tensors),
        'pure_inference_seconds': elapsed, 'pure_inference_fps': len(tensors) / elapsed,
        'scope': 'three RKNN Runtime instances pinned to NPU cores 0/1/2; excludes decode, preprocessing, reference post-processing and video writing',
        'postprocess': 'verbatim logic extracted from the transferred PC renderer; only the NPU backend and FPS overlay are added',
        'output': str(output), 'jump_stats': adjacent_track_stats(stable['tracks']),
    }
    output.with_suffix('.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    report_progress(args.progress, 100, '板端验证完成')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
