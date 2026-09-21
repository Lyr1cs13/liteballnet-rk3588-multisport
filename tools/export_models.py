"""Export all handed-off LiteBallNet checkpoints to RK3588 RKNN models.

The source demo builds a BGR current/previous/pre-previous stack, then divides
by 255.  The RKNN config below owns that division, so the board receives the
same raw uint8 NCHW tensor expected by the calibration data.
"""

import argparse
import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from rknn.api import RKNN

from liteballnet_model import LiteBallNet


SPORTS = ('tennis', 'table_tennis', 'badminton')


def build_windows(video, width, height, every, destination):
    cap = cv2.VideoCapture(str(video))
    frames = deque(maxlen=3)
    samples = []
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(frame, (width, height)))  # Preserve BGR ordering.
        if len(frames) == 3 and index % every == 2:
            tensor = np.concatenate(list(frames)[::-1], axis=2).transpose(2, 0, 1)[None]
            path = destination / f'{index:04d}.npy'
            np.save(path, np.ascontiguousarray(tensor))
            samples.append(path)
        index += 1
    cap.release()
    return samples


def export_one(root, sport, quant, width, height, calibration_stride):
    checkpoint = root / 'models' / f'{sport}_full_seed37.pt'
    video = root / 'sources' / f'{sport}.mp4'
    tag = f'{sport}_360x640_{("int8" if quant else "fp")}'
    model = LiteBallNet().eval()
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if isinstance(state, dict) and 'model_state' in state:
        state = state['model_state']
    model.load_state_dict(state, strict=True)

    onnx_path = root / 'models' / f'{tag}.onnx'
    example = torch.rand(1, 9, height, width)
    torch.onnx.export(model, example, onnx_path, opset_version=11,
                      input_names=['input'], output_names=['heatmap'])

    calibration_dir = root / 'calibration' / sport
    calibration_dir.mkdir(parents=True, exist_ok=True)
    samples = build_windows(video, width, height, calibration_stride, calibration_dir)
    if not samples:
        raise RuntimeError(f'No calibration samples from {video}')
    dataset = root / 'calibration' / f'{sport}.txt'
    dataset.write_text('\n'.join(map(str, samples)), encoding='utf-8')
    with torch.no_grad():
        reference = model(torch.from_numpy(np.load(samples[0]).astype(np.float32) / 255.0)).numpy()
    np.save(root / 'models' / f'{tag}_reference.npy', reference)

    rknn = RKNN(verbose=False)
    try:
        rknn.config(target_platform='rk3588', mean_values=[[0] * 9], std_values=[[255] * 9],
                    optimization_level=3)
        if rknn.load_onnx(model=str(onnx_path)) != 0:
            raise RuntimeError(f'ONNX load failed for {sport}')
        if rknn.build(do_quantization=quant, dataset=str(dataset) if quant else None) != 0:
            raise RuntimeError(f'RKNN build failed for {sport}')
        output = root / 'models' / f'{tag}.rknn'
        if rknn.export_rknn(str(output)) != 0:
            raise RuntimeError(f'RKNN export failed for {sport}')
    finally:
        rknn.release()

    metadata = {
        'sport': sport,
        'checkpoint': checkpoint.name,
        'input_layout': 'NCHW',
        'input_shape': [1, 9, height, width],
        'input_semantics': 'BGR current, previous, pre-previous; raw uint8 on board; /255 in RKNN config',
        'output_shape': list(reference.shape),
        'precision': 'int8' if quant else 'fp',
        'parameters': sum(item.numel() for item in model.parameters()),
        'calibration_windows': len(samples),
    }
    (root / 'models' / f'{tag}.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sport', choices=SPORTS, default=None)
    parser.add_argument('--fp-only', action='store_true')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=360)
    parser.add_argument('--calibration-stride', type=int, default=5)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sports = (args.sport,) if args.sport else SPORTS
    for sport in sports:
        export_one(root, sport, False, args.width, args.height, args.calibration_stride)
        if not args.fp_only:
            export_one(root, sport, True, args.width, args.height, args.calibration_stride)


if __name__ == '__main__':
    main()
