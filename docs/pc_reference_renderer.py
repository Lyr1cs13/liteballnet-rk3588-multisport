import argparse
import csv
import math
import statistics
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch



import torch.nn as nn
import torch.nn.functional as F


class DSConvBlock(nn.Module):
    """Depthwise 3x3 + pointwise 1x1, each followed by BN and ReLU."""

    def __init__(self, in_channels, out_channels, stride=1, bias=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1,
                      groups=in_channels, bias=bias),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class CoordinateAttention(nn.Module):
    """Bottleneck Coordinate Attention (Hou et al., CVPR 2021)."""

    def __init__(self, channels, reduction=32, min_channels=8):
        super().__init__()
        hidden = max(int(min_channels), int(channels) // int(reduction))
        self.shared = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(hidden)
        self.act = nn.Hardswish(inplace=True)
        self.attn_h = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        self.attn_w = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

    def forward(self, x):
        height = x.shape[2]
        x_h = x.mean(dim=3, keepdim=True)
        x_w = x.mean(dim=2, keepdim=True).transpose(2, 3)
        shared = self.act(self.bn(self.shared(torch.cat([x_h, x_w], dim=2))))
        feature_h, feature_w = torch.split(shared, [height, x.shape[3]], dim=2)
        feature_w = feature_w.transpose(2, 3)
        gate_h = torch.sigmoid(self.attn_h(feature_h))
        gate_w = torch.sigmoid(self.attn_w(feature_w))
        return x * gate_h * gate_w


class LiteBallNet(nn.Module):
    """DS U-Net with an optional bottleneck CA block; always returns logits."""

    def __init__(self, base_channels=24, out_channels=1, use_ca=False,
                 ca_reduction=32, ca_min_channels=8, initialization_seed=None):
        super().__init__()
        c1 = int(base_channels)
        c2 = c1 * 2
        c3 = c1 * 4
        self.use_ca = bool(use_ca)

        self.stem = DSConvBlock(9, c1)
        self.enc1 = DSConvBlock(c1, c1)
        self.down1 = DSConvBlock(c1, c2, stride=2)
        self.enc2 = DSConvBlock(c2, c2)
        self.down2 = DSConvBlock(c2, c3, stride=2)
        self.bottleneck1 = DSConvBlock(c3, c3)
        self.bottleneck2 = DSConvBlock(c3, c3)
        self.dec2 = DSConvBlock(c3 + c2, c2)
        self.dec1 = DSConvBlock(c2 + c1, c1)
        self.head = nn.Conv2d(c1, out_channels, kernel_size=1, bias=True)
        self.ca = (
            CoordinateAttention(c3, reduction=ca_reduction, min_channels=ca_min_channels)
            if self.use_ca
            else nn.Identity()
        )
        if initialization_seed is None:
            self._init_weights()
        else:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(initialization_seed))
                self._init_weights()

    def forward(self, x, testing=False):
        e1 = self.enc1(self.stem(x))
        e2 = self.enc2(self.down1(e1))
        x = self.bottleneck2(self.bottleneck1(self.down2(e2)))
        x = self.ca(x)

        x = F.interpolate(x, size=e2.shape[-2:], mode="nearest")
        x = self.dec2(torch.cat([x, e2], dim=1))
        x = F.interpolate(x, size=e1.shape[-2:], mode="nearest")
        x = self.dec1(torch.cat([x, e1], dim=1))
        logits = self.head(x)
        return torch.sigmoid(logits) if testing else logits

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)



def make_input(frame, prev_frame, preprev_frame, input_width, input_height):
    """Stack three BGR frames into the 9-channel network input."""
    img = cv2.resize(frame, (input_width, input_height))
    img_prev = cv2.resize(prev_frame, (input_width, input_height))
    img_preprev = cv2.resize(preprev_frame, (input_width, input_height))
    imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
    imgs = imgs.astype(np.float32) / 255.0
    imgs = np.rollaxis(imgs, 2, 0)
    return imgs


def distance_xy(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def refine_peak(heatmap, peak_x, peak_y, threshold, peak_window, scale_x, scale_y):
    """Response-weighted centroid inside a peak_window neighbourhood of the peak."""
    peak_window = max(3, int(peak_window))
    if peak_window % 2 == 0:
        peak_window += 1
    radius = peak_window // 2
    x0 = max(0, peak_x - radius)
    x1 = min(heatmap.shape[1], peak_x + radius + 1)
    y0 = max(0, peak_y - radius)
    y1 = min(heatmap.shape[0], peak_y + radius + 1)

    crop = heatmap[y0:y1, x0:x1]
    weights = np.maximum(crop - threshold, 0.0)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0:
        return float(peak_x) * scale_x, float(peak_y) * scale_y

    xs = np.arange(x0, x1, dtype=np.float32)
    ys = np.arange(y0, y1, dtype=np.float32)
    x_model = float(np.sum(weights * xs[None, :]) / weight_sum)
    y_model = float(np.sum(weights * ys[:, None]) / weight_sum)
    return x_model * scale_x, y_model * scale_y


def peak_shape_quality(heatmap, peak_x, peak_y, threshold, suppress_window):
    """Compactness and size plausibility of the connected peak region."""
    radius = max(3, int(suppress_window) // 2)
    x0 = max(0, peak_x - radius)
    x1 = min(heatmap.shape[1], peak_x + radius + 1)
    y0 = max(0, peak_y - radius)
    y1 = min(heatmap.shape[0], peak_y + radius + 1)
    crop = heatmap[y0:y1, x0:x1]
    if crop.size == 0:
        return {"area": 0.0, "radius": 0.0, "compactness": 0.0, "shape_quality": 0.0}

    binary = (crop >= threshold).astype(np.uint8)
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, 8)
    if num_labels <= 1:
        return {"area": 0.0, "radius": 0.0, "compactness": 0.0, "shape_quality": 0.0}

    local_x = peak_x - x0
    local_y = peak_y - y0
    label = labels[local_y, local_x]
    if label <= 0:
        label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))

    area = float(stats[label, cv2.CC_STAT_AREA])
    width = float(stats[label, cv2.CC_STAT_WIDTH])
    height = float(stats[label, cv2.CC_STAT_HEIGHT])
    radius_est = 0.5 * max(width, height)
    aspect = min(width, height) / max(width, height, 1.0)
    ideal_area = np.pi * max(radius_est, 1.0) * max(radius_est, 1.0)
    fill_ratio = min(area / max(ideal_area, 1.0), 1.5)
    compactness = max(0.0, min(1.0, aspect)) * max(0.0, min(1.0, fill_ratio))
    size_score = max(0.0, 1.0 - abs(radius_est - 4.0) / 9.0)
    shape_quality = 0.65 * compactness + 0.35 * size_score
    return {
        "area": area,
        "radius": radius_est,
        "compactness": float(compactness),
        "shape_quality": float(max(0.0, min(1.0, shape_quality))),
    }


def hough_circle_quality(heatmap, peak_x, peak_y, threshold, suppress_window):
    """Optional circularity prior; disabled unless --enable-hough-quality is given."""
    radius = max(8, int(suppress_window) // 2)
    x0 = max(0, peak_x - radius)
    x1 = min(heatmap.shape[1], peak_x + radius + 1)
    y0 = max(0, peak_y - radius)
    y1 = min(heatmap.shape[0], peak_y + radius + 1)
    crop = heatmap[y0:y1, x0:x1]
    if crop.size == 0:
        return 0.0
    norm = np.clip((crop - threshold) / max(1e-6, float(crop.max() - threshold)), 0.0, 1.0)
    img = (norm * 255).astype(np.uint8)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=1, minDist=2, param1=40,
                               param2=6, minRadius=2, maxRadius=max(3, radius))
    if circles is None:
        return 0.0
    local_peak = np.asarray([peak_x - x0, peak_y - y0], dtype=np.float32)
    best = 0.0
    for circle in circles[0]:
        center = np.asarray([circle[0], circle[1]], dtype=np.float32)
        dist = float(np.linalg.norm(center - local_peak))
        radius_score = max(0.0, 1.0 - abs(float(circle[2]) - 4.0) / 8.0)
        center_score = max(0.0, 1.0 - dist / max(1.0, radius))
        best = max(best, 0.55 * center_score + 0.45 * radius_score)
    return float(min(1.0, best))


def topk_heatmap_candidates(heatmap, args, scale_x, scale_y):
    """All peaks above threshold, each suppressed before searching the next one."""
    if heatmap.ndim != 2:
        heatmap = np.squeeze(heatmap)
    heatmap = np.asarray(heatmap, dtype=np.float32)
    work = heatmap.copy()
    candidates = []
    suppress = max(3, int(args.suppress_window))
    if suppress % 2 == 0:
        suppress += 1
    radius = suppress // 2

    for _ in range(max(1, int(args.top_k))):
        _, max_value, _, max_loc = cv2.minMaxLoc(work)
        score = float(max_value)
        if score < args.threshold:
            break
        peak_x, peak_y = max_loc
        x, y = refine_peak(heatmap, peak_x, peak_y, args.threshold, args.peak_window, scale_x, scale_y)
        shape = peak_shape_quality(heatmap, peak_x, peak_y, args.threshold, args.suppress_window)
        if getattr(args, "enable_hough_quality", False):
            shape["hough_quality"] = hough_circle_quality(
                heatmap, peak_x, peak_y, args.threshold, args.suppress_window)
        else:
            shape["hough_quality"] = 0.0
        candidates.append({"x": x, "y": y, "score": score, **shape})

        x0 = max(0, peak_x - radius)
        x1 = min(work.shape[1], peak_x + radius + 1)
        y0 = max(0, peak_y - radius)
        y1 = min(work.shape[0], peak_y + radius + 1)
        work[y0:y1, x0:x1] = -1.0

    return candidates


def run_batch(model, device, batch_inputs, width, height, args):
    """Forward one batch and decode every heatmap into candidates."""
    tensor = torch.from_numpy(np.stack(batch_inputs, axis=0)).float().to(device)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)
        heatmaps = torch.sigmoid(logits).detach().cpu().numpy()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    scale_x = width / float(args.input_width)
    scale_y = height / float(args.input_height)
    results = []
    for i in range(heatmaps.shape[0]):
        results.append(topk_heatmap_candidates(heatmaps[i, 0], args, scale_x, scale_y))
    return results, (t1 - t0) * 1000.0



def clamp_point(point, width, height):
    return (
        min(max(float(point[0]), 0.0), float(width - 1)),
        min(max(float(point[1]), 0.0), float(height - 1)),
    )


def candidate_value(candidate, predicted, gate, args):
    """Association score: peak score + shape bonus - motion penalty."""
    point = (candidate["x"], candidate["y"])
    distance = distance_xy(point, predicted)
    if distance > gate:
        return None
    shape_bonus = args.shape_weight * candidate.get("shape_quality", 0.0)
    hough_bonus = args.hough_weight * candidate.get("hough_quality", 0.0)
    motion_penalty = args.motion_weight * min(distance / max(gate, 1e-6), 1.0)
    return candidate["score"] + shape_bonus + hough_bonus - motion_penalty


def choose_tracking_candidate(candidates, predicted, strong_gate, weak_gate, args, allow_weak=True):
    """Pick the best candidate; weak peaks need a stricter shape and a tighter gate."""
    best = None
    best_value = -1e9
    best_distance = None
    for candidate in candidates:
        strong = candidate["score"] >= args.high_threshold
        if not strong:
            if not allow_weak:
                continue
            if candidate["score"] < args.low_threshold:
                continue
            if candidate.get("shape_quality", 0.0) < args.min_weak_shape:
                continue
        gate = strong_gate if strong else weak_gate
        value = candidate_value(candidate, predicted, gate, args)
        if value is None:
            continue
        distance = distance_xy((candidate["x"], candidate["y"]), predicted)
        if value > best_value:
            best = candidate
            best_value = value
            best_distance = distance
    return best, best_distance


def dynamic_motion_gate(recent_speeds, args):
    """Gate scaled by the recent median speed, clamped to [gate_min, gate_max]."""
    if len(recent_speeds) < 2:
        return args.gate_max
    median_speed = statistics.median(recent_speeds)
    gate = median_speed * args.speed_gate_factor + args.gate_margin
    return min(max(gate, args.gate_min), args.gate_max)


def abrupt_change_confirmed(candidate_rows, frame_id, point, measured_velocity, gate, args):
    """An abrupt turn is only accepted if the next frame has a strong peak near it."""
    next_frame = frame_id + 1
    if next_frame >= len(candidate_rows):
        return False
    expected = (point[0] + measured_velocity[0], point[1] + measured_velocity[1])
    confirmation_gate = min(
        args.gate_max,
        max(args.gate_min, gate, math.hypot(*measured_velocity) * 0.75 + args.gate_margin),
    )
    for candidate in candidate_rows[next_frame]:
        if candidate["score"] < args.high_threshold:
            continue
        candidate_point = (candidate["x"], candidate["y"])
        if distance_xy(candidate_point, expected) <= confirmation_gate:
            return True
    return False


def vector_angle_degrees(a, b):
    norm_a = math.hypot(a[0], a[1])
    norm_b = math.hypot(b[0], b[1])
    if norm_a < 1e-6 or norm_b < 1e-6:
        return 0.0
    cosine = (a[0] * b[0] + a[1] * b[1]) / (norm_a * norm_b)
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def stabilize_v37(candidate_rows, width, height, fps, args):

    fps_scale = 30.0 / max(float(fps), 1.0) if args.fps_adaptive else 1.0
    gate_min = args.gate_min * fps_scale
    gate_max = args.gate_max * fps_scale
    args.gate_min = min(gate_min, gate_max)
    args.gate_max = max(gate_min, gate_max)

    count = len(candidate_rows)
    tracks = [None] * count
    scores = [None] * count
    accepted = [0] * count
    sources = ["missing"] * count
    raw_tracks = []
    raw_scores = []
    predicted_tracks = [None] * count
    innovations = [None] * count
    shape_quality = [None] * count
    gaps = [0] * count
    motion_gates = [None] * count

    confirmed = False
    seed = None
    last_position = None
    velocity = (0.0, 0.0)
    gap = 0
    recent_speeds = deque(maxlen=args.speed_window)

    for frame_id, candidates in enumerate(candidate_rows):
        raw = candidates[0] if candidates else None
        raw_tracks.append(None if raw is None else (raw["x"], raw["y"]))
        raw_scores.append(None if raw is None else raw["score"])

        if not confirmed:
            strong = next((c for c in candidates if c["score"] >= args.seed_threshold), None)
            if strong is None:
                seed = None
                gaps[frame_id] = gap
                continue
            point = (strong["x"], strong["y"])
            if seed is None:
                seed = (frame_id, strong)
                sources[frame_id] = "seed_pending"
                scores[frame_id] = strong["score"]
                continue

            seed_frame, seed_candidate = seed
            seed_point = (seed_candidate["x"], seed_candidate["y"])
            frame_gap = frame_id - seed_frame
            seed_gate = args.gate_max * args.strong_gate_factor * max(1, frame_gap)
            if frame_gap <= args.seed_max_gap and distance_xy(point, seed_point) <= seed_gate:
                tracks[seed_frame] = seed_point
                scores[seed_frame] = seed_candidate["score"]
                accepted[seed_frame] = 1
                sources[seed_frame] = "seed_confirmed"
                shape_quality[seed_frame] = seed_candidate.get("shape_quality", 0.0)

                tracks[frame_id] = point
                scores[frame_id] = strong["score"]
                accepted[frame_id] = 1
                sources[frame_id] = "high_observed"
                shape_quality[frame_id] = strong.get("shape_quality", 0.0)
                velocity = (
                    (point[0] - seed_point[0]) / max(frame_gap, 1),
                    (point[1] - seed_point[1]) / max(frame_gap, 1),
                )
                recent_speeds.clear()
                recent_speeds.append(math.hypot(*velocity))
                last_position = point
                confirmed = True
                seed = None
                gap = 0
            else:
                seed = (frame_id, strong)
                sources[frame_id] = "seed_pending"
                scores[frame_id] = strong["score"]
            continue

        if gap >= args.max_reconnect_gap:
            confirmed = False
            seed = None
            last_position = None
            velocity = (0.0, 0.0)
            recent_speeds.clear()
            strong = next((c for c in candidates if c["score"] >= args.seed_threshold), None)
            if strong is not None:
                seed = (frame_id, strong)
                sources[frame_id] = "seed_pending"
                scores[frame_id] = strong["score"]
            else:
                sources[frame_id] = "track_expired"
            continue

        gap += 1
        predicted = clamp_point(
            (last_position[0] + velocity[0] * gap, last_position[1] + velocity[1] * gap),
            width,
            height,
        )
        predicted_tracks[frame_id] = predicted
        base_gate = dynamic_motion_gate(recent_speeds, args)
        strong_gate = min(args.gate_max, base_gate * args.strong_gate_factor)
        weak_gate = min(args.gate_max, base_gate * args.weak_gate_factor)
        motion_gates[frame_id] = base_gate
        allow_weak = gap <= args.weak_reconnect_gap
        chosen, innovation = choose_tracking_candidate(
            candidates, predicted, strong_gate, weak_gate, args, allow_weak=allow_weak)

        if chosen is None:
            gaps[frame_id] = gap
            if not allow_weak:
                sources[frame_id] = "high_conf_reconnect_only"
            continue

        point = (chosen["x"], chosen["y"])
        measured_velocity = (
            (point[0] - last_position[0]) / max(gap, 1),
            (point[1] - last_position[1]) / max(gap, 1),
        )
        turn_angle = vector_angle_degrees(velocity, measured_velocity)
        measured_speed = math.hypot(*measured_velocity)
        baseline_speed = statistics.median(recent_speeds) if len(recent_speeds) >= 3 else None
        acceleration_jump = (
            baseline_speed is not None
            and baseline_speed >= args.min_speed_for_accel_check
            and measured_speed > baseline_speed * args.max_accel_ratio
        )
        strong = chosen["score"] >= args.high_threshold
        abrupt_change = turn_angle >= args.turn_reset_angle or acceleration_jump
        requires_confirmation = (
            (turn_angle >= args.turn_reset_angle and acceleration_jump)
            or turn_angle >= args.extreme_turn_angle
        )
        if abrupt_change and not strong:
            gaps[frame_id] = gap
            sources[frame_id] = "weak_abrupt_rejected"
            continue
        if requires_confirmation and not abrupt_change_confirmed(
            candidate_rows, frame_id, point, measured_velocity, base_gate, args
        ):
            gaps[frame_id] = gap
            sources[frame_id] = "abrupt_unconfirmed"
            continue
        if strong and abrupt_change:
            velocity = measured_velocity
            source = "high_abrupt_confirmed"
        else:
            alpha = args.velocity_alpha if strong else args.weak_velocity_alpha
            velocity = (
                alpha * measured_velocity[0] + (1.0 - alpha) * velocity[0],
                alpha * measured_velocity[1] + (1.0 - alpha) * velocity[1],
            )
            source = "high_observed" if strong else "weak_motion_recovered"

        tracks[frame_id] = point
        scores[frame_id] = chosen["score"]
        accepted[frame_id] = 1
        sources[frame_id] = source
        innovations[frame_id] = innovation
        shape_quality[frame_id] = chosen.get("shape_quality", 0.0)
        gaps[frame_id] = 0
        last_position = point
        recent_speeds.append(measured_speed)
        gap = 0

    return {
        "tracks": tracks,
        "scores": scores,
        "accepted": accepted,
        "source": sources,
        "raw_tracks": raw_tracks,
        "raw_scores": raw_scores,
        "predicted_tracks": predicted_tracks,
        "innovations": innovations,
        "shape_quality": shape_quality,
        "gaps": gaps,
        "motion_gates": motion_gates,
    }


def remove_temporal_spikes(stable, args):
    """Stage 2: drop a frame that jumps far out and comes straight back."""
    tracks = list(stable["tracks"])
    accepted = list(stable["accepted"])
    sources = list(stable["source"])
    for frame_id in range(1, len(tracks) - 1):
        previous = tracks[frame_id - 1]
        current = tracks[frame_id]
        following = tracks[frame_id + 1]
        if previous is None or current is None or following is None:
            continue
        if (
            distance_xy(previous, following) <= args.spike_return_radius
            and distance_xy(previous, current) >= args.spike_distance
            and distance_xy(current, following) >= args.spike_distance
        ):
            tracks[frame_id] = None
            accepted[frame_id] = 0
            sources[frame_id] = "temporal_spike_removed"
    result = dict(stable)
    result["tracks"] = tracks
    result["accepted"] = accepted
    result["source"] = sources
    return result


def remove_short_fragments(stable, args):
    """Stage 4: drop fragments that are too short, or short and nearly static."""
    tracks = list(stable["tracks"])
    accepted = list(stable["accepted"])
    sources = list(stable["source"])
    index = 0
    while index < len(tracks):
        if tracks[index] is None:
            index += 1
            continue
        start = index
        while index < len(tracks) and tracks[index] is not None:
            index += 1
        end = index
        fragment = tracks[start:end]
        length = end - start
        fragment_radius = max(distance_xy(fragment[0], point) for point in fragment)
        isolated_static_fragment = (
            length <= args.isolated_static_max_length
            and fragment_radius <= args.isolated_static_radius
        )
        static_fragment = (
            length <= args.static_fragment_max_length
            and fragment_radius <= args.static_radius
        )
        if length < args.min_fragment_length or isolated_static_fragment or static_fragment:
            if isolated_static_fragment:
                reason = "isolated_static_removed"
            elif static_fragment:
                reason = "static_fragment_removed"
            else:
                reason = "short_fragment_removed"
            for frame_id in range(start, end):
                tracks[frame_id] = None
                accepted[frame_id] = 0
                sources[frame_id] = reason
    result = dict(stable)
    result["tracks"] = tracks
    result["accepted"] = accepted
    result["source"] = sources
    return result


def adjacent_track_stats(tracks):
    """Diagnostics: how many frame-to-frame jumps exceed 80 / 120 / 180 px."""
    distances = []
    for previous, current in zip(tracks, tracks[1:]):
        if previous is not None and current is not None:
            distances.append(distance_xy(previous, current))
    return {
        "pairs": len(distances),
        "jump80": sum(distance > 80 for distance in distances),
        "jump120": sum(distance > 120 for distance in distances),
        "jump180": sum(distance > 180 for distance in distances),
        "max_jump": max(distances, default=0.0),
    }


def interpolate_guarded(stable, max_gap, max_step):
    """Stage 3: fill holes of at most max_gap frames, only when plausible.

    A hole is filled with a linear ramp between its two neighbours only if the two
    endpoints are within `max_step * (gap + 1) * 1.5`. Anything larger is left empty
    on purpose, so no trajectory is invented where the ball was really lost.
    """
    if max_gap <= 0:
        return stable
    tracks = list(stable["tracks"])
    scores = list(stable["scores"])
    accepted = list(stable["accepted"])
    sources = list(stable["source"])
    n = len(tracks)
    index = 0

    while index < n:
        if tracks[index] is not None:
            index += 1
            continue
        start = index
        while index < n and tracks[index] is None:
            index += 1
        end = index - 1
        gap = end - start + 1
        previous_index = start - 1
        next_index = index
        if gap > max_gap or previous_index < 0 or next_index >= n:
            continue
        previous = tracks[previous_index]
        following = tracks[next_index]
        if previous is None or following is None:
            continue
        if distance_xy(previous, following) > max_step * (gap + 1) * 1.5:
            continue

        span = next_index - previous_index
        previous_score = scores[previous_index] or 0.0
        following_score = scores[next_index] or previous_score
        for frame_id in range(start, end + 1):
            ratio = (frame_id - previous_index) / float(span)
            tracks[frame_id] = (
                previous[0] * (1.0 - ratio) + following[0] * ratio,
                previous[1] * (1.0 - ratio) + following[1] * ratio,
            )
            scores[frame_id] = previous_score * (1.0 - ratio) + following_score * ratio
            accepted[frame_id] = 2
            sources[frame_id] = "guarded_interpolation"

    result = dict(stable)
    result["tracks"] = tracks
    result["scores"] = scores
    result["accepted"] = accepted
    result["source"] = sources
    return result



def load_model(model_path, device, args):
    model = LiteBallNet(
        base_channels=int(args.base_channels),
        use_ca=not args.no_ca,
        ca_reduction=int(args.ca_reduction),
        ca_min_channels=int(args.ca_min_channels),
    )
    state = torch.load(model_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"model load: {model_path}")
    print(f"  parameters={sum(p.numel() for p in model.parameters())} "
          f"missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if missing:
        print("  missing sample:", missing[:5], flush=True)
    if unexpected:
        print("  unexpected sample:", unexpected[:5], flush=True)
    return model.to(device).eval()


def infer_candidate_rows(model, device, args):
    """Run inference over [start_frame, end_frame] and return per-frame candidates."""
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {args.video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = max(0, int(args.start_frame))
    end_frame = int(args.end_frame) if args.end_frame and args.end_frame > 0 else total - 1
    # The network input is a three-frame stack, so two predecessors are needed.
    pre_roll = 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame - pre_roll))

    frames = []
    candidate_rows = []
    batch_inputs = []
    batch_indices = []
    infer_ms = []
    absolute_index = max(0, start_frame - pre_roll)
    read_frames = 0

    while True:
        if absolute_index > end_frame:
            break
        ok, frame = cap.read()
        if not ok:
            break
        read_frames += 1
        frames.append(frame)
        if len(frames) > 3:
            frames.pop(0)

        if absolute_index >= start_frame:
            output_index = absolute_index - start_frame
            while len(candidate_rows) <= output_index:
                candidate_rows.append([])
            if len(frames) < 3:
                candidate_rows[output_index] = []
            else:
                batch_inputs.append(
                    make_input(frames[-1], frames[-2], frames[-3], args.input_width, args.input_height))
                batch_indices.append(output_index)
                if len(batch_inputs) == args.batch_size:
                    results, ms = run_batch(model, device, batch_inputs, width, height, args)
                    infer_ms.append(ms)
                    for idx, result in zip(batch_indices, results):
                        candidate_rows[idx] = result
                    batch_inputs = []
                    batch_indices = []

        if absolute_index % args.print_interval == 0:
            print(f"  infer frame {absolute_index} (window {start_frame}-{end_frame})", flush=True)
        absolute_index += 1

    if batch_inputs:
        results, ms = run_batch(model, device, batch_inputs, width, height, args)
        infer_ms.append(ms)
        for idx, result in zip(batch_indices, results):
            candidate_rows[idx] = result
    cap.release()

    while len(candidate_rows) < max(0, end_frame - start_frame + 1):
        candidate_rows.append([])
    avg_run_ms = float(np.mean(infer_ms)) if infer_ms else 0.0
    raw_detected = sum(1 for row in candidate_rows if row)
    print(f"  inference done: frames={len(candidate_rows)} raw_detected={raw_detected} "
          f"read={read_frames} avg_batch_ms={avg_run_ms:.2f}", flush=True)
    return candidate_rows, width, height, fps, start_frame



def draw_trace(frame, trace_points, raw_point=None):
    """Red fading trail of tracked points, optional orange circle for the raw peak."""
    if raw_point is not None:
        cv2.circle(frame, (int(raw_point[0]), int(raw_point[1])), radius=5,
                   color=(255, 180, 0), thickness=1)
    for age, point in enumerate(reversed(trace_points)):
        if point is None:
            continue
        x, y = point
        thickness = max(2, 10 - age)
        cv2.circle(frame, (int(x), int(y)), radius=0, color=(0, 0, 255), thickness=thickness)
    return frame


def write_outputs(args, stable, width, height, fps, start_frame):
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {args.video}")
    out_fps = float(args.output_fps) if args.output_fps and args.output_fps > 0 else float(fps)
    Path(args.video_out).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.video_out, cv2.VideoWriter_fourcc(*args.codec), out_fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer: {args.video_out}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    csv_handle = None
    csv_writer = None
    if args.csv_out:
        Path(args.csv_out).parent.mkdir(parents=True, exist_ok=True)
        csv_handle = Path(args.csv_out).open("w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_handle)
        csv_writer.writerow([
            "frame", "x", "y", "score", "raw_x", "raw_y", "raw_score",
            "predicted_x", "predicted_y", "innovation", "shape_quality",
            "motion_gate", "accepted", "source", "gap",
        ])

    trace = deque(maxlen=args.trace)

    def pair(value):
        return ("", "") if value is None else (f"{value[0]:.2f}", f"{value[1]:.2f}")

    for local_index in range(len(stable["tracks"])):
        ok, frame = cap.read()
        if not ok:
            break
        point = stable["tracks"][local_index]
        raw = stable["raw_tracks"][local_index]
        predicted = stable["predicted_tracks"][local_index]
        trace.append(point)
        writer.write(draw_trace(frame, trace, raw if args.draw_raw else None))

        if csv_writer is not None:
            x, y = pair(point)
            raw_x, raw_y = pair(raw)
            predicted_x, predicted_y = pair(predicted)
            score = stable["scores"][local_index]
            raw_score = stable["raw_scores"][local_index]
            innovation = stable["innovations"][local_index]
            quality = stable["shape_quality"][local_index]
            motion_gate = stable["motion_gates"][local_index]
            csv_writer.writerow([
                start_frame + local_index, x, y, "" if score is None else f"{score:.6f}",
                raw_x, raw_y, "" if raw_score is None else f"{raw_score:.6f}",
                predicted_x, predicted_y,
                "" if innovation is None else f"{innovation:.3f}",
                "" if quality is None else f"{quality:.4f}",
                "" if motion_gate is None else f"{motion_gate:.2f}",
                stable["accepted"][local_index], stable["source"][local_index],
                stable["gaps"][local_index],
            ])

    writer.release()
    cap.release()
    if csv_handle is not None:
        csv_handle.close()

    source_counts = {}
    for source in stable["source"]:
        source_counts[source] = source_counts.get(source, 0) + 1
    print(f"video_out={args.video_out}")
    print(f"csv_out={args.csv_out or '(none)'}")
    print(f"frames={len(stable['tracks'])} out_fps={out_fps:.2f} "
          f"duration={len(stable['tracks']) / max(out_fps, 1e-6):.2f}s")
    print(f"source_counts={source_counts}")
    print(f"jump_stats={adjacent_track_stats(stable['tracks'])}")

SPORTS = {
    "tennis": {
        "model": "models/tennis_full_seed37.pt",
        "video": "sources/tennis/tennis_clip_f0195-f0475.mp4",
        "start_frame": 0,
        "end_frame": 0,
        "output_fps": 0.0,
        "threshold": 0.30,
        "seed_threshold": 0.85,
        "note": "Nadal-Federer broadcast rally; source 27.48 fps, output keeps the source rate",
    },
    "table_tennis": {
        "model": "models/table_tennis_full_seed37.pt",
        "video": "sources/table_tennis/table_tennis_clip_f7060-f7430.mp4",
        "start_frame": 0,
        "end_frame": 0,
        "output_fps": 30.0,
        "threshold": 0.30,
        "seed_threshold": 0.85,
        "note": "source 120 fps written at 30 fps = 4x slow motion",
    },
    "badminton": {
        "model": "models/badminton_full_seed37.pt",
        "video": "sources/badminton/badminton_clip_f0097-f0280.mp4",
        "start_frame": 0,
        "end_frame": 0,
        "output_fps": 30.0,
        "threshold": 0.30,
        "seed_threshold": 0.85,
        "note": "assembled from the released TrackNetV2 jpg frames (640x360 -> 1280x720)",
    },
}


def build_parser():
    parser = argparse.ArgumentParser(
        description="LiteBallNet demo renderer (single file).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sport", choices=sorted(SPORTS), default=None,
                        help="fill model/video/frame rate from a demo preset")
    parser.add_argument("--list", action="store_true", help="print the presets and exit")
    parser.add_argument("--model", default=None, help="checkpoint (model_best.pt)")
    parser.add_argument("--video", default=None, help="source video")
    parser.add_argument("--video-out", default="out/demo.mp4")
    parser.add_argument("--csv-out", default="out/demo_track.csv",
                        help="per-frame track table; pass \"\" to skip")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=0, help="0 = until the end of the video")
    parser.add_argument("--output-fps", type=float, default=0.0,
                        help="container frame rate; 0 keeps the source rate")

    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--no-ca", action="store_true", help="baseline backbone without Coordinate Attention")
    parser.add_argument("--ca-reduction", type=int, default=32)
    parser.add_argument("--ca-min-channels", type=int, default=8)
    parser.add_argument("--input-height", type=int, default=360)
    parser.add_argument("--input-width", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--print-interval", type=int, default=200)
    parser.add_argument("--codec", default="mp4v")

    # candidate extraction
    parser.add_argument("--threshold", type=float, default=None,
                        help="peak score below which a candidate is dropped "
                             "(preset value 0.30; falls back to 0.35 without --sport)")
    parser.add_argument("--low-threshold", type=float, default=0.35)
    parser.add_argument("--high-threshold", type=float, default=0.88)
    parser.add_argument("--seed-threshold", type=float, default=None,
                        help="score a peak needs before it can seed the tracker "
                             "(preset value 0.85; falls back to 0.96 without --sport)")
    parser.add_argument("--peak-window", type=int, default=9)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--suppress-window", type=int, default=15)
    parser.add_argument("--min-weak-shape", type=float, default=0.05)
    parser.add_argument("--shape-weight", type=float, default=0.05)
    parser.add_argument("--hough-weight", type=float, default=0.00)
    parser.add_argument("--enable-hough-quality", action="store_true")

    # association and motion model
    parser.add_argument("--gate-min", type=float, default=60.0)
    parser.add_argument("--gate-max", type=float, default=220.0)
    parser.add_argument("--speed-window", type=int, default=5)
    parser.add_argument("--speed-gate-factor", type=float, default=2.2)
    parser.add_argument("--gate-margin", type=float, default=25.0)
    parser.add_argument("--strong-gate-factor", type=float, default=1.5)
    parser.add_argument("--weak-gate-factor", type=float, default=0.85)
    parser.add_argument("--motion-weight", type=float, default=0.20)
    parser.add_argument("--velocity-alpha", type=float, default=0.65)
    parser.add_argument("--weak-velocity-alpha", type=float, default=0.35)
    parser.add_argument("--turn-reset-angle", type=float, default=70.0)
    parser.add_argument("--extreme-turn-angle", type=float, default=120.0)
    parser.add_argument("--max-accel-ratio", type=float, default=3.0)
    parser.add_argument("--min-speed-for-accel-check", type=float, default=5.0)
    parser.add_argument("--seed-max-gap", type=int, default=3)
    parser.add_argument("--weak-reconnect-gap", type=int, default=4)
    parser.add_argument("--max-reconnect-gap", type=int, default=8)
    parser.add_argument("--fps-adaptive", action="store_true",
                        help="rescale the motion gates by 30/source_fps (for high-frame-rate sources)")

    # cleanup
    parser.add_argument("--interp-gap", type=int, default=3)
    parser.add_argument("--min-fragment-length", type=int, default=3)
    parser.add_argument("--isolated-static-max-length", type=int, default=3)
    parser.add_argument("--isolated-static-radius", type=float, default=8.0)
    parser.add_argument("--static-fragment-max-length", type=int, default=8)
    parser.add_argument("--static-radius", type=float, default=5.0)
    parser.add_argument("--spike-return-radius", type=float, default=35.0)
    parser.add_argument("--spike-distance", type=float, default=80.0)

    # rendering
    parser.add_argument("--trace", type=int, default=12, help="length of the fading trail")
    parser.add_argument("--draw-raw", action="store_true",
                        help="also draw the raw top-1 peak of the current frame")
    return parser


def apply_sport_preset(args, parser):
    if not args.sport:
        return args
    preset = SPORTS[args.sport]
    base = Path(__file__).resolve().parent
    if args.model is None:
        args.model = str(base / preset["model"])
    if args.video is None:
        args.video = str(base / preset["video"])
    if args.start_frame == 0:
        args.start_frame = preset["start_frame"]
    if args.end_frame == 0:
        args.end_frame = preset["end_frame"]
    if args.output_fps == 0.0:
        args.output_fps = preset["output_fps"]
    # The demo runs used threshold 0.30 / seed 0.85 rather than the parser defaults.
    if args.threshold is None:
        args.threshold = float(preset.get("threshold", 0.35))
    if args.seed_threshold is None:
        args.seed_threshold = float(preset.get("seed_threshold", 0.96))
    print(f"preset '{args.sport}': {preset['note']}")
    print(f"  threshold={args.threshold} seed_threshold={args.seed_threshold} "
          f"output_fps={args.output_fps}")
    return args


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.list:
        for name, preset in SPORTS.items():
            print(f"{name}:\n  model      {preset['model']}\n  video      {preset['video']}\n"
                  f"  threshold  {preset['threshold']} (seed {preset['seed_threshold']})\n"
                  f"  output_fps {preset['output_fps']}\n  note       {preset['note']}")
        return

    args = apply_sport_preset(args, parser)
    if args.threshold is None:
        args.threshold = 0.35
    if args.seed_threshold is None:
        args.seed_threshold = 0.96
    if not args.model or not args.video:
        parser.error("--model and --video are required unless --sport is given")
    if not Path(args.model).exists():
        parser.error(f"model not found: {args.model}")
    if not Path(args.video).exists():
        parser.error(f"video not found: {args.video}")
    if args.threshold > args.low_threshold:
        args.threshold = args.low_threshold

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} video={args.video} model={args.model}")
    model = load_model(args.model, device, args)
    rows, width, height, fps, start_frame = infer_candidate_rows(model, device, args)
    print(f"source: {width}x{height} @ {fps:.2f} fps, window starts at frame {start_frame}", flush=True)

    if args.fps_adaptive:
        args.fps = fps

    stable = stabilize_v37(rows, width, height, fps, args)
    stable = remove_temporal_spikes(stable, args)
    stable = interpolate_guarded(stable, args.interp_gap, args.gate_max)
    stable = remove_short_fragments(stable, args)
    write_outputs(args, stable, width, height, fps, start_frame)


if __name__ == "__main__":
    main()
