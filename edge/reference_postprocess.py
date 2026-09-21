# Auto-extracted without logic changes from LiteBallNet_demo_bundle/render_demo.py.
# Only the PyTorch model execution section is intentionally excluded.
import math
import statistics
from collections import deque

import cv2
import numpy as np

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



