"""개발자 프리뷰와 웹캠 데모가 공유하는 영상/특징 표시 함수."""

import cv2
import numpy as np

from .preview import draw_face_postprocessing, draw_landmarks


CALIBRATION_REASON_TEXT = {
    "no_face_detected": "show one clear face",
    "multiple_faces_detected": "keep exactly one face in view",
    "face_quality_not_observable": "face quality metadata is missing or unavailable",
    "head_angles_unavailable": "head angles could not be calculated",
    "calibration_frame_gap": "capture gap exceeded 1 second; hold still and retry",
    "calibration_timestamp_not_increasing": "capture clock did not advance; restart calibration",
    "head_pose_unstable": "head moved during calibration; face forward and hold still",
}
MODEL_REASON_TEXT = {
    "face_not_found": "no face detected",
    "multiple_faces_unsupported": "the score demo accepts one face",
    "baseline_unavailable": "press C and finish calibration",
    "landmark_low_quality": "face landmark quality is not valid",
    "relative_angles_unavailable": "relative head angles are missing",
    "looking_down_deferred": "downward gaze is deferred by current rules",
    "looking_down_grace": "downward posture is in the 5-second grace period",
    "looking_down_score_decay": "downward posture score is gradually decreasing",
    "orientation_ambiguous": "head direction is between configured ranges",
    "away_grace_without_previous_score": "away grace is active without a prior score",
    "observation_gap": "continuous observation was interrupted; timer restarted",
    "tracking_key_changed": "temporary face track changed; timer restarted",
}


def _number(value, digits=3):
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "n/a"


def render_feature_panel(result, face_page=0, width=1200, model_result=None):
    """같은 분석 결과를 읽기 쉬운 텍스트 패널로 표시한다.

    참여도 모델은 아직 연결되지 않았으므로 실제 점수를 만들지 않고
    측정 불가 상태를 명시한다.
    """
    features = result.get("features", {})
    faces = features.get("faces", [])
    bodies = features.get("upper_bodies", [])
    page_size = 2
    page_count = max(1, (len(faces) + page_size - 1) // page_size)
    face_page = min(max(0, face_page), page_count - 1)
    start = face_page * page_size
    shown_faces = faces[start:start + page_size]

    sampling = features.get("sampling", {})
    source = sampling.get("source_size", {})
    analysis = sampling.get("analysis_size", {})
    calibration = features.get("head_calibration_diagnostics", {})
    calibration_reason = calibration.get("last_reason_code") or "none"
    calibration_samples = calibration.get("sample_count", 0)
    calibration_required = calibration.get("required_samples", 0)
    calibration_elapsed = calibration.get("elapsed_ms", 0) / 1000.0
    calibration_duration = calibration.get("required_duration_ms", 0) / 1000.0
    calibration_progress = calibration.get("progress", 0.0) * 100
    if calibration_reason == "none":
        calibration_detail = (
            "press C to start; face forward and hold still"
            if calibration.get("status", "waiting") == "waiting" else "no current calibration issue"
        )
    else:
        calibration_detail = CALIBRATION_REASON_TEXT.get(calibration_reason, calibration_reason)
    lines = [
        "Developer preview: same captured frame supplied to the analyzers",
        (f"Capture: {result.get('start_ms', 'n/a')} ms | faces: {len(faces)} | "
         f"bodies: {len(bodies)} | quality: {result.get('quality', {}).get('status', 'n/a')}"),
        (f"Calibration: {features.get('head_calibration_status', 'waiting')} | "
         f"samples={calibration_samples}/{calibration_required} | "
         f"time={calibration_elapsed:.1f}/{calibration_duration:.1f}s | "
         f"progress={calibration_progress:.0f}% | last issue={calibration_reason}"),
        f"Calibration help: {calibration_detail}",
        (f"Postprocess: {features.get('face_postprocessing_version', 'n/a')} | "
         "calibration: press C, face forward, keep still for 3 seconds"),
        (f"Analysis Hz face target/actual: {_number(sampling.get('target_face_hz'), 0)}/"
         f"{_number(sampling.get('actual_face_hz'), 1)} | pose: "
         f"{_number(sampling.get('target_pose_hz'), 0)}/"
         f"{_number(sampling.get('actual_pose_hz'), 1)}"),
        (f"Frame size: {source.get('width', 'n/a')}x{source.get('height', 'n/a')} -> "
         f"{analysis.get('width', 'n/a')}x{analysis.get('height', 'n/a')} | "
         f"pose observation age: {_number(sampling.get('pose_observation_age_ms'), 0)} ms"),
        (f"Faces {start + 1}-{min(start + page_size, len(faces))} / {len(faces)} "
         f"(page {face_page + 1}/{page_count}) | V overlay, M mesh, [ ] page, C calibrate, Q close"),
    ]

    if model_result is None:
        lines.append("Model: not connected | participation score: unavailable")
    else:
        model_score = model_result.get("score")
        display_score = model_result.get("display_score")
        display_state = model_result.get("display_state", "unavailable")
        display_age = model_result.get("display_age_ms")
        if isinstance(model_score, (int, float)):
            score_text = f"{model_score:.1f}/100 (current)"
        elif display_state == "held" and isinstance(display_score, (int, float)):
            age_text = f", {display_age} ms old" if isinstance(display_age, (int, float)) else ""
            score_text = f"{display_score:.1f}/100 (HELD{age_text}; current score unavailable)"
        elif display_state == "expired":
            age_text = f" after {display_age} ms" if isinstance(display_age, (int, float)) else ""
            score_text = f"unavailable (previous display expired{age_text})"
        else:
            score_text = "unavailable"
        reason_text = ", ".join(model_result.get("reason_codes", [])) or "none"
        reason_help = "; ".join(
            MODEL_REASON_TEXT.get(code, code) for code in model_result.get("reason_codes", [])
        ) or "no score blocking reason"
        model_angles = model_result.get("angles") or {}
        lines.extend([
            (f"Model: {model_result.get('model_id', 'n/a')} "
             f"{model_result.get('model_version', 'n/a')} | config {model_result.get('config_version', 'n/a')}"),
            (f"Rules: front pitch +/-{model_result.get('thresholds', {}).get('front_pitch_degrees', 10)} deg; "
             f"down start/exit pitch >= {model_result.get('thresholds', {}).get('down_pitch_degrees', 15)}/"
             f">{model_result.get('thresholds', {}).get('down_exit_pitch_degrees', 10)} deg; "
             f"up away <= -{model_result.get('thresholds', {}).get('away_up_degrees', 20)} deg; "
             f"yaw away >= {model_result.get('thresholds', {}).get('away_yaw_degrees', 30)} deg; "
             f"yaw deadband/soft scale "
             f"{model_result.get('thresholds', {}).get('yaw_neutral_deadband_degrees', 2)}/"
             f"{model_result.get('thresholds', {}).get('yaw_small_angle_scale', 0.75)} through "
             f"{model_result.get('thresholds', {}).get('yaw_full_scale_start_degrees', 15)} deg; "
             f"away grace {model_result.get('thresholds', {}).get('away_grace_seconds', 3)} s; "
             f"down grace/decay {model_result.get('thresholds', {}).get('down_grace_seconds', 5)}/"
             f"{model_result.get('thresholds', {}).get('down_decay_seconds', 10)} s, max penalty "
             f"{model_result.get('thresholds', {}).get('down_penalty', 20)}"),
            (f"Model output: {score_text} | display_state={display_state} "
             f"| state={model_result.get('orientation_state', 'n/a')} "
             f"| status={model_result.get('status', 'n/a')} | reasons={reason_text}"),
            f"Model diagnostic: {reason_help}",
            ("Model relative P/Y/R (yaw adjusted): " + ", ".join(
                _number(model_angles.get(axis), 1) for axis in ("pitch", "yaw", "roll")) + " deg | "
             f"source yaw={_number((model_result.get('source_angles') or {}).get('yaw'), 1)} deg | "
             f"front={_number(model_result.get('front_seconds'), 2)} s "
             f"away={_number(model_result.get('away_seconds'), 2)} s "
             f"down={_number(model_result.get('looking_down_seconds'), 2)} s "
             f"previous stable={_number(model_result.get('previous_stable_score'), 1)}"),
        ])

    for index, face in enumerate(shown_faces, start=start + 1):
        raw = face.get("head_angles_degrees") or {}
        smooth = face.get("head_angles_smoothed_degrees") or {}
        relative = face.get("head_angles_relative_degrees") or {}
        movement = face.get("head_angle_change_candidate") or {}
        delta = movement.get("delta_degrees") or {}
        blink = face.get("blink_metrics") or {}
        expressions = face.get("expression_blendshapes") or {}
        position = face.get("head_position") or {}
        face_quality = face.get("quality") or {}
        face_quality_reasons = ",".join(face_quality.get("reason_codes", [])) or "none"
        track = face.get("transient_track_key", "?")
        lines.extend([
            (f"Face {index} / temporary T{track} | tracking: "
             f"{face.get('tracking_quality', 'n/a')} | quality: "
             f"{face_quality.get('status', 'n/a')} ({face_quality_reasons})"),
            (f"  Model input: raw head angles={'available' if face.get('head_angles_degrees') else 'missing'} | "
             f"relative angles={'available' if relative else 'missing'} | "
             f"source={face.get('head_angle_model_input_source', 'n/a')}"),
            (f"  Eyes: openness={_number(face.get('eye_openness'))}, "
             f"EAR L/R={_number(face.get('eye_aspect_ratio_left'))}/"
             f"{_number(face.get('eye_aspect_ratio_right'))}"),
            "  Raw pitch/yaw/roll: " + ", ".join(_number(raw.get(k), 1) for k in ("pitch", "yaw", "roll")) + " deg",
            "  Smoothed pitch/yaw/roll: " + ", ".join(_number(smooth.get(k), 1) for k in ("pitch", "yaw", "roll")) + " deg",
            "  Vision relative P/Y/R (pre-model yaw adjustment): " + ", ".join(
                _number(relative.get(k), 1) for k in ("pitch", "yaw", "roll")) + " deg",
            "  Raw relative pitch/yaw/roll (comparison): " + ", ".join(
                _number((face.get("head_angles_relative_raw_degrees") or {}).get(k), 1)
                for k in ("pitch", "yaw", "roll")) + " deg",
            ("  Angle change P/Y/R: " + ", ".join(_number(delta.get(k), 2) for k in ("pitch", "yaw", "roll"))
             + f" deg / {movement.get('interval_ms', 'n/a')} ms" if movement
             else "  Angle change: waiting for another observation"),
            (f"  Blink count: total={blink.get('total_count', 0)}, "
             f"left={blink.get('left_count', 0)}, right={blink.get('right_count', 0)} | "
             f"recent={blink.get('count_in_window', 0)}/{blink.get('count_window_ms', 0)} ms | "
             f"rate={_number(blink.get('blinks_per_minute'), 1)}/min"),
            f"  Nose position (normalized): x={_number(position.get('x'))}, y={_number(position.get('y'))}",
            "  Expression blendshapes (not emotion/engagement): " +
            (", ".join(f"{name}={_number(value, 2)}" for name, value in expressions.items()) or "unavailable"),
        ])

    if not faces:
        lines.append("No face detected in the selected window.")

    for index, body in enumerate(bodies, start=1):
        center = body.get("shoulder_center") or {}
        lines.append(
            f"Body {index}: shoulder center x={_number(center.get('x'))}, y={_number(center.get('y'))} | "
            f"visibility={_number(body.get('shoulder_visibility'))} | "
            f"quality={body.get('quality', {}).get('status', 'n/a')}"
        )

    for label, key in (("Head movement candidate", "head_movement_candidate"),
                       ("Upper-body movement candidate", "upper_body_movement_candidate")):
        motion = features.get(key)
        lines.append(
            f"{label}: distance={_number(motion.get('distance_normalized'))} / "
            f"{motion.get('interval_ms', 'n/a')} ms" if motion else f"{label}: unavailable"
        )

    line_height = 23
    panel = np.zeros((max(190, 34 + len(lines) * line_height), width, 3), dtype=np.uint8)
    for row, line in enumerate(lines):
        # 긴 수치 문자열이 패널 밖으로 밀리지 않도록 글자 크기를 고정하고 한 줄로 표시한다.
        cv2.putText(panel, line, (14, 25 + row * line_height), cv2.FONT_HERSHEY_SIMPLEX,
                    0.43, (235, 235, 235), 1, cv2.LINE_AA)
    return panel


def render_video_frame(frame, result, face_landmarks, pose_landmarks, overlay_smoother,
                       *, show_overlay=True, show_tessellation=False):
    """두 화면에서 공통으로 쓰는 메시/자세/후처리 오버레이를 만든다."""
    if frame is None:
        return None
    output = frame.copy()
    features = result.get("features", {})
    faces = features.get("faces", [])
    if show_overlay:
        smoothed_landmarks = overlay_smoother.smooth(
            face_landmarks, faces, result.get("start_ms", 0)
        )
        output = draw_landmarks(
            output, smoothed_landmarks, pose_landmarks,
            face_mesh=True, tessellation=show_tessellation, show_points=False,
        )
    return draw_face_postprocessing(output, faces)
