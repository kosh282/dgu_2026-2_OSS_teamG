"""프런트와 웹캠 데모가 공유하는 다인 분석 표시 및 모델 생성."""
from types import SimpleNamespace

from engagement_model.config import ModelConfig
from engagement_model.multi_face import MultiFaceGaussianRuntime
from engagement_model.position_tracks import PositionTrackGaussianRuntime


def make_group_runtime(reidentification_enabled=False):
    """두 진입점의 다인 모델과 눈 감김 설정을 일치시킨다."""
    if reidentification_enabled:
        model = MultiFaceGaussianRuntime(eye_fusion=True)
        version = 'multi-gaussian-0.3+face-transform-eye'
    else:
        model = PositionTrackGaussianRuntime(eye_fusion=True)
        version = 'position-gaussian-0.2+face-transform-eye'
    model.model_version = version
    model.latest['model_version'] = version
    return model


def render_group_panel(image, result, face, runtime, diagnostics, overlay_smoother,
                       *, show_overlay, page=0, calibration=None, model_enabled=True):
    """실제 다인 상태·참가자 점수와 얼굴 진단을 공통 패널에 전달한다."""
    # webcam_demo가 웹캠 데모를 간접 참조하므로 순환 import를 피한다.
    from engagement_model.webcam_demo import render_panel

    rows = runtime.participants() if runtime is not None else []
    active = [row for row in rows if row.get('status') != 'missing']
    ready = sum(row.get('calibration_status') == 'ready' for row in active)
    if calibration is None:
        state = ('ready' if active and ready == len(active)
                 else 'collecting' if active else 'waiting')
        calibration = {'status': state, 'progress': ready / len(active) if active else 0}
    latest = dict(runtime.latest) if runtime is not None else {
        'score': None, 'current_direction_score': None, 'orientation_state': 'not_observable',
        'status': 'unavailable', 'reason_codes': ['model_unavailable']}
    sampler = getattr(runtime, 'sampling_diagnostics', None)
    model = SimpleNamespace(latest=latest, config=runtime.config if runtime else ModelConfig.gaussian(),
        display_group=model_enabled, eye_fusion=getattr(runtime, 'eye_fusion', False),
        sampling_diagnostics=sampler if callable(sampler) else lambda: {
            'min_interval_ms': None, 'max_interval_ms': None, 'regular': None})
    quality = diagnostics.observe(face.last_landmarks, image)
    lines = ['LANDMARK STABILITY (diagnostic only)',
             f'MULTI-FACE: {len(active)} tracked, {ready} calibrated']
    for row in active:
        lines.append(f"{row['id']}: {row['status']} total={row.get('score')}")
        lines.append(f"  head={row.get('head_score')} eye={row.get('eye_score')}")
        lines.append(f"  relative P/Y/R: {row.get('relative_angles') or 'unavailable'}")
        if row.get('reason_codes'):
            lines.append(f"  reasons: {', '.join(row['reason_codes'])}")
    lines.extend(f'{key}: {value:.4f}' if value is not None else f'{key}: unavailable'
                 for key, value in quality.items())
    for index, item in enumerate(result.get('features', {}).get('faces', []), 1):
        lines.append(f'FACE {index}:')
        for label, field in [('raw', 'head_angles_degrees'),
                             ('smoothed', 'head_angles_smoothed_degrees'),
                             ('relative', 'head_angles_relative_degrees')]:
            lines.append(f'{label} P/Y/R: {item.get(field) or "unavailable"}')
    display = {**result, 'features': {**result.get('features', {}),
               'head_calibration_status': calibration.get('status', 'waiting'),
               'head_calibration_diagnostics': calibration}}
    return render_panel(image, display, {'gaussian': model}, overlay_smoother=overlay_smoother,
                        face=face, show_overlay=show_overlay, primary='gaussian', debug=True,
                        diagnostics=SimpleNamespace(lines=lines), debug_page=page, live=True)
