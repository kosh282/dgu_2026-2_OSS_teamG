"""수업 화면의 강한 수평·수직 경계에서 일시적 분석 영역을 찾는다."""


def _line_positions(strength, threshold, limit=12):
    """연속된 경계 픽셀을 하나의 선으로 합친다."""
    import numpy as np
    indices = np.flatnonzero(strength >= threshold)
    groups = []
    for index in indices:
        if not groups or index > groups[-1][-1] + 1:
            groups.append([int(index)])
        else:
            groups[-1].append(int(index))
    peaks = [(max(group, key=lambda i: strength[i]), max(strength[i] for i in group))
             for group in groups]
    return sorted(i + 1 for i, _ in sorted(peaks, key=lambda item: item[1], reverse=True)[:limit])


def detect_layout_regions(image, max_regions=12):
    """경계가 뚜렷한 후보만 반환한다. 좌표는 입력 프레임 기준이며 ID가 아니다."""
    import cv2
    import numpy as np

    height, width = image.shape[:2]
    if width < 64 or height < 64 or max_regions <= 0:
        return []
    scale = min(1.0, 960 / width, 540 / height)
    small = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))),
                       interpolation=cv2.INTER_AREA) if scale < 1 else image
    sh, sw = small.shape[:2]
    pixels = cv2.GaussianBlur(small, (3, 3), 0).astype(np.int16)
    regions = []

    def split_bands(oriented, transposed):
        oh, ow = oriented.shape[:2]
        horizontal = (np.abs(oriented[2:] - oriented[:-2]).mean(axis=2) > 15).mean(axis=1)
        ys = [0] + _line_positions(horizontal, 0.45) + [oh]
        for top, bottom in zip(ys, ys[1:]):
            if bottom - top < oh * 0.12 or bottom - top > oh * 0.8:
                continue
            band = oriented[top:bottom]
            vertical = (np.abs(band[:, 2:] - band[:, :-2]).mean(axis=2) > 15).mean(axis=0)
            xs = [0] + _line_positions(vertical, 0.60) + [ow]
            for left, right in zip(xs, xs[1:]):
                x0, y0, x1, y1 = ((top, left, bottom, right) if transposed
                                  else (left, top, right, bottom))
                band_width, band_height = x1 - x0, y1 - y0
                area = band_width * band_height
                if (band_width < sw * 0.12 or band_height < sh * 0.12
                        or area < sw * sh * 0.025 or area > sw * sh * 0.65
                        or not 0.45 <= band_width / band_height <= 4):
                    continue
                # 균일한 여백은 타일 후보로 쓰지 않는다.
                if float(np.std(small[y0:y1, x0:x1])) < 6:
                    continue
                pad_x, pad_y = max(2, round(band_width * 0.02)), max(2, round(band_height * 0.02))
                region = (max(0, x0 - pad_x), max(0, y0 - pad_y),
                          min(sw, x1 + pad_x), min(sh, y1 + pad_y))
                regions.append(region)

    split_bands(pixels, False)
    split_bands(np.transpose(pixels, (1, 0, 2)), True)
    regions.sort(key=lambda r: (-(r[2] - r[0]) * (r[3] - r[1]), r[1], r[0]))
    unique = []
    for region in regions:
        x0, y0, x1, y1 = region
        area = (x1 - x0) * (y1 - y0)
        def same_region(other):
            intersection = (max(0, min(x1, other[2]) - max(x0, other[0]))
                            * max(0, min(y1, other[3]) - max(y0, other[1])))
            union = area + (other[2] - other[0]) * (other[3] - other[1]) - intersection
            return union > 0 and intersection / union >= 0.8
        if any(same_region(b) for b in unique):
            continue
        unique.append(region)
    return [(round(x0 * width / sw), round(y0 * height / sh),
             round(x1 * width / sw), round(y1 * height / sh))
            for x0, y0, x1, y1 in unique[:max_regions]]
