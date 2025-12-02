# ---------------- 마디 수 계산(오류 절대 발생하지 않도록) ----------------

def safe_calculate_measures(bpm, duration):
    # bpm 또는 duration이 None / 0 / NaN / inf / 음수인 경우 → 계산 불가 처리
    if bpm is None:
        return None
    if duration is None:
        return None
    if isinstance(bpm, (int, float)) is False:
        return None
    if isinstance(duration, (int, float)) is False:
        return None
    if bpm <= 0:
        return None
    if duration <= 0:
        return None
    if np.isnan(bpm) or np.isnan(duration):
        return None
    if np.isinf(bpm) or np.isinf(duration):
        return None

    # 모두 정상일 때만 계산
    try:
        measures_value = duration / (60 / bpm)
        return round(measures_value)
    except:
        return None


# ---------------- app.py 안에서 사용 ----------------

measures = safe_calculate_measures(bpm, duration)
