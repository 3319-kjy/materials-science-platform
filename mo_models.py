
import numpy as np

PLANCK_CONSTANT_EV_S = 4.135667696e-15  # eV·s
SPEED_OF_LIGHT_NM_S = 2.99792458e17     # nm/s (c를 nm/s 단위로 변환)


def calculate_band_gap(homo_level_eV: float, lumo_level_eV: float) -> float:
    return lumo_level_eV - homo_level_eV


def band_gap_to_wavelength(band_gap_eV: float) -> float:
    if band_gap_eV <= 0:
        return float("inf")
    return (PLANCK_CONSTANT_EV_S * SPEED_OF_LIGHT_NM_S) / band_gap_eV


def wavelength_to_rgb(wavelength_nm: float) -> str:
    """가시광선 파장(380~750nm)을 근사 RGB 색상 문자열로 변환. 범위 밖이면 회색."""
    w = wavelength_nm

    if w < 380 or w > 750:
        return "rgb(60,60,60)"  # 가시광선 밖 (적외선/자외선) -> 회색으로 표시

    if 380 <= w < 440:
        r, g, b = -(w - 440) / (440 - 380), 0.0, 1.0
    elif 440 <= w < 490:
        r, g, b = 0.0, (w - 440) / (490 - 440), 1.0
    elif 490 <= w < 510:
        r, g, b = 0.0, 1.0, -(w - 510) / (510 - 490)
    elif 510 <= w < 580:
        r, g, b = (w - 510) / (580 - 510), 1.0, 0.0
    elif 580 <= w < 645:
        r, g, b = 1.0, -(w - 645) / (645 - 580), 0.0
    else:  # 645 <= w <= 750
        r, g, b = 1.0, 0.0, 0.0

    # 가장자리 파장에서 밝기 감쇠
    if 380 <= w < 420:
        factor = 0.3 + 0.7 * (w - 380) / (420 - 380)
    elif 700 <= w <= 750:
        factor = 0.3 + 0.7 * (750 - w) / (750 - 700)
    else:
        factor = 1.0

    r, g, b = [int(255 * (c * factor)) for c in (r, g, b)]
    return f"rgb({r},{g},{b})"


def evaluate_display_suitability(band_gap_eV: float) -> str:
    """OLED 디스플레이 소재로서의 적합성을 밴드갭 기준으로 간단 평가."""
    if 1.8 <= band_gap_eV <= 3.1:
        return "적합"
    return "부적합"


# ---------------------------------------------------------------
# Marcus 전하 이동 이론 (Marcus Electron Transfer Theory)
# ---------------------------------------------------------------

R_GAS_CONSTANT_EV = 8.617333262e-5  # eV / (mol·K) 단위계와 맞춘 기체상수 (eV 기준)
PRE_EXPONENTIAL_FACTOR_A = 1e13     # 주변 인자 A (s^-1), 전형적인 분자 진동 빈도 수준으로 고정


def calculate_delta_G(lumo_donor_eV: float, lumo_acceptor_eV: float) -> float:
    """공여체(A)와 수용체(B)의 LUMO 준위 차이를 반응 깁스 자유에너지 변화량으로 정의."""
    return lumo_acceptor_eV - lumo_donor_eV


def calculate_marcus_rate(delta_G_eV: float, reorganization_energy_eV: float, temperature_K: float) -> float:
    """
    Marcus 전하 이동 속도 계산.
    k_ET = A * exp( -(ΔG + λ)^2 / (4 λ R T) )
    """
    lam = reorganization_energy_eV
    if lam <= 0:
        return 0.0

    exponent = -((delta_G_eV + lam) ** 2) / (4 * lam * R_GAS_CONSTANT_EV * temperature_K)
    return PRE_EXPONENTIAL_FACTOR_A * np.exp(exponent)


def calculate_activation_energy(delta_G_eV: float, reorganization_energy_eV: float) -> float:
    """Marcus 이론의 활성화 에너지 장벽: Ea = (ΔG + λ)^2 / (4λ)"""
    lam = reorganization_energy_eV
    if lam <= 0:
        return 0.0
    return ((delta_G_eV + lam) ** 2) / (4 * lam)


def marcus_potential_curves(
    delta_G_eV: float,
    reorganization_energy_eV: float,
    n_points: int = 300,
    q_range: float = 2.5,
):
    """
    반응좌표(Q)에 대한 두 개의 포텐셜 우물(반응물 상태 R, 생성물 상태 P) 포물선을 생성.

    반응물 포물선의 최소값은 Q=0, 에너지=0에 위치.
    생성물 포물선은 Q=1 지점에 최소를 가지며, 최소 에너지는 ΔG.
    두 포물선의 곡률은 재배치 에너지 λ로 결정 (E = λ * (Q - Q0)^2 + E0).

    Returns
    -------
    q, E_reactant, E_product : np.ndarray
    """
    lam = max(reorganization_energy_eV, 1e-6)
    q = np.linspace(-q_range, 1 + q_range, n_points)

    E_reactant = lam * q ** 2
    E_product = lam * (q - 1) ** 2 + delta_G_eV

    return q, E_reactant, E_product


def evaluate_charge_transport(k_ET: float) -> str:
    """전자 이동 속도 기준 소재 적합성 평가 (경험적 임계값)."""
    if k_ET >= 1e10:
        return "전하 수송 능력이 우수한 디스플레이/태양전지 소재 후보군입니다."
    elif k_ET >= 1e6:
        return "전하 수송 능력이 보통 수준으로, 추가 최적화가 필요한 소재입니다."
    else:
        return "전하 수송 속도가 낮아 디스플레이/태양전지 소재로는 부적합할 가능성이 높습니다."


# ---------------------------------------------------------------
# 온도 상승에 따른 디스플레이 열화 및 내부 양자 효율(IQE) 예측
# ---------------------------------------------------------------

PRE_EXPONENTIAL_FACTOR_NR = 1e14  # 무방사 과정의 주변 인자 A_nr (s^-1), 고정 상수


def calculate_k_nr(activation_energy_eV: float, temperature_K: float) -> float:
    """
    아레니우스 법칙에 따른 무방사 속도 상수 계산.
    k_nr = A_nr * exp(-Ea / R T)
    """
    if temperature_K <= 0:
        return 0.0
    return PRE_EXPONENTIAL_FACTOR_NR * np.exp(-activation_energy_eV / (R_GAS_CONSTANT_EV * temperature_K))


def calculate_iqe(k_r: float, activation_energy_eV: float, temperature_K: float) -> float:
    """
    내부 양자 효율(IQE) 계산.
    IQE (%) = k_r / (k_r + k_nr) * 100
    """
    k_nr = calculate_k_nr(activation_energy_eV, temperature_K)
    if (k_r + k_nr) <= 0:
        return 0.0
    return (k_r / (k_r + k_nr)) * 100


def iqe_vs_temperature_curve(k_r: float, activation_energy_eV: float, temp_min: float = 0.1, temp_max: float = 500, n_points: int = 300):
    """온도(0~500K) 구간에 대한 IQE 곡선 좌표를 생성."""
    temperatures = np.linspace(temp_min, temp_max, n_points)
    iqe_values = np.array([calculate_iqe(k_r, activation_energy_eV, T) for T in temperatures])
    return temperatures, iqe_values


def diagnose_thermal_degradation(iqe_percent: float) -> str:
    """IQE 값 기준 열적 열화 진단 메시지."""
    if iqe_percent < 50:
        return "경고: 고온으로 인한 열적 열화(Thermal Degradation) 발생 위험이 높습니다. 방열 설계가 필요합니다."
    return "현재 온도에서 발광 효율이 안정적으로 유지되고 있습니다."


# 새 모델(실제 화합물 DB 연동 등) 추가 위치
