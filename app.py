import numpy as np
import plotly.graph_objects as go
import streamlit as st

import mo_models as mo

st.set_page_config(page_title="HOMO-LUMO / Marcus 전하이동 분석 플랫폼", layout="wide")

st.title("유기 신소재 HOMO-LUMO 밴드갭 및 Marcus 전하이동 분석 플랫폼")

tab1, tab2, tab3 = st.tabs(["단일 분자 밴드갭", "Marcus 전하이동 속도 (Donor-Acceptor)", "온도별 IQE 열화 예측"])

# =====================================================================
# 탭 1: 기존 단일 분자 HOMO-LUMO 밴드갭 분석 (그대로 유지)
# =====================================================================
with tab1:
    col_control, col_diagram, col_optical = st.columns([1, 2, 1.3])

    with col_control:
        st.subheader("오비탈 에너지 조절")
        homo = st.slider("HOMO 에너지 준위 (eV)", -8.0, -3.0, -5.5, 0.05, key="t1_homo")
        lumo = st.slider("LUMO 에너지 준위 (eV)", -4.0, 0.0, -2.5, 0.05, key="t1_lumo")

        if lumo <= homo:
            st.error("LUMO는 HOMO보다 높아야 합니다.")
            st.stop()

        band_gap = mo.calculate_band_gap(homo, lumo)
        wavelength = mo.band_gap_to_wavelength(band_gap)

        st.metric("밴드갭 Eg", f"{band_gap:.2f} eV")
        st.metric("방출 파장 λ", f"{wavelength:.0f} nm" if wavelength != float("inf") else "N/A")

    with col_diagram:
        st.subheader("분자 오비탈(MO) 다이어그램")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[homo, homo], mode="lines",
                                  line=dict(color="#1f77b4", width=5), name="HOMO"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[lumo, lumo], mode="lines",
                                  line=dict(color="#d62728", width=5), name="LUMO"))
        fig.add_annotation(x=1.05, y=homo, text="HOMO", showarrow=False, xanchor="left")
        fig.add_annotation(x=1.05, y=lumo, text="LUMO", showarrow=False, xanchor="left")
        fig.add_annotation(
            x=0.5, y=(homo + lumo) / 2,
            text=f"Eg = {band_gap:.2f} eV",
            showarrow=False, font=dict(size=14, color="gray"),
        )
        fig.update_layout(
            xaxis=dict(visible=False, range=[-0.2, 1.4]),
            yaxis=dict(title="에너지 (eV)", range=[-9, 1]),
            template="plotly_white",
            height=500,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_optical:
        st.subheader("발광 색상")
        color = mo.wavelength_to_rgb(wavelength)
        st.markdown(
            f"""<div style="width:100%;height:150px;background-color:{color};
            border-radius:10px;border:1px solid #ccc;"></div>""",
            unsafe_allow_html=True,
        )
        st.write("")
        suitability = mo.evaluate_display_suitability(band_gap)
        if suitability == "적합":
            st.success(f"이 유기 분자는 디스플레이 소재로 {suitability}합니다.")
        else:
            st.warning(f"이 유기 분자는 디스플레이 소재로 {suitability}합니다.")

# =====================================================================
# 탭 2: Marcus 전하이동 이론 (신규)
# =====================================================================
with tab2:
    col_control, col_graphs = st.columns([1, 2.5])

    with col_control:
        st.subheader("분자 A (Donor) / B (Acceptor)")

        lumo_donor = st.slider("분자 A LUMO 준위 (eV)", -4.0, 0.0, -2.5, 0.05, key="t2_lumo_a")
        lumo_acceptor = st.slider("분자 B LUMO 준위 (eV)", -4.0, 0.0, -3.0, 0.05, key="t2_lumo_b")

        st.subheader("반응 조건")
        reorg_energy = st.slider("재배치 에너지 λ (eV)", 0.05, 2.0, 0.5, 0.05, key="t2_lambda")
        temperature = st.slider("온도 T (K)", 100, 500, 298, 5, key="t2_temp")

        delta_G = mo.calculate_delta_G(lumo_donor, lumo_acceptor)
        k_ET = mo.calculate_marcus_rate(delta_G, reorg_energy, temperature)
        Ea = mo.calculate_activation_energy(delta_G, reorg_energy)

        st.divider()
        st.metric("ΔG (반응 자유에너지)", f"{delta_G:.3f} eV")
        st.metric("활성화 에너지 Ea", f"{Ea:.3f} eV")

    with col_graphs:
        st.subheader("전자 이동 속도 k_ET")
        st.markdown(f"<h1 style='text-align:center;color:#1f77b4;'>{k_ET:.3e} s⁻¹</h1>", unsafe_allow_html=True)

        message = mo.evaluate_charge_transport(k_ET)
        if k_ET >= 1e10:
            st.success(message)
        elif k_ET >= 1e6:
            st.warning(message)
        else:
            st.error(message)

        graph_col1, graph_col2 = st.columns(2)

        with graph_col1:
            st.markdown("**MO 준위 비교 (A vs B)**")
            fig_mo = go.Figure()
            fig_mo.add_trace(go.Scatter(x=[0, 1], y=[lumo_donor, lumo_donor], mode="lines",
                                         line=dict(color="#1f77b4", width=5), name="A: LUMO"))
            fig_mo.add_trace(go.Scatter(x=[2, 3], y=[lumo_acceptor, lumo_acceptor], mode="lines",
                                         line=dict(color="#d62728", width=5), name="B: LUMO"))
            fig_mo.add_annotation(x=0.5, y=lumo_donor, text="분자 A LUMO", showarrow=False, yshift=12)
            fig_mo.add_annotation(x=2.5, y=lumo_acceptor, text="분자 B LUMO", showarrow=False, yshift=12)
            fig_mo.update_layout(
                xaxis=dict(visible=False, range=[-0.5, 3.5]),
                yaxis=dict(title="에너지 (eV)", range=[-4.5, 0.5]),
                template="plotly_white",
                height=420,
                showlegend=False,
            )
            st.plotly_chart(fig_mo, use_container_width=True)

        with graph_col2:
            st.markdown("**Marcus 포텐셜 우물 곡선**")
            q, E_r, E_p = mo.marcus_potential_curves(delta_G, reorg_energy)

            fig_marcus = go.Figure()
            fig_marcus.add_trace(go.Scatter(x=q, y=E_r, mode="lines",
                                             line=dict(color="#1f77b4", width=3), name="반응물 상태 (A+B)"))
            fig_marcus.add_trace(go.Scatter(x=q, y=E_p, mode="lines",
                                             line=dict(color="#d62728", width=3), name="생성물 상태 (A⁺+B⁻)"))

            lam_safe = max(reorg_energy, 1e-6)
            q_cross = (lam_safe - delta_G) / (2 * lam_safe)
            E_cross = lam_safe * q_cross ** 2

            fig_marcus.add_trace(go.Scatter(
                x=[q_cross], y=[E_cross], mode="markers",
                marker=dict(color="black", size=10, symbol="x"),
                name="활성화 에너지 지점",
            ))

            fig_marcus.update_layout(
                xaxis=dict(title="반응 좌표 Q"),
                yaxis=dict(title="에너지 (eV)"),
                template="plotly_white",
                height=420,
                legend=dict(orientation="h", yanchor="bottom", y=-0.3),
            )
            st.plotly_chart(fig_marcus, use_container_width=True)

# =====================================================================
# 탭 3: 온도별 내부 양자 효율(IQE) 열화 예측 (신규)
# =====================================================================
with tab3:
    col_control, col_graph = st.columns([1, 2.5])

    with col_control:
        st.subheader("소재 발광 특성 조절")

        k_r = st.slider(
            "방사 속도 상수 k_r (log₁₀ s⁻¹)", 6.0, 10.0, 8.0, 0.1, key="t3_kr",
            help="전자가 LUMO에서 HOMO로 내려오며 빛을 내는 속도. 값이 클수록 발광 효율이 높음.",
        )
        k_r_value = 10 ** k_r

        activation_energy = st.slider(
            "열화 활성화 에너지 Ea (eV)", 0.05, 1.0, 0.3, 0.01, key="t3_ea",
            help="무방사 열손실 과정이 시작되는 에너지 장벽. 클수록 고온에 더 잘 버팀.",
        )

        st.subheader("주변 온도")
        current_temp = st.slider("현재 온도 T (K)", 1, 500, 300, 1, key="t3_temp")

        current_iqe = mo.calculate_iqe(k_r_value, activation_energy, current_temp)

        st.divider()
        st.metric("현재 온도에서의 IQE", f"{current_iqe:.1f} %")

        diagnosis = mo.diagnose_thermal_degradation(current_iqe)
        if current_iqe < 50:
            st.error(diagnosis)
        else:
            st.success(diagnosis)

    with col_graph:
        st.subheader("온도 vs 내부 양자 효율(IQE) 곡선")

        temps, iqe_curve = mo.iqe_vs_temperature_curve(k_r_value, activation_energy)

        fig_iqe = go.Figure()
        fig_iqe.add_trace(go.Scatter(
            x=temps, y=iqe_curve, mode="lines",
            line=dict(color="#2E6F95", width=3), name="IQE 곡선",
        ))
        fig_iqe.add_trace(go.Scatter(
            x=[current_temp], y=[current_iqe], mode="markers",
            marker=dict(color="#d62728", size=14, symbol="circle"),
            name="현재 작동 상태",
        ))
        fig_iqe.add_hline(y=50, line_dash="dash", line_color="gray",
                           annotation_text="50% 기준선", annotation_position="bottom right")

        fig_iqe.update_layout(
            xaxis=dict(title="온도 T (K)", range=[0, 500]),
            yaxis=dict(title="내부 양자 효율 IQE (%)", range=[0, 105]),
            template="plotly_white",
            height=550,
        )
        st.plotly_chart(fig_iqe, use_container_width=True)

# 확장 지점 (실제 화합물 DB 연동 등):
# mo_models.py 에 함수 추가 후 새로운 탭(tab4)을 만들어 이어붙이면 된다.
# 기존 tab1, tab2, tab3 코드는 전혀 건드릴 필요 없다.
