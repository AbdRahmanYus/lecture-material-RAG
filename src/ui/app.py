"""
app.py

Streamlit UI for the Control Engineering RAG Study Assistant.
Provides two tabs:
  1. Ask  — question answering via the RAG pipeline
  2. Solve — symbolic transfer function analysis via SymPy
"""

import streamlit as st
from sympy import symbols

# Local imports
from src.rag.query_engine import answer_question
from src.rag.formula_solver import (
    find_poles,
    check_stability,
    routh_hurwitz,
    to_state_space,
)

# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Control Engineering Assistant",
    page_icon="⚙️",
    layout="centered",
)

s = symbols('s')

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("⚙️ Control Engineering Study Assistant")
st.caption("Powered by your lecture notes · RAG + SymPy")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_ask, tab_solve = st.tabs(["💬 Ask", "🔢 Solve"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ASK
# ══════════════════════════════════════════════════════════════════════════════
with tab_ask:
    st.subheader("Ask a question from your lecture notes")
    st.write(
        "Type any Control Engineering question. "
        "The assistant retrieves relevant lecture note chunks "
        "and uses an LLM to synthesise an answer."
    )

    question = st.text_input(
        label="Your question",
        placeholder="e.g. What is the Routh-Hurwitz stability criterion?",
    )

    n_chunks = st.slider(
        label="Number of lecture note chunks to retrieve",
        min_value=1,
        max_value=8,
        value=3,
        help="More chunks = more context for the LLM, but slower response.",
    )

    if st.button("Ask", type="primary", key="ask_button"):
        if not question.strip():
            st.warning("Please enter a question first.")
        else:
            with st.spinner("Searching lecture notes and generating answer..."):
                try:
                    result = answer_question(question, n_results=n_chunks)

                    st.success("Answer ready")

                    st.markdown("### Answer")
                    st.write(result["answer"])

                    st.markdown("### Sources")
                    for source, page in result["sources"]:
                        st.markdown(f"- `{source}` — page {page}")

                except RuntimeError as e:
                    st.error(f"Error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SOLVE
# ══════════════════════════════════════════════════════════════════════════════
with tab_solve:
    st.subheader("Transfer Function Analyser")
    st.write(
        "Enter the coefficients of your transfer function denominator "
        "and (optionally) numerator. Coefficients go in **descending order**."
    )

    st.markdown("**Example:** For `s³ + 2s² + 3s + 8`, enter `1, 2, 3, 8`")

    col1, col2 = st.columns(2)

    with col1:
        denom_input = st.text_input(
            label="Denominator coefficients",
            placeholder="e.g. 1, 2, 3, 8",
            key="denom",
        )

    with col2:
        numer_input = st.text_input(
            label="Numerator coefficients (optional)",
            placeholder="e.g. 1  (just a gain)",
            key="numer",
        )

    if st.button("Analyse", type="primary", key="solve_button"):

        # ── Parse inputs ───────────────────────────────────────────────────────
        try:
            denom_coeffs = [float(x.strip()) for x in denom_input.split(",")]
        except ValueError:
            st.error("Denominator: please enter numbers separated by commas.")
            st.stop()

        if numer_input.strip():
            try:
                numer_coeffs = [float(x.strip()) for x in numer_input.split(",")]
            except ValueError:
                st.error("Numerator: please enter numbers separated by commas.")
                st.stop()
        else:
            numer_coeffs = [1.0]  # default gain of 1

        # ── Run analysis ───────────────────────────────────────────────────────
        with st.spinner("Computing..."):

            # 1 — Poles
            try:
                denom_expr = sum(
                    coeff * s**(len(denom_coeffs) - 1 - i)
                    for i, coeff in enumerate(denom_coeffs)
                )
                poles = find_poles(denom_expr)
                stability = check_stability(poles)
            except Exception as e:
                st.error(f"Pole computation failed: {e}")
                st.stop()

            # 2 — Routh array
            try:
                routh = routh_hurwitz(denom_coeffs)
            except Exception as e:
                st.error(f"Routh-Hurwitz failed: {e}")
                st.stop()

            # 3 — State space
            try:
                ss = to_state_space(numer_coeffs, denom_coeffs)
            except Exception as e:
                st.error(f"State space conversion failed: {e}")
                st.stop()

        # ── Display results ────────────────────────────────────────────────────

        # Stability verdict banner
        if stability["stable"]:
            st.success(f"✅ {stability['verdict']}")
        else:
            st.error(f"❌ {stability['verdict']}")

        # Poles table
        st.markdown("### Poles")
        for entry in stability["poles"]:
            marker = "🟢" if entry["stable"] else "🔴"
            st.markdown(
                f"{marker} `{entry['pole']}` — Re = `{entry['real_part']:.4f}`"
            )

        # Routh array
        st.markdown("### Routh-Hurwitz Array")
        st.markdown(f"Sign changes in first column: **{routh['sign_changes']}**")

        routh_display = []
        for i, row in enumerate(routh["array"]):
            order_label = f"s^{len(routh['array']) - 1 - i}"
            routh_display.append(
                [order_label] + [f"{float(x):.4f}" for x in row]
            )

        st.table(routh_display)

        # State space matrices
        st.markdown("### State Space (Controllable Canonical Form)")

        ss_col1, ss_col2 = st.columns(2)
        with ss_col1:
            st.markdown("**A matrix**")
            st.code(str(ss["A"]))
            st.markdown("**B matrix**")
            st.code(str(ss["B"]))
        with ss_col2:
            st.markdown("**C matrix**")
            st.code(str(ss["C"]))
            st.markdown("**D matrix**")
            st.code(str(ss["D"]))