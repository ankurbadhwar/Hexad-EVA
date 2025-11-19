import streamlit as st
import pandas as pd
import requests
import re
import json
import random

# ---------------------------
# Basic page setup
# ---------------------------
st.set_page_config(page_title="EVA — AI That Builds AIs", layout="wide")


def sanitize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


# ---------------------------
# Title & Intro
# ---------------------------
st.title("EVA — The AI That Builds Other AIs")
st.markdown(
    """
EVA automatically **generates, evaluates, evolves, and finalizes** the best AI agent for a given task.  
This UI is fully connected to the **FastAPI + Gemini backend**, which:

- Generates multiple agent blueprints  
- Builds & runs them using **Gemini**  
- Evaluates them with deterministic metrics  
- Mutates the best one and re-evaluates  
- Returns the **best-performing agent + Python code**  
"""
)

# ---------------------------
# Sidebar: Settings
# ---------------------------
with st.sidebar:
    st.header("EVA Settings")

    backend_url = st.text_input(
        "Backend URL",
        "http://localhost:8000",
        help="FastAPI backend base URL. Make sure app.py is running with uvicorn.",
    )

    n_agents = st.number_input(
        "Number of agent blueprints", min_value=2, max_value=6, value=3, step=1
    )

    style_hint = st.text_input(
        "Style hint (optional)",
        placeholder="e.g., formal, layman, bullet-style",
    )

    seed = st.number_input(
        "Random seed (for backend if used)", min_value=0, value=42, step=1
    )

    st.caption("Backend must be running for this UI to work.")


# ---------------------------
# Main layout: Task & Inputs
# ---------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Task Input")
    task = st.text_area(
        "Describe the user task for which EVA should build an agent",
        height=180,
        placeholder="E.g., Summarize this lecture in 3 bullets.",
    )

    st.subheader("Optional notes / context")
    extra_notes = st.text_area(
        "Optional: Notes you want to keep for yourself (not sent to backend)",
        height=100,
        placeholder="Hackathon notes, judge comments, etc.",
    )

with col2:
    st.subheader("Run EVA")
    st.write("This will call the **backend** and run the full EVA pipeline.")
    run_backend = st.button("🚀 Run EVA (Backend)")

    st.markdown("---")
    st.subheader("Last Run Info")
    if "last_status" in st.session_state:
        st.write(st.session_state["last_status"])
    else:
        st.caption("No runs yet.")


# ---------------------------
# Session state for backend job
# ---------------------------
if "job" not in st.session_state:
    st.session_state["job"] = None


# ---------------------------
# Call backend when button clicked
# ---------------------------
if run_backend:
    if not task.strip():
        st.warning("Please provide a task before running EVA.")
    elif not backend_url.strip():
        st.warning("Please provide a valid backend URL.")
    else:
        payload = {
            "user_id": "team_hexad",
            "problem_text": task,
            "domain": "lecture",
            "target": {"format": "3_bullets", "max_words": 60},
            "n_agents": int(n_agents),
            "style_hint": style_hint,
            "seed": int(seed),
        }

        # Optional: set seed for consistency if backend uses it
        random.seed(int(seed))

        with st.spinner("Running EVA pipeline on backend (Gemini + evolution)..."):
            try:
                resp = requests.post(f"{backend_url.rstrip('/')}/request", json=payload)
                resp.raise_for_status()
                job = resp.json()
                st.session_state["job"] = job
                st.session_state["last_status"] = "✅ Last run: success"
                st.success("EVA pipeline complete!")
            except Exception as e:
                st.session_state["last_status"] = f"❌ Last run failed: {e}"
                st.error(f"Backend error: {e}")
                job = None


# ---------------------------
# Display results from backend
# ---------------------------
job = st.session_state.get("job")

if job:
    st.markdown("---")
    st.header("📌 EVA Results")

    # ---------- Final Agent ----------
    st.subheader("1️⃣ Final Selected Agent")
    final_agent = job.get("final_agent")

    if final_agent:
        st.json(final_agent)
    else:
        st.info("No final_agent metadata returned from backend.")

    # ---------- Generation 1 Results ----------
    gen1_results = job.get("gen1_results") or []
    if gen1_results:
        st.markdown("---")
        st.subheader("2️⃣ Generation 1 – Candidate Agents & Scores")

        # Flatten for display (drop deep metrics column to keep table clean)
        df = pd.DataFrame(gen1_results)
        if "metrics" in df.columns:
            df_short = df.drop(columns=["metrics"])
        else:
            df_short = df

        st.dataframe(df_short)

        with st.expander("View detailed metrics for each agent"):
            st.json(gen1_results, expanded=False)

    # ---------- Mutated Result ----------
    mutated_result = job.get("mutated_result")
    if mutated_result:
        st.markdown("---")
        st.subheader("3️⃣ Mutated / Evolved Agent Evaluation")
        st.json(mutated_result)

    # ---------- Generated Agent Code ----------
    st.markdown("---")
    st.subheader("4️⃣ Generated Python Agent Code (Best Agent)")

    agent_code = job.get("agent_code")
    if agent_code:
        st.code(agent_code, language="python")

        st.download_button(
            label="⬇ Download Agent Code as .py",
            data=agent_code,
            file_name="eva_generated_agent.py",
            mime="text/x-python",
        )
    else:
        st.info(
            "No agent_code returned from backend. "
            "Double-check that your orchestrator uses blueprint_to_python_code "
            "and sets job['agent_code']."
        )

    # ---------- Raw job JSON (debug / judges) ----------
    with st.expander("🔍 Debug: Full raw job JSON from backend"):
        st.json(job, expanded=False)

st.markdown("---")
st.caption(
    "EVA Streamlit UI — fully dynamic, backend-powered (FastAPI + Gemini + .env)."
)
