# streamlit_frontend.py
import streamlit as st
import requests
import json
from urllib.parse import urljoin
import html

st.set_page_config(page_title="EVA — Code-only Frontend", layout="centered")
st.title("🧬 EVA — Code-only pipeline")
st.caption("Backend returns paste-ready code (text/plain). Copy or download the file and run locally.")

# --- Backend settings ---
BACKEND_BASE = st.text_input("Backend URL (include protocol)", value="http://127.0.0.1:8000")
RUN_ENDPOINT = urljoin(BACKEND_BASE, "/run")
HEALTH_ENDPOINT = urljoin(BACKEND_BASE, "/health")

if not BACKEND_BASE:
    st.error("Please set the backend URL")
    st.stop()

# --- User input ---
problem = st.text_area("Describe the task for EVA", height=220,
                       placeholder="E.g. Write a python script that fetches top headlines and prints titles")

col_lang, col_file = st.columns([1, 1])
with col_lang:
    language = st.selectbox("Expected language (for highlighting & filename)", 
                            options=["python", "javascript", "bash", "typescript", "java", "text"],
                            index=0)
with col_file:
    default_name = {
        "python": "agent.py",
        "javascript": "agent.js",
        "bash": "agent.sh",
        "typescript": "agent.ts",
        "java": "Agent.java",
        "text": "agent.txt"
    }.get(language, "agent.txt")
    filename = st.text_input("Filename for download", value=default_name)

col1, col2 = st.columns([1, 1])
run_btn = col1.button("Run")
clear_btn = col2.button("Clear")

if clear_btn:
    st.experimental_rerun()

# --- Diagnostics sidebar ---
st.sidebar.header("Diagnostics")
if st.sidebar.button("Health check"):
    try:
        r = requests.get(HEALTH_ENDPOINT, timeout=5)
        r.raise_for_status()
        st.sidebar.success("OK")
        try:
            st.sidebar.json(r.json())
        except Exception:
            st.sidebar.write(r.text[:1000])
    except Exception as e:
        st.sidebar.error(f"Health failed: {e}")

# --- Run handler ---
if run_btn:
    if not problem.strip():
        st.error("Please enter a problem/task")
        st.stop()

    # Quick health check before long call
    try:
        h = requests.get(HEALTH_ENDPOINT, timeout=5)
        h.raise_for_status()
    except Exception as e:
        st.error(f"Backend health check failed: {e}")
        st.stop()

    st.info("Submitting task to backend — waiting for code response...")
    code_box = st.empty()
    info_box = st.empty()
    download_box = st.empty()

    # Use a generous timeout; backend may take time to run LLM calls
    try:
        with st.spinner("Running pipeline (this may take tens of seconds)..."):
            resp = requests.post(RUN_ENDPOINT, json={"problem": problem}, timeout=300)
            # If backend returned an HTTP error, show it
            try:
                resp.raise_for_status()
            except requests.HTTPError:
                # Try to show helpful message if backend returned JSON error
                content_type = resp.headers.get("Content-Type", "")
                if "application/json" in content_type:
                    try:
                        err = resp.json()
                        st.error(f"Backend error: {err}")
                    except Exception:
                        st.error(f"Backend error: {resp.text[:2000]}")
                else:
                    st.error(f"Backend returned HTTP {resp.status_code}: {resp.text[:2000]}")
                st.stop()

            # Success path: expect plain text / code
            returned = resp.text or ""
            if not returned.strip():
                st.warning("Backend returned empty response (no code).")
                st.stop()

            # Display code with syntax highlighting
            code_box.markdown("### ✅ Paste-ready code (copy or download below)")
            # Use st.code for highlighting
            st.code(returned, language=language)

            # Provide a download button
            download_box.download_button(
                label="Download code",
                data=returned,
                file_name=filename,
                mime="text/plain"
            )

            # Provide copy-to-clipboard button via JS
            safe_code = html.escape(returned)
            copy_html = f"""
            <textarea id="code-to-copy" style="display:none">{safe_code}</textarea>
            <button onclick="navigator.clipboard.writeText(document.getElementById('code-to-copy').value)"
                    style="padding:8px 12px; border-radius:6px; border:1px solid #ccc; background:#f6f6f6;">
              Copy to clipboard
            </button>
            <small style="display:block; margin-top:6px">Click to copy entire file to clipboard</small>
            """
            st.components.v1.html(copy_html, height=90)

            # Show short run instructions
            if language == "python":
                info_box.info(f"Run locally: `python {filename}`")
            elif language in ("bash", "sh"):
                info_box.info(f"Make executable and run: `chmod +x {filename}` then `./{filename}`")
            elif language in ("javascript", "node"):
                info_box.info(f"Run with Node.js: `node {filename}`")
            elif language == "typescript":
                info_box.info("Transpile then run or use ts-node: `ts-node {filename}`")
            elif language == "java":
                info_box.info("Compile & run: `javac Agent.java` then `java Agent`")
            else:
                info_box.info("Download the file and run according to its language/runtime.")

    except requests.Timeout:
        st.error("Request timed out. The backend may be busy — try again or increase timeout.")
    except Exception as e:
        st.error(f"Request failed: {e}")

st.markdown("---")
st.caption("If you want the frontend to show multiple candidate codes (instead of only the final code), tell me and I will adapt the UI to fetch and display them (requires backend change to return candidates).")
