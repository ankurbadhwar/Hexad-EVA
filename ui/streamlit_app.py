import streamlit as st
import pandas as pd
import time
import random
import json
import re
from io import StringIO

st.set_page_config(page_title="EVA — AI That Builds AIs", layout='wide')

# --------------------------------------------------
# Minimal Streamlit frontend for the EVA hackathon MVP
# --------------------------------------------------
# How it works:
# - User inputs a task and some test examples
# - The app "generates" N agent blueprints (prompt templates + params)
# - The app "tests" the agents on the provided examples with a simulated scoring function
# - The app performs a simple evolutionary mutation on the best agent and shows an improved version
#
# IMPORTANT: This file purposely uses simulated generation & evaluation functions so it's runnable
# immediately without keys or heavy dependencies. Replace the fake_* functions with real LLM
# and evaluation code (OpenAI or other LLM providers) when wiring the backend.
# --------------------------------------------------

# ----- Helpers -----

def sanitize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def fake_generate_agents(task: str, n: int, style_hint: str = ""):
    """Create N different prompt blueprints and parameters.
    Replace with real prompt-generation using an LLM.
    """
    agents = []
    base_prompts = [
        "You are an expert concise summarizer. Given the text, produce a 3-bullet summary that is clear and actionable.",
        "You are a teaching assistant. Summarize the lecture into 3 bullets with one practical takeaway each.",
        "You are a product manager summarizer. Provide 3 prioritized bullets: problem, solution, next-step." 
    ]
    temperatures = [0.2, 0.5, 0.8]
    for i in range(n):
        prompt = base_prompts[i % len(base_prompts)]
        if style_hint:
            prompt += f" Hint: {style_hint}" 
        agents.append({
            "id": f"agent_{i+1}",
            "name": f"Agent {i+1}",
            "prompt": f"{prompt}\n\nTask: {task}",
            "params": {"temperature": temperatures[i % len(temperatures)], "max_tokens": 256},
            "notes": "Blueprint (replace with live prompt + guardrails)",
        })
    return agents


def fake_score_output(agent_prompt: str, example: str):
    """Simulate a score between 0-1 based on overlap of important words. Used only for demo.
    Replace with real evaluation (ROUGE, embedding similarity, human eval, etc.)
    """
    # heuristics: reward shorter prompts that mention "concise" or "bullet"
    score = 0.5
    if 'concise' in agent_prompt.lower() or 'bullet' in agent_prompt.lower():
        score += 0.2
    # small random factor
    score += random.uniform(-0.15, 0.15)
    # penalize extremely long prompts
    if len(agent_prompt) > 800:
        score -= 0.1
    return max(0.0, min(1.0, score))


def fake_test_agents(agents, examples):
    results = []
    for a in agents:
        scores = [fake_score_output(a['prompt'], ex) for ex in examples]
        avg = sum(scores) / len(scores) if scores else 0
        results.append({
            "agent_id": a['id'],
            "name": a['name'],
            "avg_score": round(avg, 3),
            "raw_scores": [round(s, 3) for s in scores],
        })
    return results


def simple_mutation(best_agent):
    """Create a mutated/improved blueprint from the best agent.
    Replace this with evolutionary strategies calling LLM for mutation and re-evaluation.
    """
    new_prompt = best_agent['prompt']
    # small mutations: tighten instruction and add constraints
    if 'concise' not in new_prompt.lower():
        new_prompt = 'Be concise. ' + new_prompt
    # add evaluation constraint
    if '3 bullets' not in new_prompt.lower():
        new_prompt += ' Output must be exactly 3 bullets.'
    # tweak temperature param
    new_params = best_agent['params'].copy()
    new_params['temperature'] = max(0.1, new_params.get('temperature', 0.5) - 0.1)
    return {
        'id': best_agent['id'] + '_mut1',
        'name': best_agent['name'] + ' (evolved)',
        'prompt': new_prompt,
        'params': new_params,
        'notes': 'Evolved via simple mutation rule. Replace with LLM-driven mutations.'
    }


# ----- UI -----

st.title('EVA — The AI That Builds Other AIs (Streamlit MVP)')
st.markdown("""
EVA automates: **generate → build → test → evolve → finalize** an agent for the user's task.
This prototype demonstrates the *frontend flow*. Replace the fake_* functions with calls to
an LLM provider and your evaluation pipeline for a full implementation.
""")

with st.sidebar:
    st.header('EVA Settings')
    n_agents = st.number_input('Number of agent blueprints', min_value=2, max_value=6, value=3)
    style_hint = st.text_input('Style hint (optional)', placeholder='e.g., formal, layman, bullet-style')
    seed = st.number_input('Random seed (demo only)', value=42)
    st.write(' ') 
    st.write('Prototype notes: This UI simulates the pipeline. Integrate your LLM & evals in the backend.')

random.seed(seed)

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader('Task Input')
    task = st.text_area('Describe the user task for which EVA should build an agent', height=160,
                         placeholder='E.g., Summarize this lecture in 3 bullets.')

    st.subheader('Small testset (one example per line)')
    examples_raw = st.text_area('Example inputs to test generated agents (one per line)', height=160,
                                placeholder='Paste short lecture transcripts or text snippets here')

with col2:
    st.subheader('Quick Controls')
    run_gen = st.button('Generate Agents')
    run_test = st.button('Test Agents')
    run_evolve = st.button('Evolve Best Agent')
    st.write(' ')
    st.download_button('Download last run report (JSON)', data='{}', file_name='eva_report.json')

# Initialize session state
if 'agents' not in st.session_state:
    st.session_state['agents'] = []
if 'test_results' not in st.session_state:
    st.session_state['test_results'] = []
if 'evolved' not in st.session_state:
    st.session_state['evolved'] = None

# Generate agents
if run_gen:
    if not task.strip():
        st.warning('Please provide a task before generating agents.')
    else:
        with st.spinner('Generating agent blueprints...'):
            agents = fake_generate_agents(sanitize(task), int(n_agents), style_hint=sanitize(style_hint))
            time.sleep(0.8)
            st.session_state['agents'] = agents
            st.session_state['test_results'] = []
            st.session_state['evolved'] = None
        st.success(f'Generated {len(agents)} agent blueprints.')

# Display agents
if st.session_state['agents']:
    st.markdown('---')
    st.subheader('Generated Agent Blueprints')
    cols = st.columns(len(st.session_state['agents']))
    for i, a in enumerate(st.session_state['agents']):
        with cols[i]:
            st.caption(a['id'])
            st.markdown(f"**{a['name']}**")
            st.write(a['prompt'][:500] + ('...' if len(a['prompt']) > 500 else ''))
            st.write('**Params:**', a['params'])
            st.info(a['notes'])

# Test agents
if run_test:
    examples = [sanitize(x) for x in examples_raw.splitlines() if sanitize(x)]
    if not st.session_state['agents']:
        st.warning('No agents available. Click "Generate Agents" first.')
    elif not examples:
        st.warning('Provide at least one example in the testset.')
    else:
        with st.spinner('Testing agents on the examples...'):
            results = fake_test_agents(st.session_state['agents'], examples)
            st.session_state['test_results'] = results
            time.sleep(0.8)
        st.success('Testing complete.')

# Display test results
if st.session_state['test_results']:
    st.markdown('---')
    st.subheader('Test Results')
    df = pd.DataFrame(st.session_state['test_results'])
    # Expand raw_scores into columns if consistent
    max_scores = max(len(r['raw_scores']) for r in st.session_state['test_results'])
    for idx in range(max_scores):
        df[f'ex_{idx+1}'] = df['raw_scores'].apply(lambda rs: rs[idx] if idx < len(rs) else None)
    st.dataframe(df[['agent_id', 'name', 'avg_score'] + [c for c in df.columns if c.startswith('ex_')]])

# Evolve
if run_evolve:
    if not st.session_state['test_results']:
        st.warning('Run tests first so EVA can pick the best agent to evolve.')
    else:
        best = max(st.session_state['test_results'], key=lambda r: r['avg_score'])
        # find agent object
        best_agent = next((a for a in st.session_state['agents'] if a['id'] == best['agent_id']), None)
        if best_agent is None:
            st.error('Best agent not found — unexpected error')
        else:
            with st.spinner('Evolving the best agent...'):
                evolved = simple_mutation(best_agent)
                # simulate re-test for evolved
                evolved_score = fake_test_agents([evolved], [sanitize(x) for x in examples_raw.splitlines() if sanitize(x)])
                st.session_state['evolved'] = {'evolved': evolved, 'retest': evolved_score}
                time.sleep(0.8)
            st.success('Evolution complete.')

# Show evolved agent and comparison
if st.session_state.get('evolved'):
    st.markdown('---')
    st.subheader('Evolved Agent')
    ev = st.session_state['evolved']
    st.markdown(f"**{ev['evolved']['name']}**")
    st.write(ev['evolved']['prompt'][:800] + ('...' if len(ev['evolved']['prompt']) > 800 else ''))
    st.write('Params:', ev['evolved']['params'])
    st.info(ev['evolved']['notes'])

    st.write('Re-test results (simulated):')
    st.json(ev['retest'])

# Finalize & Export
st.markdown('---')
st.subheader('Finalize')
if st.session_state.get('evolved'):
    final_agent = st.session_state['evolved']['evolved']
    st.write('Final agent ready to export:')
    st.json(final_agent)
    payload = json.dumps({
        'task': task,
        'final_agent': final_agent,
        'test_results': st.session_state.get('test_results'),
    }, indent=2)
    st.download_button('Download final agent (JSON)', data=payload, file_name='eva_final_agent.json')
else:
    st.info('Evolve an agent to enable finalization and export.')

# Footer notes
st.markdown('---')
st.caption('Streamlit EVA MVP — replace fake generation/testing with live LLM calls and evaluation metric code for a production version.')
