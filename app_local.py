import os
import time
import json
import requests
import logging
import pandas as pd
import streamlit as st
from databricks import sql
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# --- LOGGER SETUP ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- SECRETS ---
DATABRICKS_TOKEN = os.getenv("DATABRICKS_ACCESS_TOKEN")
workspace_url = os.getenv("DATABRICKS_HOST")
http_path = os.getenv("DATABRICKS_HTTP_PATH")

GENIE_SPACE_IDS = {
    "filmmonitor": os.getenv("GENIE_FILMMONITOR_SPACE_ID"),
    "mobility": os.getenv("GENIE_MOBILITY_SPACE_ID")
}
headers = {"Authorization": f"Bearer {DATABRICKS_TOKEN}"}

# --- INIT TOP LLM ---
llm = OpenAI(
    api_key=DATABRICKS_TOKEN,
    base_url=f"https://{workspace_url}/serving-endpoints"
)

# --- INIT SESSION STATE ---
if "genie_conversation_ids" not in st.session_state:
    st.session_state.genie_conversation_ids = {}

# --- FUNCTIONS ---
def classify_domain(question):
    prompt = f"""
Classify the domain of the following question strictly as one of:
- filmmonitor
- mobility
Examples:
- "Which target groups watched Barbie?" → filmmonitor
- "Which areas have long first mile distances?" → mobility
Only respond with: filmmonitor or mobility.
Question: "{question}"
"""
    response = llm.chat.completions.create(
        model="databricks-llama-4-maverick",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=10,
    )
    domain = response.choices[0].message.content.strip().lower()
    logger.info(f"🌍 Domain classified as: {domain.upper()}")
    return domain

def ask_genie(question, domain):
    space_id = GENIE_SPACE_IDS[domain]
    if domain not in st.session_state.genie_conversation_ids:
        # Start new conversation
        start_url = f"https://{workspace_url}/api/2.0/genie/spaces/{space_id}/start-conversation"
        logger.info(f"🆕 Starting new Genie conversation ({domain.upper()}): {question}")
        start_resp = requests.post(start_url, headers=headers, json={"content": question})
        start = start_resp.json()
        if "conversation" not in start:
            raise ValueError(f"Invalid response from Genie when starting conversation: {start}")
        conv_id = start["conversation"]["id"]
        msg_id = start["message"]["id"]
        st.session_state.genie_conversation_ids[domain] = conv_id
    else:
        # Follow-up message
        conv_id = st.session_state.genie_conversation_ids[domain]
        msg_url = f"https://{workspace_url}/api/2.0/genie/spaces/{space_id}/conversations/{conv_id}/messages"
        logger.info(f"↩️ Sending follow-up to Genie ({domain.upper()}): {question}")
        post_resp = requests.post(msg_url, headers=headers, json={"content": question})
        post = post_resp.json()
        msg_id = post["id"]

    # Poll for result
    poll_url = f"https://{workspace_url}/api/2.0/genie/spaces/{space_id}/conversations/{conv_id}/messages/{msg_id}"
    for _ in range(20):
        poll = requests.get(poll_url, headers=headers).json()
        if poll.get("status") == "COMPLETED" and poll.get("attachments"):
            attachment_id = poll["attachments"][0]["attachment_id"]
            break
        time.sleep(2)
    else:
        raise TimeoutError("❌ Genie timed out")

    # Get result
    result_url = f"https://{workspace_url}/api/2.0/genie/spaces/{space_id}/conversations/{conv_id}/messages/{msg_id}/attachments/{attachment_id}/query-result"
    result_resp = requests.get(result_url, headers=headers).json()
    rows = result_resp["statement_response"]["result"]["data_array"]
    cols = [col["name"] for col in result_resp["statement_response"]["manifest"]["schema"]["columns"]]
    return cols, rows

def explain_answer(question, cols, rows):
    df = pd.DataFrame(rows, columns=cols)
    preview = df.head().to_markdown(index=False)
    prompt = f"""Given this SQL result:\n{preview}\nExplain it simply in context of the original question: '{question}'"""
    response = llm.chat.completions.create(
        model="databricks-llama-4-maverick",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=250,
    )
    return response.choices[0].message.content.strip()

# --- STREAMLIT UI ---
st.set_page_config(page_title="Multi-Domain Genie Assistant", layout="wide")
st.title("🎯 Multi-Domain Genie Assistant")
q = st.text_input("Ask a question:")
if q:
    with st.spinner("Processing..."):
        try:
            domain = classify_domain(q)
            st.write(f"📁 **Classification:** `{domain}`")
            cols, rows = ask_genie(q, domain)
            explanation = explain_answer(q, cols, rows)
            st.subheader("📊 Raw Table")
            st.dataframe(pd.DataFrame(rows, columns=cols), use_container_width=True)
            st.subheader("💬 Explanation")
            st.write(explanation)
        except Exception as e:
            logger.exception("❌ Unhandled error")
            st.error(f"❌ {str(e)}")