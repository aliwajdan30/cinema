import os, time, requests
import pandas as pd
import streamlit as st
from databricks import sql
from openai import OpenAI
import logging

# --- LOGGER SETUP ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- SECRETS ---
DATABRICKS_TOKEN = st.secrets["DATABRICKS_ACCESS_TOKEN"]
workspace_url = st.secrets["DATABRICKS_HOST"]
http_path = st.secrets["DATABRICKS_HTTP_PATH"]
space_id = st.secrets["GENIE_SPACE_ID"]
headers = {"Authorization": f"Bearer {DATABRICKS_TOKEN}"}

# --- INIT LLM ---
llm = OpenAI(
    api_key=DATABRICKS_TOKEN,
    base_url="https://adb-1428801648127728.8.azuredatabricks.net/serving-endpoints"
)

# --- INIT SESSION STATE ---
if "genie_conversation_id" not in st.session_state:
    st.session_state.genie_conversation_id = None

# --- SCHEMA LOADER ---
@st.cache_data(show_spinner=False)
@st.cache_data(show_spinner=False)
def fetch_table_schemas():
    schema_info = {}
    try:
        conn = sql.connect(
            server_hostname=workspace_url,
            http_path=http_path,
            access_token=DATABRICKS_TOKEN
        )
        cursor = conn.cursor()

        # 👇 Add more tables if needed here
        tables = ["survey"]

        for table in tables:
            try:
                cursor.execute(f"DESCRIBE TABLE dev.cinema.{table}")
                rows = cursor.fetchall()
                cleaned = []
                for r in rows:
                    # Ensure r has at least 2 entries and both are strings
                    if len(r) >= 2 and isinstance(r[0], str) and isinstance(r[1], str):
                        cleaned.append(f"{r[0]} ({r[1]})")
                schema_info[table] = cleaned
            except Exception as inner_e:
                schema_info[table] = [f"⚠️ Failed to load schema: {str(inner_e)}"]

        cursor.close()
        conn.close()
    except Exception as e:
        schema_info["error"] = str(e)

    return schema_info

TABLE_SCHEMAS = fetch_table_schemas()

# --- GENIE SQL FLOW ---
def ask_genie(question):
    if st.session_state.genie_conversation_id:
        conv_id = st.session_state.genie_conversation_id
        msg_url = f"https://{workspace_url}/api/2.0/genie/spaces/{space_id}/conversations/{conv_id}/messages"
        logger.info(f"Sending follow-up to Genie: {question}")
        post_resp = requests.post(msg_url, headers=headers, json={"content": question})
        post = post_resp.json()
        msg_id = post["id"]
    else:
        start_url = f"https://{workspace_url}/api/2.0/genie/spaces/{space_id}/start-conversation"
        logger.info(f"Starting new conversation with Genie: {question}")
        start_resp = requests.post(start_url, headers=headers, json={"content": question})
        start = start_resp.json()
        conv_id, msg_id = start["conversation"]["id"], start["message"]["id"]
        st.session_state.genie_conversation_id = conv_id

    poll_url = f"https://{workspace_url}/api/2.0/genie/spaces/{space_id}/conversations/{conv_id}/messages/{msg_id}"
    for _ in range(20):
        poll = requests.get(poll_url, headers=headers).json()
        if poll.get("status") == "COMPLETED" and poll.get("attachments"):
            attachment_id = poll["attachments"][0]["attachment_id"]
            break
        time.sleep(2)
    else:
        raise TimeoutError("❌ Genie timed out")

    result_url = f"https://{workspace_url}/api/2.0/genie/spaces/{space_id}/conversations/{conv_id}/messages/{msg_id}/attachments/{attachment_id}/query-result"
    result_resp = requests.get(result_url, headers=headers).json()
    rows = result_resp["statement_response"]["result"]["data_array"]
    cols = [col["name"] for col in result_resp["statement_response"]["manifest"]["schema"]["columns"]]
    return cols, rows

# --- INTERPRETATION ---
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
st.set_page_config(page_title="Cinema/Norstat Assistant", layout="wide")
st.title("🎯 Cinema/Norstat Assistant")

if "error" in TABLE_SCHEMAS:
    st.error(f"Schema failed to load: {TABLE_SCHEMAS['error']}")
else:
    st.success("✅ Schema loaded. Databricks-hosted LLM is ready!")

q = st.text_input("Ask a question:")

if q:
    st.write("📋 **Question:**", q)
    with st.spinner("Processing..."):
        try:
            colnames, rows = ask_genie(q)
            explanation = explain_answer(q, colnames, rows)

            st.subheader("📊 Raw Table")
            st.dataframe(pd.DataFrame(rows, columns=colnames), use_container_width=True)

            st.subheader("💬 Explanation")
            st.write(explanation)

        except Exception as e:
            logger.exception("Unhandled error during processing.")
            st.error(f"❌ {str(e)}")
