import os, time, requests
import pandas as pd
import streamlit as st
from databricks import sql
from openai import OpenAI  # from new OpenAI SDK
import json
import logging

# --- LOGGER SETUP ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- SECRETS ---
DATABRICKS_TOKEN = st.secrets["DATABRICKS_ACCESS_TOKEN"]
access_token = DATABRICKS_TOKEN
workspace_url = st.secrets["DATABRICKS_HOST"]
http_path = st.secrets["DATABRICKS_HTTP_PATH"]
space_id = st.secrets["GENIE_SPACE_ID"]
headers = {"Authorization": f"Bearer {access_token}"}

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
def fetch_table_schemas():
    schema_info = {}
    try:
        conn = sql.connect(
            server_hostname=workspace_url,
            http_path=http_path,
            access_token=access_token
        )
        cursor = conn.cursor()
        for table in ["survey"]:
            cursor.execute(f"DESCRIBE TABLE dev.cinema.{table}")
            rows = cursor.fetchall()
            schema_info[table] = [f"{r[0]} ({r[1]})" for r in rows if r[0] and not r[0].startswith("#")]
        cursor.close()
        conn.close()
    except Exception as e:
        schema_info["error"] = str(e)
    return schema_info

TABLE_SCHEMAS = fetch_table_schemas()

# --- QUESTION CLASSIFIER ---
def is_mixed_question(question):
    prompt = f"""
You are a classifier. Decide whether the following question is:

(a) only about cinema-related data (e.g. movies, showings, visits, locations, demographics, distributors), or
(b) a combination of cinema data and lifestyle/survey information (e.g. banking, shopping, food, housing, preferences, income, etc.).

Only return: 'cinema' or 'mixed'

Question: "{question}"
"""
    response = llm.chat.completions.create(
        model="databricks-llama-4-maverick",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=10,
    )
    return response.choices[0].message.content.strip().lower() == "mixed"

# --- LLM QUESTION SIMPLIFIER ---
def simplify_question(user_question):
    schema_text = ""
    for table, cols in TABLE_SCHEMAS.items():
        if table != "error":
            schema_text += f"\n📂 {table}:\n" + "\n".join([f"  - {c}" for c in cols])

    prompt = f"""
Below is the schema of available tables in the database. Rewrite the following user question into a form that a SQL generator (Databricks Genie) can understand.

📘 Schema:
{schema_text}

User question: "{user_question}"

Rewrite clearly describing the analytical intent (but do not write SQL):
"""

    response = llm.chat.completions.create(
        model="databricks-llama-4-maverick",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=150,
    )
    return response.choices[0].message.content.strip()

# --- GENIE SQL FLOW ---
def ask_genie(simplified_question):
    if st.session_state.genie_conversation_id:
        conv_id = st.session_state.genie_conversation_id
        msg_url = f"https://{workspace_url}/api/2.0/genie/spaces/{space_id}/conversations/{conv_id}/messages"
        logger.info(f"Sending follow-up to Genie: {simplified_question}")
        post_resp = requests.post(msg_url, headers=headers, json={"content": simplified_question})
        logger.info(f"Genie follow-up response: {post_resp.status_code} | {post_resp.text}")
        post = post_resp.json()
        msg_id = post["id"]
    else:
        start_url = f"https://{workspace_url}/api/2.0/genie/spaces/{space_id}/start-conversation"
        logger.info(f"Sending to Genie: {simplified_question}")
        start_resp = requests.post(start_url, headers=headers, json={"content": simplified_question})
        logger.info(f"Genie response: {start_resp.status_code} | {start_resp.text}")
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
    logger.info(f"Fetching Genie result from: {result_url}")
    result_resp = requests.get(result_url, headers=headers)
    logger.info(f"Genie result status: {result_resp.status_code} | {result_resp.text}")
    result = result_resp.json()
    rows = result["statement_response"]["result"]["data_array"]
    cols = [col["name"] for col in result["statement_response"]["manifest"]["schema"]["columns"]]
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
            is_mixed = is_mixed_question(q)
            st.write("🔍 **Classified as:**", "Mixed" if is_mixed else "Cinema")
            logger.info(f"Question classified as: {'mixed' if is_mixed else 'cinema'}")

            if is_mixed:
                # Split the mixed question using LLM
                classify_prompt = f"""
Split the following user question into two separate sub-questions:
(1) A cinema-related sub-question
(2) A related lifestyle/survey sub-question based on target group

User question: \"{q}\"

Return as JSON:
{{
  \"cinema_question\": \"...\",
  \"survey_question\": \"...\"
}}
"""
                classify_response = llm.chat.completions.create(
                    model="databricks-llama-4-maverick",
                    messages=[{"role": "user", "content": classify_prompt}],
                    temperature=0.0,
                    max_tokens=200,
                )
                try:
                    content = classify_response.choices[0].message.content.strip()
                    logger.info(f"LLM classify response: {content}")
                    parts = json.loads(content)
                    cinema_q = parts["cinema_question"]
                    survey_q = parts["survey_question"]
                except json.JSONDecodeError as e:
                    raise ValueError(f"Failed to split the question properly: {e}")

                logger.info(f"Original mixed question: {q}")
                logger.info(f"Cinema sub-question: {cinema_q}")
                logger.info(f"Survey sub-question: {survey_q}")

                simplified = cinema_q #simplify_question(cinema_q)
                st.code(f"Simplified Question (cinema): {simplified}", language="markdown")
                logger.info(f"Simplified question (cinema): {simplified}")

                colnames, rows = ask_genie(simplified)
                logger.info(f"Genie returned columns: {colnames}")

                df_genie = pd.DataFrame(rows, columns=colnames)
                preview = df_genie.head().to_dict(orient="records")

                prompt = f"""This is the result of querying cinema data for the question: '{q}'.
{preview}
Now use this to find relevant information from the 'survey' table based on target_group.
Write a brief answer combining both parts."""
                response = llm.chat.completions.create(
                    model="databricks-llama-4-maverick",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.5,
                    max_tokens=300,
                )

                st.subheader("💬 Combined Insight")
                st.write(response.choices[0].message.content.strip())

                st.subheader("📊 Genie Output")
                st.dataframe(df_genie, use_container_width=True)

            else:
                simplified = simplify_question(q)
                st.code(f"Simplified Question: {simplified}", language="markdown")
                logger.info(f"Simplified question: {simplified}")

                colnames, rows = ask_genie(simplified)
                logger.info(f"Genie returned columns: {colnames}")

                explanation = explain_answer(q, colnames, rows)

                st.subheader("📊 Raw Table")
                st.dataframe(pd.DataFrame(rows, columns=colnames), use_container_width=True)

                st.subheader("💬 Explanation")
                st.write(explanation)

        except Exception as e:
            logger.exception("Unhandled error during processing.")
            st.error(f"❌ {str(e)}")
