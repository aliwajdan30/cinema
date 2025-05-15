import os, time, requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from databricks import sql
from openai import OpenAI  # from new OpenAI SDK

# --- ENVIRONMENT ---
load_dotenv()
DATABRICKS_TOKEN = os.getenv("DATABRICKS_ACCESS_TOKEN")
access_token = os.getenv("DATABRICKS_ACCESS_TOKEN")
workspace_url = os.getenv("DATABRICKS_HOST")
http_path = os.getenv("DATABRICKS_HTTP_PATH")
space_id = os.getenv("GENIE_SPACE_ID")
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
        for table in ["cluster_results", "cinema_cluster_results", "survey_cluster_results"]:
            cursor.execute(f"DESCRIBE TABLE dev.persona.{table}")
            rows = cursor.fetchall()
            schema_info[table] = [f"{r[0]} ({r[1]})" for r in rows if r[0] and not r[0].startswith("#")]
        cursor.close()
        conn.close()
    except Exception as e:
        schema_info["error"] = str(e)
    return schema_info

TABLE_SCHEMAS = fetch_table_schemas()

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
        post = requests.post(msg_url, headers=headers, json={"content": simplified_question}).json()
        msg_id = post["id"]
    else:
        start_url = f"https://{workspace_url}/api/2.0/genie/spaces/{space_id}/start-conversation"
        start = requests.post(start_url, headers=headers, json={"content": simplified_question}).json()
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
    result = requests.get(result_url, headers=headers).json()
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

# --- UI ---
st.set_page_config(page_title="Cinema/Norstat Assistant", layout="wide")
st.title("🎯 Cinema/Norstat Assistant")

if "error" in TABLE_SCHEMAS:
    st.error(f"Schema failed to load: {TABLE_SCHEMAS['error']}")
else:
    st.success("✅ Schema loaded. Databricks-hosted LLM is ready!")

q = st.text_input("Ask a question:")

if q:
    with st.spinner("Processing..."):
        try:
            simplified = simplify_question(q)
            colnames, rows = ask_genie(simplified)
            explanation = explain_answer(q, colnames, rows)

            st.subheader("📊 Raw Table")
            st.dataframe(pd.DataFrame(rows, columns=colnames), use_container_width=True)

            st.subheader("💬 Explanation")
            st.write(explanation)
        except Exception as e:
            st.error(f"❌ {str(e)}")