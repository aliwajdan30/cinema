import json
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
def fetch_table_schemas():
    schema_info = {}
    try:
        conn = sql.connect(server_hostname=workspace_url, http_path=http_path, access_token=DATABRICKS_TOKEN)
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

@st.cache_data(show_spinner=False)
def fetch_survey_schema():
    try:
        conn = sql.connect(server_hostname=workspace_url, http_path=http_path, access_token=DATABRICKS_TOKEN)
        cursor = conn.cursor()
        cursor.execute("SELECT column_name, category, subcategory FROM dev.cinema.survey_columns")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return {row[0]: {"category": row[1], "subcategory": row[2]} for row in rows}
    except Exception as e:
        logger.error(f"Error fetching survey schema: {e}")
        return {}

def is_mixed_question(question):
    prompt = f"""
Classify the following question strictly as 'cinema' or 'mixed'.
- If the question is ONLY about movies, cinemas, showings, gender of cinema goers, locations, distributors, target groups etc., classify it as 'cinema'.
- If the question ALSO involves lifestyle dimensions like (but not limited to) income, car ownership, food, groceries, housing, banking, or family, classify it as 'mixed'.
Only respond with: cinema or mixed.

Question: \"{question}\"
"""
    response = llm.chat.completions.create(
        model="databricks-llama-4-maverick",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=10,
    )
    result = response.choices[0].message.content.strip().lower()
    logger.info(f"\U0001f9e0 Question classified as: {result.upper()}")
    return result

def is_generic_movie_question(question):
    keywords = ["tell me about", "info about", "information about", "what can you tell me"]
    return any(k in question.lower() for k in keywords) or len(question.strip().split()) <= 4

def generate_prompt_with_schema(question, survey_schema):
    schema_info = "\n".join([f"- {col}: Category: {desc['category']}, Subcategory: {desc['subcategory']}" for col, desc in survey_schema.items()])
    prompt = f"""
You are provided with the following survey schema:

{schema_info}

Based on this schema, identify which survey columns are relevant to answer the question: \"{question}\"
Return only the column names in a comma-separated list.
"""
    return prompt

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

def split_into_cinema_and_survey(question):
    prompt = f"""
You are a JSON-generating assistant.

Your task is to split the user's question into 2 parts:
1. CINEMA: Focused on movies, film titles, showings, admissions, distributors, or audience.
2. LIFESTYLE: Related to preferences or demographics like car brands, banks, food, grocery stores, family, housing, etc.

Split the question **accurately** by moving all lifestyle-related expressions (e.g. "which cars do they prefer", "what groceries stores do they shop at", "which type of cheese do they like", "do they eat meat") to the `survey` part, even if it overlaps with a movie reference.

Always respond with:
{{
  "cinema": "...",
  "survey": "..."
}}

Question: \"{question}\"
"""
    response = llm.chat.completions.create(
        model="databricks-llama-4-maverick",
        messages=[
            {"role": "system", "content": "You are a strict JSON generator. Do not explain or include any extra text."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=300,
    )
    try:
        return json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        logger.warning(f"⚠️ Failed to parse split JSON: {e}\nReturned text:\n{response.choices[0].message.content}")
        return {"cinema": question, "survey": ""}

# --- STREAMLIT UI ---
st.set_page_config(page_title="Film Monitor Assistant", layout="wide")
st.title("🎯 Film Monitor Assistant")

if "error" in TABLE_SCHEMAS:
    st.error(f"Schema failed to load: {TABLE_SCHEMAS['error']}")
else:
    st.success("✅ Schema loaded. Databricks-hosted LLM is ready!")

q = st.text_input("Ask a question:")

if q:
    st.write("📋 **Question:**", q)
    with st.spinner("Processing..."):
        try:
            q_type = is_mixed_question(q)
            st.write(f"🔍 **Classification:** `{q_type.upper()}`")

            if q_type == "mixed":
                parts = split_into_cinema_and_survey(q)
                cinema_q, survey_q = parts["cinema"], parts["survey"]
                st.write("🔍 **Classification:** Mixed")
                st.write("🎬 **Cinema Part:**", cinema_q)
                st.write("🛍️ **Lifestyle Part:**", survey_q)

                logger.info("🔗 Fetching target group from cinema question...")
                target_cols, target_rows = ask_genie(cinema_q)

                logger.info("📚 Fetching relevant survey columns...")
                survey_schema = fetch_survey_schema()
                survey_prompt = generate_prompt_with_schema(survey_q, survey_schema)
                col_response = llm.chat.completions.create(
                    model="databricks-llama-4-maverick",
                    messages=[{"role": "user", "content": survey_prompt}],
                    temperature=0.0,
                    max_tokens=150,
                )
                raw_col_output = col_response.choices[0].message.content.strip()
                # Attempt to extract columns from any text (if necessary)
                relevant_columns = raw_col_output.split(":")[-1].strip()
                logger.info(f"🧩 Relevant survey columns: {relevant_columns}")

                final_genie_prompt = f"""
                    For the previously identified target group, analyze these columns:
                    {relevant_columns}.
                    {survey_q}
                    """
                logger.info(f"🧠 Final genie question: {final_genie_prompt}")
                colnames, rows = ask_genie(final_genie_prompt)
                explanation = explain_answer(q, colnames, rows)

                st.subheader("📊 Raw Table")
                st.dataframe(pd.DataFrame(rows, columns=colnames), use_container_width=True)
                st.subheader("💬 Explanation")
                st.write(explanation)

            else:
                if is_generic_movie_question(q):
                    st.subheader("🎞️ General Movie Summary Mode")
                    movie_name = q.replace("tell me about", "").strip(" ?")

                    gender_q = f"What is the gender breakdown of people who watched {movie_name}, by location?"
                    age_q = f"What is the age group breakdown of people who watched {movie_name}, by location?"
                    admissions_q = f"What are the total admissions for {movie_name} by location?"
                    target_q = f"What is the target group profile for people who watched {movie_name}?"

                    for sub_q, label in [
                        (gender_q, "🧑‍🤝‍🧑 Gender Breakdown by Location"),
                        (age_q, "🎂 Age Group Breakdown by Location"),
                        (admissions_q, "🎟️ Total Admissions by Location"),
                        (target_q, "🎯 Target Group Summary")
                    ]:
                        cols, rows = ask_genie(sub_q)
                        exp = explain_answer(sub_q, cols, rows)
                        st.subheader(label)
                        st.dataframe(pd.DataFrame(rows, columns=cols), use_container_width=True)
                        st.write(exp)

                else:
                    colnames, rows = ask_genie(q)
                    explanation = explain_answer(q, colnames, rows)
                    st.subheader("📊 Raw Table")
                    st.dataframe(pd.DataFrame(rows, columns=colnames), use_container_width=True)
                    st.subheader("💬 Explanation")
                    st.write(explanation)

        except Exception as e:
            logger.exception("Unhandled error during processing.")
            st.error(f"❌ {str(e)}")