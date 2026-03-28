import streamlit as st
import pandas as pd
import plotly.express as px
import re
import os
import logging
import warnings
from sqlalchemy import create_engine, text
from langchain_community.utilities import SQLDatabase
from langchain_ollama import ChatOllama 
from langchain_groq import ChatGroq
from langchain_community.agent_toolkits import create_sql_agent

# --- 1. CONFIGURATION & ERROR SILENCING ---
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore", category=UserWarning, module='duckdb_engine')

st.set_page_config(page_title="i2e Intelligent BI | Hireathon Edition", page_icon="📈", layout="wide")

# Professional UI Styling
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .executive-header { font-size: 1.6rem; color: #58a6ff; font-weight: 700; margin-bottom: 10px; border-left: 5px solid #58a6ff; padding-left: 15px; }
    .stButton>button { width: 100%; border-radius: 5px; background-color: #238636; color: white; }
    .stInfo { background-color: #161b22; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATABASE ENGINE ---
@st.cache_resource
def get_analytics_engine():
    db_path = "instacart_analytics.duckdb"
    if not os.path.exists(db_path):
        st.error(f"FATAL: '{db_path}' not found. Run 1_database_setup.py first!")
        st.stop()
    return create_engine(f"duckdb:///{db_path}", pool_pre_ping=True)

# --- 3. INTELLIGENT CHARTING ---
def generate_industry_chart(df: pd.DataFrame):
    try:
        if df.empty or len(df.columns) < 2: return None
        plot_df = df.copy()
        plot_df.columns = [str(c).replace('_', ' ').title() for c in plot_df.columns]
        num_cols = plot_df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = plot_df.select_dtypes(exclude=['number']).columns.tolist()

        if len(num_cols) >= 2:
            return px.scatter(plot_df, x=num_cols[0], y=num_cols[1], hover_name=plot_df.columns[0], 
                             trendline="ols", title="Metric Correlation", template="plotly_dark")
        elif len(cat_cols) >= 1 and len(num_cols) >= 1:
            return px.bar(plot_df, x=cat_cols[0], y=num_cols[0], color=num_cols[0], 
                          title=f"{num_cols[0]} by {cat_cols[0]}", template="plotly_dark")
        return None
    except: return None

# --- 4. THE LEAD CONSULTANT AGENT ---
def build_senior_agent(api_key, model_name, is_local=False):
    engine = get_analytics_engine()
    # Explicitly matching your 1_database_setup.py table names
    db = SQLDatabase(engine, include_tables=['order_products', 'products', 'aisles', 'departments', 'orders'])
    
    if is_local:
        # 4096 context is vital for schema-heavy SQL tasks
        llm = ChatOllama(model=model_name, temperature=0, num_ctx=4096)
        agent_type = "zero-shot-react-description"
    else:
        llm = ChatGroq(model=model_name, api_key=api_key, temperature=0)
        agent_type = "tool-calling"

    prefix = """
    You are a Lead BI Consultant. You solve problems using SQL.
    
    CRITICAL INSTRUCTIONS:
    1. You MUST use the 'sql_db_query' tool to fetch data. Do not guess.
    2. Your final response MUST follow this exact structure:
       EXECUTIVE SUMMARY: (Short answer)
       STRATEGIC NARRATIVE: (Business impact)
       SQL BLOCK: (The code in a ```sql block)

    TABLE INFO:
    - 'order_products' (order_id, product_id, reordered)
    - 'products' (product_id, product_name, aisle_id)
    - 'aisles' (aisle_id, aisle)
    """
    
    return create_sql_agent(
        llm=llm, db=db, agent_type=agent_type, verbose=True, 
        prefix=prefix, handle_parsing_errors=True, max_iterations=10
    )

# --- 5. UI LAYOUT ---
st.title("🚀 i2e Intelligent BI Agent")
st.markdown("##### *32M Row Real-Time Analytics | Powered by DuckDB & Qwen2.5-Coder*")

with st.sidebar:
    st.header("🛠️ Configuration")
    exec_mode = st.toggle("Local Mode (Ollama)", value=True)
    if exec_mode:
        selected_model = st.selectbox("Specialized LLM", ["qwen2.5-coder:7b", "sqlcoder:7b", "qwen2.5-coder:3b"])
        api_key = None
    else:
        api_key = st.text_input("Groq API Key", type="password")
        selected_model = st.selectbox("Cloud LLM", ["llama-3.3-70b-versatile"])
    
    if st.button("Reset Analysis Session"):
        st.session_state.history = []
        st.rerun()

# --- 6. EXECUTION ENGINE ---
if "history" not in st.session_state: st.session_state.history = []

for chat in st.session_state.history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])
        if "df" in chat: st.dataframe(chat["df"], use_container_width=True)
        if "chart" in chat: st.plotly_chart(chat["chart"])

if user_input := st.chat_input("Ask a business question (e.g., 'Top 5 aisles by reorders')"):
    st.chat_message("user").markdown(user_input)
    
    with st.chat_message("assistant"):
        with st.status(f"Consulting {selected_model}...", expanded=True) as status:
            try:
                agent = build_senior_agent(api_key, selected_model, is_local=exec_mode)
                # The "handle_parsing_errors=True" in build_senior_agent is key here
                response = agent.invoke({"input": f"STRICT: Execute SQL for: {user_input}"})
                
                full_raw = response["output"]
                # Robust extraction for SQL blocks
                sql_match = re.search(r"```sql\n(.*?)\n```", full_raw, re.DOTALL | re.IGNORECASE)
                narrative = re.sub(r"```sql\n(.*?)\n```", "", full_raw, flags=re.DOTALL).strip()

                df_final = None
                if sql_match:
                    query = sql_match.group(1).strip()
                    with get_analytics_engine().connect() as conn:
                        df_final = pd.read_sql(text(query), conn)

                status.update(label="Analysis Verified", state="complete", expanded=False)

                # OUTPUT RENDERING
                st.markdown("<div class='executive-header'>📊 Executive Insights</div>", unsafe_allow_html=True)
                
                if narrative:
                    st.info(narrative)
                
                if df_final is not None and not df_final.empty:
                    chart_fig = generate_industry_chart(df_final)
                    if chart_fig: st.plotly_chart(chart_fig, use_container_width=True)
                    
                    with st.expander("🔍 Detailed Data Audit & Export", expanded=True):
                        st.dataframe(df_final, use_container_width=True)
                        if sql_match: st.code(query, language="sql")
                        csv = df_final.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Export CSV Report", data=csv, file_name="i2e_analysis.csv", mime="text/csv")
                    
                    st.session_state.history.append({"role": "assistant", "content": narrative, "df": df_final, "chart": chart_fig})
                else:
                    st.warning("Query returned successfully but no rows were found. Check filters.")

            except Exception as e:
                status.update(label="System Error", state="error")
                st.error(f"The Consultant encountered an error: {str(e)}")