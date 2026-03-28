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
    .executive-header { font-size: 1.6rem; color: #58a6ff; font-weight: 700; margin-bottom: 15px; border-left: 5px solid #58a6ff; padding-left: 15px; }
    .stButton>button { width: 100%; border-radius: 5px; background-color: #238636; color: white; border: none; }
    .stButton>button:hover { background-color: #2ea043; }
    .stInfo { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; font-size: 1.05rem; line-height: 1.5; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; padding: 15px; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATABASE ENGINE ---
@st.cache_resource
def get_analytics_engine():
    db_path = "instacart_analytics.duckdb"
    if not os.path.exists(db_path):
        st.error(f"FATAL: '{db_path}' not found. Run your database setup script first!")
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
                             trendline="ols", title="Metric Correlation Analysis", template="plotly_dark")
        elif len(cat_cols) >= 1 and len(num_cols) >= 1:
            return px.bar(plot_df, x=cat_cols[0], y=num_cols[0], color=num_cols[0], 
                          title=f"{num_cols[0]} by {cat_cols[0]}", template="plotly_dark")
        return None
    except: return None

# --- 4. THE SPEED-OPTIMIZED AGENT ---
def build_senior_agent(api_key, model_name, is_local=False):
    engine = get_analytics_engine()
    db = SQLDatabase(engine, include_tables=['order_products', 'products', 'aisles', 'departments', 'orders'])
    actual_schema = db.get_table_info()
    
    if is_local:
        # Llama 3.1 is highly optimized for this context size and tool calling
        llm = ChatOllama(model=model_name, temperature=0, num_ctx=4096)
        agent_type = "zero-shot-react-description"
    else:
        llm = ChatGroq(model=model_name, api_key=api_key, temperature=0)
        agent_type = "tool-calling"

    prefix = f"""
    You are a Lead BI Consultant at i2e Consulting. You solve business problems using SQL.

    SPEED & EFFICIENCY RULE (CRITICAL):
    DO NOT use the 'sql_db_list_tables', 'sql_db_schema', or 'sql_db_query_checker' tools. 
    The complete database schema is provided below. Proceed IMMEDIATELY to using the 'sql_db_query' tool.

    DATABASE SCHEMA:
    {actual_schema}

    MANDATORY JOIN LOGIC:
    'order_products' does NOT contain an 'aisle_id' or 'department_id'. 
    To connect them, you MUST use the 'products' table as a bridge:
    - JOIN order_products TO products ON order_products.product_id = products.product_id
    - JOIN products TO aisles ON products.aisle_id = aisles.aisle_id
    - JOIN products TO departments ON products.department_id = departments.department_id
    
    CRITICAL TOOL FORMATTING (FAILURE WILL RESULT IN SYSTEM CRASH):
    When you use the sql_db_query tool, your Action Input MUST be raw, plain text SQL.
    ABSOLUTELY NO MARKDOWN. NO BACKTICKS (`). NO ```sql. 

    Example of CORRECT Action Input:
    SELECT a.aisle, SUM(op.reordered) FROM aisles a JOIN products p ON a.aisle_id = p.aisle_id LIMIT 5;

    MANDATORY OUTPUT FORMAT: You MUST begin your final response with the exact words "Final Answer:" followed by:
       
    Final Answer:
    EXECUTIVE SUMMARY: [1-2 sentences directly answering the prompt]
    STRATEGIC NARRATIVE: [1 paragraph explaining the business impact]
    ```sql
    [Your query here]
    ```
    """
    
    return create_sql_agent(
        llm=llm, db=db, agent_type=agent_type, verbose=True, 
        prefix=prefix, handle_parsing_errors=True, max_iterations=5,
        use_query_checker=False # Bypasses the LLM double-check trap
    )

# --- 5. UI LAYOUT & SIDEBAR ---
st.title("🚀 i2e Intelligent BI Agent")
st.markdown("##### *32M Row Real-Time Analytics | Powered by DuckDB & Local LLMs*")

with st.sidebar:
    st.header("🛠️ Configuration")
    exec_mode = st.toggle("Local Mode (Ollama)", value=True)
    if exec_mode:
        # Llama set as the default model
        selected_model = st.selectbox(
            "Specialized LLM", 
            ["llama3.1:8b", "qwen2.5-coder:7b", "sqlcoder:7b", "deepseek-r1:7b", "qwen2.5-coder:3b"]
        )
        api_key = None
    else:
        api_key = st.text_input("Groq API Key", type="password")
        selected_model = st.selectbox("Cloud LLM", ["llama-3.3-70b-versatile"])
    
    if st.button("Reset Analysis Session"):
        st.session_state.history = []
        st.rerun()

# --- 6. EXECUTION ENGINE WITH MEMORY ---
if "history" not in st.session_state: st.session_state.history = []

for chat in st.session_state.history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])
        if "df" in chat: st.dataframe(chat["df"], use_container_width=True)
        if "chart" in chat: st.plotly_chart(chat["chart"])

if user_input := st.chat_input("Ask a complex business question..."):
    st.chat_message("user").markdown(user_input)
    
    with st.chat_message("assistant"):
        with st.status(f"Consulting {selected_model}...", expanded=True) as status:
            try:
                # CONVERSATIONAL MEMORY INJECTION
                chat_context = ""
                if len(st.session_state.history) > 0:
                    last_interaction = st.session_state.history[-1]
                    chat_context = f"Previous Query Context: {last_interaction['content'][:250]}... "

                final_prompt = f"STRICT RULE: {chat_context} Execute SQL and calculate exact metrics for: {user_input}"
                
                agent = build_senior_agent(api_key, selected_model, is_local=exec_mode)
                response = agent.invoke({"input": final_prompt})
                
                full_raw = response["output"]
                
                # Clean up the output to remove the 'Final Answer:' trigger word for the UI
                sql_match = re.search(r"```sql\n(.*?)\n```", full_raw, re.DOTALL | re.IGNORECASE)
                narrative = re.sub(r"```sql\n(.*?)\n```", "", full_raw, flags=re.DOTALL)
                narrative = narrative.replace("Final Answer:", "").strip()

                df_final = None
                if sql_match:
                    query = sql_match.group(1).strip()
                    with get_analytics_engine().connect() as conn:
                        df_final = pd.read_sql(text(query), conn)

                status.update(label="Analysis Complete", state="complete", expanded=False)

            except Exception as e:
                status.update(label="System Error", state="error")
                st.error(f"Consultant Error: {str(e)}")
                st.stop()

        # --- PROFESSIONAL UI RENDERING ---
        st.markdown("<div class='executive-header'>📊 Executive Analysis</div>", unsafe_allow_html=True)
        
        if narrative:
            st.info(narrative)
        
        if df_final is not None and not df_final.empty:
            st.write("---")
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                st.metric("Records Found", f"{len(df_final):,}")
            with col2:
                st.metric("Data Source", "DuckDB Engine")
            
            chart_fig = generate_industry_chart(df_final)
            if chart_fig: 
                st.plotly_chart(chart_fig, use_container_width=True)
            
            st.subheader("📁 Data & Audit Trail")
            st.dataframe(df_final, use_container_width=True)
            
            with st.expander("🛠️ View Generated SQL Query", expanded=False):
                if sql_match: st.code(query, language="sql")
            
            st.download_button("📥 Download Result Set (CSV)", 
                             data=df_final.to_csv(index=False).encode('utf-8'), 
                             file_name="i2e_analysis.csv", mime="text/csv")
            
            st.session_state.history.append({"role": "assistant", "content": narrative, "df": df_final, "chart": chart_fig})
        else:
            st.warning("Analysis completed, but no relevant data was found. Try adjusting your query parameters.")
            st.session_state.history.append({"role": "assistant", "content": narrative})