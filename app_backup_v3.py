import streamlit as st
import pandas as pd
import plotly.express as px
import os
import logging
import warnings
import time

from sqlalchemy import create_engine, text
from langchain_community.utilities import SQLDatabase
from langchain_ollama import ChatOllama 
from langchain_groq import ChatGroq

# --- 1. CONFIGURATION & ERROR SILENCING ---
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore", category=UserWarning, module='duckdb_engine')

st.set_page_config(page_title="i2e Intelligent BI | Hireathon Edition", page_icon="📈", layout="wide")

# ⚡ UI POLISH: Premium Enterprise CSS
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .executive-header { font-size: 1.6rem; color: #58a6ff; font-weight: 700; margin-bottom: 15px; border-left: 5px solid #58a6ff; padding-left: 15px; }
    .stButton>button { width: 100%; border-radius: 5px; background-color: #238636; color: white; border: none; font-weight: bold; }
    .stButton>button:hover { background-color: #2ea043; box-shadow: 0 4px 12px rgba(46, 160, 67, 0.4); transition: all 0.3s ease; }
    .stInfo { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; font-size: 1.05rem; line-height: 1.6; border-left: 4px solid #a371f7; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; padding: 15px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.2); }
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

# --- 3. INTELLIGENT CHARTING (With Elastic UI Guardrails) ---
def generate_industry_chart(df: pd.DataFrame):
    try:
        if df.empty or len(df.columns) < 1: return None
        plot_df = df.copy()
        plot_df.columns = [str(c).replace('_', ' ').title() for c in plot_df.columns]
        
        num_cols = plot_df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = plot_df.select_dtypes(exclude=['number']).columns.tolist()

        if len(cat_cols) >= 1 and len(num_cols) >= 1:
            if len(plot_df) > 30:
                plot_df = plot_df.sort_values(by=num_cols[0], ascending=False).head(10)
                st.toast("⚠️ Chart limited to Top 10 for readability.", icon="📉")
                
            return px.bar(plot_df, x=cat_cols[0], y=num_cols, barmode='group', 
                          title=f"Metrics by {cat_cols[0]}", template="plotly_dark")
            
        elif len(num_cols) >= 2:
            return px.scatter(plot_df, x=num_cols[0], y=num_cols[1], 
                              trendline="ols", title="Metric Correlation Analysis", template="plotly_dark")
            
        elif len(num_cols) == 1:
            return px.histogram(plot_df, x=num_cols[0], 
                                title=f"Distribution of {num_cols[0]}", template="plotly_dark")
            
        elif len(cat_cols) >= 1 and len(num_cols) == 0:
            counts = plot_df[cat_cols[0]].value_counts().reset_index()
            counts.columns = [cat_cols[0], 'Count']
            return px.pie(counts.head(10), names=cat_cols[0], values='Count', 
                          title=f"Distribution of {cat_cols[0]}", template="plotly_dark")
        return None
    except Exception as e:
        logging.warning(f"Chart generation failed: {e}")
        return None

# --- 4. THE FAST LLM ROUTER ---
def get_llm(api_key, model_name, is_local=False):
    if is_local:
        return ChatOllama(model=model_name, temperature=0, num_ctx=4096)
    else:
        return ChatGroq(model=model_name, api_key=api_key, temperature=0)

# --- 5. UI LAYOUT & SIDEBAR ---
st.title("🚀 i2e Intelligent BI Agent")
st.markdown("##### *32M Row Real-Time Analytics | Powered by DuckDB & Local LLMs*")

with st.sidebar:
    st.header("🛠️ Compute Engine")
    exec_mode = st.toggle("Local Mode (Air-Gapped)", value=True)
    if exec_mode:
        selected_model = st.selectbox("Specialized LLM", ["llama3.1:8b", "qwen2.5-coder:7b", "sqlcoder:7b"])
        api_key = None
    else:
        api_key = st.text_input("Groq API Key", type="password")
        selected_model = st.selectbox("Cloud LLM", [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
            "deepseek-r1-distill-llama-70b"
        ])
    
    st.divider()
    
    st.header("💡 Query Ideas")
    if st.button("🛒 Reorder Correlation", help="Analyze if basket position impacts reorder rates across aisles."):
        st.session_state.demo_prompt = "Show me which aisles have the highest reorder rate and how that correlates with average basket position."
        st.rerun()
    if st.button("📈 Department Volume", help="Compare total item volume between Produce and Dairy."):
        st.session_state.demo_prompt = "Compare total product order volume between the produce and dairy eggs departments."
        st.rerun()

    st.divider()
    if st.button("Reset Analysis Session"):
        st.session_state.history = []
        st.rerun()

# --- 6. DIRECT PIPELINE EXECUTION ENGINE ---
if "history" not in st.session_state: st.session_state.history = []

for i, chat in enumerate(st.session_state.history):
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])
        if "df" in chat and chat["df"] is not None: 
            st.dataframe(chat["df"], use_container_width=True)
        if "chart" in chat and chat["chart"] is not None: 
            st.plotly_chart(chat["chart"], key=f"hist_chart_{i}")

if "demo_prompt" in st.session_state:
    user_input = st.session_state.demo_prompt
    del st.session_state.demo_prompt
else:
    user_input = st.chat_input("Ask a complex business question...")

if user_input:
    # --- API KEY VALIDATION ---
    if not exec_mode and not api_key:
        st.error("🔑 Groq API Key is required for Cloud Mode. Please enter it in the sidebar.")
        st.stop()

    st.session_state.history.append({"role": "user", "content": user_input})
    st.chat_message("user").markdown(user_input)
    
    recent_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.history[-5:-1]])
    
    with st.chat_message("assistant"):
        status_box = st.empty()
        status_box.info(f"⏳ Initializing {selected_model} Pipeline...")
        
        start_time = time.time()
        llm = get_llm(api_key, selected_model, is_local=exec_mode)

        max_retries, attempt, success, df_final, query, error_msg = 3, 0, False, None, "", ""

        while attempt < max_retries and not success:
            attempt += 1
            try:
                status_box.info(f"⏳ Step 1/3: Generating SQL (Attempt {attempt}/3)...")
                
                sql_prompt = f"""
                You are an elite Data Engineer writing optimized DuckDB SQL.
                DO NOT wrap the code in markdown blocks. Just return the raw SELECT statement.
                
                GOLD RULE: NEVER return only IDs (like aisle_id). ALWAYS join with 'aisles' or 'departments' 
                to get the text NAME (e.g., a.aisle, d.department) so the chart is readable.

                SCHEMA:
                - aisles (aisle_id, aisle)
                - departments (department_id, department)
                - products (product_id, product_name, aisle_id, department_id)
                - order_products (order_id, product_id, reordered, add_to_cart_order)

                CONVERSATION HISTORY: {recent_history}
                PREVIOUS ERRORS: {error_msg if attempt > 1 else 'None'}
                USER QUESTION: {user_input}
                """
                
                raw_sql = llm.invoke(sql_prompt).content
                query = raw_sql.replace("```sql", "").replace("```", "").strip()

                status_box.info(f"⏳ Step 2/3: Executing Query...")
                with get_analytics_engine().connect() as conn:
                    df_final = pd.read_sql(text(query), conn)
                success = True 

            except Exception as e:
                error_msg = f"ERROR: {str(e)}"
                status_box.warning(f"⚠️ Self-Correcting...")
                time.sleep(1) 

        if not success:
            status_box.error("❌ System Error: Max Retries Exceeded.")
            st.stop()

        try:
            status_box.info("⏳ Step 3/3: Synthesizing Insights...")
            data_sample = df_final.head(10).to_markdown()
            combo_prompt = f"Analyze: {user_input}\nData: {data_sample}\nFormat: 📌 KEY INSIGHT, 💡 WHY IT MATTERS, 🎯 RECOMMENDED ACTION, 🔄 FOLLOW-UPS: [Q1] | [Q2] | [Q3]"
            
            raw_response = llm.invoke(combo_prompt).content
            
            if "🔄 FOLLOW-UPS:" in raw_response:
                parts = raw_response.split("🔄 FOLLOW-UPS:")
                narrative = parts[0].strip()
                suggested_questions = [q.strip() for q in parts[1].strip().split('|') if len(q.strip()) > 5][:3]
            else:
                narrative, suggested_questions = raw_response.strip(), []

            exec_time = time.time() - start_time
            status_box.empty()

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.stop()

        st.markdown("<div class='executive-header'>📊 Data Intelligence Report</div>", unsafe_allow_html=True)
        if narrative: st.info(narrative)
        
        if df_final is not None and not df_final.empty:
            st.write("---")
            col1, col2, col3 = st.columns(3)
            col1.metric("Records Returned", f"{len(df_final):,}")
            col2.metric("Compute Engine", "DuckDB Native")
            col3.metric("Time to Insight", f"{exec_time:.2f}s")
            
            chart_fig = generate_industry_chart(df_final)
            if chart_fig: 
                st.plotly_chart(chart_fig, use_container_width=True, key=f"active_chart_{time.time()}")
            
            with st.expander("📁 View Full Data & Audit Trail"):
                st.dataframe(df_final, use_container_width=True)
                st.code(query, language="sql")
                st.download_button("📥 Download CSV", data=df_final.to_csv(index=False).encode('utf-8'), file_name="analysis.csv", mime="text/csv")
            
            if suggested_questions:
                st.markdown("##### 🔍 Explore Deeper:")
                btn_cols = st.columns(len(suggested_questions))
                for i, question in enumerate(suggested_questions):
                    if btn_cols[i].button(question, key=f"btn_{time.time()}_{i}"):
                        st.session_state.demo_prompt = question
                        st.rerun()

            st.session_state.history.append({"role": "assistant", "content": narrative, "df": df_final, "chart": chart_fig})
        else:
            st.warning("No data found.")
            st.session_state.history.append({"role": "assistant", "content": narrative})