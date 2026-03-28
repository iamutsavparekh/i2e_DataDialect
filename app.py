import solara
import solara.lab
import pandas as pd
import plotly.express as px
import os
import time
import logging
import threading
import re
from sqlalchemy import create_engine, text
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI

# --- 0. STATE MANAGEMENT ---
class State:
    user_query = solara.reactive("")
    exec_mode = solara.reactive(True) 
    api_key = solara.reactive("")
    history = solara.reactive([]) 
    is_loading = solara.reactive(False)
    is_cancelled = solara.reactive(False) # New: Cancellation token
    status_msg = solara.reactive("")
    selected_model = solara.reactive("llama3.1:8b")

# --- 1. DATABASE ENGINE ---
logging.basicConfig(level=logging.INFO)

def get_analytics_engine():
    db_path = os.getenv("DATABASE_URL", "instacart_analytics.duckdb")
    if not os.path.exists(db_path):
        return None
    return create_engine(f"duckdb:///{db_path}", pool_pre_ping=True)

# --- 2. INTELLIGENT CHARTING ---
def generate_industry_chart(df: pd.DataFrame):
    try:
        if df is None or df.empty or len(df.columns) < 1: return None
        plot_df = df.copy()
        plot_df.columns = [str(c).replace('_', ' ').title() for c in plot_df.columns]
        num_cols = plot_df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = plot_df.select_dtypes(exclude=['number']).columns.tolist()
        color_seq = ["#38bdf8", "#818cf8", "#34d399", "#fbbf24", "#f87171"]

        fig = None
        if len(cat_cols) >= 1 and len(num_cols) >= 1:
            if len(plot_df) > 10:
                plot_df = plot_df.sort_values(by=num_cols[0], ascending=False).head(10)
            fig = px.bar(plot_df, x=cat_cols[0], y=num_cols, barmode='group', 
                         title=f"Top 10 Metrics by {cat_cols[0]}", template="plotly_dark", color_discrete_sequence=color_seq)
        elif len(num_cols) >= 2:
            fig = px.scatter(plot_df, x=num_cols[0], y=num_cols[1], title="Metric Correlation", template="plotly_dark", color_discrete_sequence=color_seq)
        
        if fig:
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", color="#cbd5e1"), margin=dict(l=20, r=20, t=40, b=20))
        return fig
    except Exception as e:
        logging.warning(f"Chart generation failed: {e}")
        return None

# --- 3. CORE LOGIC PIPELINE ---

def stop_generation():
    """Interrupts the background thread and resets UI state."""
    State.is_cancelled.value = True
    State.is_loading.value = False
    State.status_msg.value = "Generation Stopped."
    time.sleep(0.5) # Allow thread to catch up
    State.is_cancelled.value = False

def handle_query(query_text=None):
    input_val = query_text if isinstance(query_text, str) else State.user_query.value
    if not input_val or State.is_loading.value: return

    if not State.exec_mode.value and not State.api_key.value:
        State.history.value = State.history.value[-19:] + [{"role": "assistant", "content": "⚠️ API Key required."}]
        return 

    start_time_stamp = time.time()
    State.is_loading.value = True
    State.is_cancelled.value = False
    State.status_msg.value = "Analyzing Intent..."
    
    recent_history = "\n".join([f"{msg['role'].capitalize()}: {msg['content'][:150]}" for msg in State.history.value[-4:]])
    State.history.value = State.history.value[-19:] + [{"role": "user", "content": input_val}]
    State.user_query.value = "" 

    def process_in_background():
        try:
            primary_llm = None
            backup_llm = None
            
            if State.exec_mode.value:
                primary_llm = ChatOllama(model=State.selected_model.value, temperature=0, timeout=30)
            else:
                if "gemini" in State.selected_model.value:
                    primary_llm = ChatGoogleGenerativeAI(model=State.selected_model.value, google_api_key=State.api_key.value, temperature=0)
                else:
                    primary_llm = ChatGroq(model=State.selected_model.value, api_key=State.api_key.value, temperature=0)
                backup_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=State.api_key.value, temperature=0)

            def safe_invoke(prompt_text, step_name):
                # CHECK FOR CANCELLATION
                if State.is_cancelled.value: return "CANCELLED"
                try:
                    return primary_llm.invoke(prompt_text).content
                except Exception as e:
                    err = str(e).lower()
                    if any(x in err for x in ["429", "resource_exhausted", "ratelimit", "timeout"]) and backup_llm:
                        return backup_llm.invoke(prompt_text).content
                    raise e

            # STEP 1: ROUTING
            router_p = f"DATABASE or CHAT? User: {input_val}"
            intent_raw = safe_invoke(router_p, "Routing")
            if intent_raw == "CANCELLED": return

            if "DATABASE" not in intent_raw.upper():
                State.status_msg.value = "Thinking..."
                res = safe_invoke(f"Context: {recent_history}\nUser: {input_val}", "Chat")
                if res != "CANCELLED":
                    State.history.value = State.history.value[-19:] + [{"role": "assistant", "content": res, "time": time.time() - start_time_stamp}]
                State.is_loading.value = False
                return

            # STEP 2: DB PIPELINE
            max_retries, attempt, success, df_final, query_str, error_msg = 3, 0, False, None, "", ""
            engine = get_analytics_engine()
            
            while attempt < max_retries and not success and not State.is_cancelled.value:
                attempt += 1
                try:
                    State.status_msg.value = f"SQL Logic (Attempt {attempt}/3)..."
                    sql_p = f"Return raw DuckDB SQL. Schema: aisles, departments, products, order_products, orders. User: {input_val}"
                    raw_sql = safe_invoke(sql_p, "SQL")
                    if raw_sql == "CANCELLED": return
                    query_str = raw_sql.replace("```sql", "").replace("```", "").strip()

                    if not query_str.lower().startswith("select"): raise Exception("Only SELECT allowed.")

                    State.status_msg.value = "Querying Engine..."
                    with engine.connect() as conn:
                        df_final = pd.read_sql(text(query_str), conn)
                    success = True 
                except Exception as e:
                    error_msg = str(e)
                    time.sleep(1) 

            if success and not State.is_cancelled.value:
                State.status_msg.value = "Synthesizing..."
                data_sample = df_final.head(20).to_markdown()
                combo_p = f"Analyze: {input_val}\nData: {data_sample}\nFormat: 📌 KEY INSIGHT, 💡 WHY IT MATTERS, 🎯 RECOMMENDED ACTION, 🔄 FOLLOW-UPS: [Q1] | [Q2] | [Q3]"
                raw_response = safe_invoke(combo_p, "Synthesis")
                if raw_response == "CANCELLED": return
                
                narrative = raw_response
                suggested_questions = []
                parts = re.split(r'(?i)🔄?\s*follow-ups?\s*[:\-]*', raw_response)
                if len(parts) > 1:
                    narrative = parts[0].strip()
                    raw_qs = re.split(r'[|\n]', parts[1])
                    for q in raw_qs:
                        clean_q = re.sub(r'^(?:\[?Q\d\]?|\d+[\.\)]|[\-\*\u2022])\s*', '', q.strip(), flags=re.IGNORECASE)
                        if len(clean_q) > 8: suggested_questions.append(clean_q)

                State.history.value = State.history.value[-19:] + [{
                    "role": "assistant", "content": narrative, "df": df_final.head(100),
                    "chart": generate_industry_chart(df_final), "sql": query_str,
                    "follow_ups": suggested_questions[:3], "time": time.time() - start_time_stamp
                }]
            elif not State.is_cancelled.value:
                State.history.value = State.history.value[-19:] + [{"role": "assistant", "content": f"❌ Error: {error_msg}"}]
        
        except Exception as e:
            if not State.is_cancelled.value:
                State.history.value = State.history.value[-19:] + [{"role": "assistant", "content": f"❌ Pipeline Exception: {str(e)}"}]
        
        State.is_loading.value = False
        State.status_msg.value = ""

    threading.Thread(target=process_in_background, daemon=True).start()

# --- 4. UI COMPONENTS ---

@solara.component
def SidebarContent():
    with solara.Column(style={"padding": "20px", "height": "100%", "display": "flex", "flex-direction": "column"}):
        solara.Markdown("### ⚙️ Engine Config", style={"color": "#94a3b8", "font-size": "0.85rem", "text-transform": "uppercase"})
        solara.Checkbox(label="Local Mode (Air-gapped)", value=State.exec_mode, disabled=State.is_loading.value)
        
        if not State.exec_mode.value:
            solara.InputText("Cloud API Key", value=State.api_key, password=True, disabled=State.is_loading.value)
            solara.Select(label="Cloud Model", value=State.selected_model, values=["gemini-2.5-flash", "llama-3.3-70b-versatile"], disabled=State.is_loading.value)
        else:
            solara.Select(label="Local Model", value=State.selected_model, values=["llama3.1:8b", "qwen2.5-coder:7b"], disabled=State.is_loading.value)
        
        solara.v.Divider(style_="margin: 20px 0; border-color: #1e293b;")
        
        solara.Markdown("### 💡 Guided Queries", style={"color": "#94a3b8", "font-size": "0.85rem", "text-transform": "uppercase"})
        solara.Button("🛒 Reorder Analysis", on_click=lambda: handle_query("Aisles with highest reorder rates."), color="primary", block=True, style={"margin-bottom": "10px", "border-radius": "8px"}, disabled=State.is_loading.value)
        solara.Button("📈 Volume Drill-down", on_click=lambda: handle_query("Compare volume of Produce vs Dairy."), color="primary", block=True, style={"border-radius": "8px"}, disabled=State.is_loading.value)
        
        solara.v.Spacer() 
        solara.Button("🗑️ Clear Session", on_click=lambda: State.history.set([]), text=True, style={"color": "#ef4444", "width": "100%"}, disabled=State.is_loading.value)

@solara.component
def ChatThread():
    with solara.Column(style={"flex-grow": "1", "overflow-y": "auto", "padding": "20px"}):
        if not State.history.value:
            with solara.Column(align="center", style={"margin-top": "15vh", "opacity": "0.4"}):
                solara.Markdown("## DataDialect Intelligence")
                solara.Markdown("Monitoring real-time analytics stream from DuckDB.")

        for item in State.history.value:
            is_user = item["role"] == "user"
            bg = "#2563eb" if is_user else "#111827"
            radius = "16px 16px 4px 16px" if is_user else "16px 16px 16px 4px"
            
            with solara.Row(justify="end" if is_user else "start", style={"margin-bottom": "20px"}):
                with solara.Column(style={"max-width": "85%", "background-color": bg, "border": "1px solid #1e293b" if not is_user else "none", "border-radius": radius, "padding": "18px"}):
                    solara.Markdown(item["content"])
                    if not is_user:
                        if "df" in item: 
                            with solara.Details("📁 Dataset"): solara.DataFrame(item["df"])
                        if "chart" in item and item["chart"]: solara.FigurePlotly(item["chart"])
                        if "sql" in item:
                            with solara.Details("🔍 SQL Audit"): solara.Markdown(f"```sql\n{item['sql']}\n```")
                        if "time" in item:
                            solara.Markdown(f"⏱️ {item['time']:.2f}s", style={"font-size": "0.7rem", "color": "#94a3b8", "margin-top": "8px"})

                        if "follow_ups" in item and item["follow_ups"]:
                            solara.v.Divider(style_="margin: 15px 0; border-color: #334155;")
                            with solara.Row(style={"flex-wrap": "wrap", "gap": "8px"}):
                                for q in item["follow_ups"]:
                                    solara.Button(label=q, on_click=lambda q=q: handle_query(q), outlined=True, style={"text-transform": "none", "font-size": "0.8rem", "height": "auto", "min-height": "32px", "padding": "6px 12px", "border-radius": "6px"}, disabled=State.is_loading.value)

@solara.component
def Page():
    solara.lab.theme.dark = True
    solara.Style("""
        html, body, .v-application { font-family: 'Inter', sans-serif !important; background-color: #0B0F19 !important; }
        .v-navigation-drawer { background-color: #0B0F19 !important; border-right: 1px solid #1e293b !important; }
        header.v-app-bar { background-color: #0B0F19 !important; border-bottom: 1px solid #1e293b !important; box-shadow: none !important; }
        .gradient-title { background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; font-size: 1.5rem; }
    """)

    with solara.AppLayout(color="#0B0F19"):
        with solara.Sidebar(): SidebarContent()
        
        with solara.Column(style={"height": "100%", "display": "flex", "flex-direction": "column", "overflow": "hidden"}):
            with solara.Row(justify="center", style={"padding": "15px", "border-bottom": "1px solid #1e293b", "flex-shrink": "0"}):
                solara.HTML(tag="h1", unsafe_innerHTML="<span class='gradient-title'>DataDialect Agent</span>")
            
            ChatThread()

            with solara.Column(style={"padding": "20px 30px", "border-top": "1px solid #1e293b", "flex-shrink": "0"}):
                if State.is_loading.value: 
                    with solara.Row(style={"align-items": "center", "margin-bottom": "10px"}):
                        solara.ProgressLinear(True, style={"flex-grow": "1", "margin-right": "15px", "color": "#3B82F6"})
                        solara.Text(State.status_msg.value, style={"color": "#3B82F6", "font-weight": "600", "font-size": "0.85rem"})
                
                # --- FLEX ROW FOR INPUT + STOP BUTTON ---
                with solara.Row(style={"align-items": "center", "gap": "10px"}):
                    solara.InputText(
                        "Analyze your data...", 
                        value=State.user_query, 
                        on_value=handle_query, 
                        disabled=State.is_loading.value,
                        style={"flex-grow": "1"}
                    )
                    if State.is_loading.value:
                        solara.Button(
                            "Stop", 
                            icon_name="mdi-stop-circle-outline",
                            on_click=stop_generation, 
                            color="error", 
                            outlined=True,
                            style={"height": "48px", "border-radius": "8px", "text-transform": "none"}
                        )