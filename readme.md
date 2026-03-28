# 🚀 DataDialect: Enterprise Intelligent BI Agent

**DataDialect** is a high-performance Business Intelligence (BI) agent built for the Instacart Analytics dataset. It translates natural language business questions into optimized DuckDB SQL, executes them securely, and synthesizes data into executive-ready insights and interactive visualizations.

## ✨ Advanced Features

- **Hybrid LLM Architecture:** Seamlessly switch between **Local Execution** (Ollama/Llama 3.1) for total data privacy and **Cloud Execution** (Groq/Gemini) for high-speed reasoning.
- **Self-Healing SQL Engine:** A robust 3-stage retry pipeline that captures database execution errors and feeds them back to the LLM for real-time logic adjustment.
- **Failover Protection:** Automatically detects rate limits or timeouts in primary models and triggers an **Instant Failover** to Gemini 2.5 Flash to ensure zero-downtime performance.
- **Hardware-Optimized Data Layer:** Utilizes DuckDB with B-Tree indexing and hardware-accelerated pragmas for lightning-fast joins on millions of rows.
- **Production UX:** Includes a "Stop" button, real-time "Time-to-Insight" (TTI) metrics, and a polished glassmorphic UI.

## 🛠️ Tech Stack

- **UI Framework:** [Solara](https://solara.dev/) (Reactive Python Framework)
- **Database:** [DuckDB](https://duckdb.org/) (High-performance OLAP engine)
- **Orchestration:** [LangChain](https://www.langchain.com/)
- **Visuals:** [Plotly](https://plotly.com/python/)
- **LLM Connectors:** Groq (Llama 3.3), Google Generative AI (Gemini 2.5), Ollama

## 🚀 Setup & Installation

### 1. Install Packages
```bash
pip install -r requirements.txt
```
### 2. Database Setup (Ingestion & Optimization)
Before running the agent, you must ingest the Instacart CSV files into a high-performance DuckDB file. Place your CSV files (aisles.csv, departments.csv, products.csv, orders.csv, and order_products__prior.csv) in the root directory and run:
```Bash
python 1_database_setup.py
```
This script optimizes hardware threads, allocates memory limits, and builds B-Tree indexes on Foreign Keys to ensure the LLM-generated queries execute in milliseconds.

### 3. Setup Local LLM (Optional)
If using Local Mode, ensure Ollama is installed and the model is pulled:
```Bash
ollama run llama3.1:8b
```

### 4. Launch the Agent
```Bash
solara run app.py
```

The agent will be live at: http://localhost:8765
## 🧠 Key Design Decisions & Why

### 🔹 Intent-Based Routing (Gatekeeper LLM)
Classifies queries as:
- `DATABASE`
- `CHAT`

**Why?**  
Prevents the SQL engine from attempting to execute queries on conversational inputs (e.g., "Hello"), reducing API costs and avoiding system crashes.

---

### 🔹 Multi-LLM Failover (Safe Invoke)
The system defaults to a primary model but automatically switches to **Gemini Flash** when encountering:
- Rate limits (429 errors)  
- Timeouts  

**Why?**  
Ensures high availability and keeps the **Time-to-Insight (TTI)** consistent even under heavy load.

---

### 🔹 Hardware-Optimized Ingestion
In `1_database_setup.py`, the system explicitly sets:
- `PRAGMA threads`
- `PRAGMA memory_limit`

**Why?**  
DuckDB performs best when it has clear boundaries on hardware usage, enabling efficient coexistence with LLM workloads.

---

### 🔹 Regex-Based Synthesis Parsing
Uses case-insensitive regex to extract follow-up questions from LLM responses.

**Why?**  
LLMs often produce inconsistent formats (e.g., `[Q1]` vs `1.`). Regex ensures UI components remain stable and do not break.

---

## ⚠️ Known Limitations & Failure Modes

### ❗ Complex Multi-Join Hallucinations
While the 3-stage retry loop resolves most SQL issues, highly complex queries involving 4+ joins may still produce incorrect logic.

---

### ❗ Schema Bottleneck
The agent assumes a static schema. If new tables are added to DuckDB without updating the prompt’s schema description, they will not be recognized.

---

### ❗ Stateless Visualization
Chart selection is heuristic-based. In some cases, a less optimal visualization (e.g., pie chart instead of bar chart) may be chosen.

---

### ❗ Cold Start (Local Mode)
When using Ollama locally, the first query may be slower due to model loading into system VRAM.
