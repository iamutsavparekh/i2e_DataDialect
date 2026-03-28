# 🚀 DataDialect: Intelligent BI Agent

**DataDialect** is an enterprise-grade Business Intelligence (BI) agent designed to interact with the Instacart Analytics dataset. It allows users to query complex relational data using natural language, providing real-time SQL generation, interactive data visualizations, and automated business insights.

## ✨ Key Features

- **Hybrid LLM Architecture:** Seamlessly switch between **Local Execution** (via Ollama) for high privacy and **Cloud Execution** (via Groq or Gemini) for high performance.
- **Intelligent Intent Routing:** A dedicated classification layer that distinguishes between conversational chat ("Hi, how are you?") and data-heavy analytical queries ("What are my top 5 aisles?").
- **Self-Healing SQL Pipeline:** A 3-stage retry loop that captures database execution errors and feeds them back to the LLM to self-correct SQL syntax or logic hallucinations.
- **Failover & High Availability:** Automatically detects API rate limits (429 errors) or timeouts in primary models and silently fails over to **Gemini 2.5 Flash** to ensure a 100% success rate during live demos.
- **Enterprise Guardrails:** Hardened security layer that validates all LLM-generated strings to ensure only `SELECT` operations are executed, protecting against accidental data mutation.
- **Interactive UX:** - Glassmorphism UI with a real-time status tracker.
  - **Time-to-Insight (TTI):** Live benchmarking of query performance.
  - **Execution Control:** A "Stop" button to interrupt long-running background generations.
  - **Dynamic Follow-ups:** Context-aware suggested queries to guide user exploration.

## 🛠️ Tech Stack

- **UI Framework:** [Solara](https://solara.dev/) (Reactive Python Web Framework)
- **Database:** [DuckDB](https://duckdb.org/) (In-process OLAP database)
- **Orchestration:** [LangChain](https://www.langchain.com/)
- **Visuals:** [Plotly](https://plotly.com/python/)
- **Models Supported:** - **Cloud:** Google Gemini 2.5, Groq (Llama 3.3 / Mixtral)
  - **Local:** Ollama (Llama 3.1, Qwen 2.5 Coder)

## 🚀 Installation & Setup

### 1. Clone the Repository
Ensure your DuckDB file (`instacart_analytics.duckdb`) is in the project root.

### 2. Install Dependencies
```bash
pip install solara solara-lab pandas plotly sqlalchemy duckdb-engine langchain-groq langchain-ollama langchain-google-genai anywidget tabulate

### 3. Setup Local LLM (Optional)
```bash
ollama run llama3.1:8b

### 4. Launch the Agent
```bash
solara run app.py


# The agent will be live at: http://localhost:8765