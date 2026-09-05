# Agentic RAG – Multi-Hop Question Answering

A from-scratch **Agentic Retrieval-Augmented Generation (RAG) system** for answering multi-hop questions over a bounded collection of documents.

Unlike a conventional RAG pipeline that retrieves documents once and then generates an answer, this system uses an explicit **ReAct (Reason + Act) loop**. The agent decides what to search, observes the retrieved evidence, maintains memory of previous steps, reformulates queries when necessary, and decides when to finish.

The system was built without LangChain, LlamaIndex, CrewAI, or other pre-built agent/RAG frameworks, with the core retrieval, tool use, action parsing, memory, and agent-loop logic implemented directly.

---

## 🏆 Result

### Professor's Unseen Evaluation

**93.34% answer correctness**

- Evaluated on an unseen test set provided by the professor
- **Highest score among 3 participating teams**
- The reported 93.34% represents **answer correctness** and is not being presented as Exact Match (EM), since the exact evaluation procedure used for this score is not known.

---

## 🎯 Problem Statement

The goal of the project is to build an agentic RAG system capable of answering **multi-hop questions** by iteratively retrieving and reasoning over a small document pool.

Each question is accompanied by **10 candidate paragraphs**, containing relevant documents as well as distractors. The agent must answer using only information retrieved from these documents rather than relying on the full context, pretrained knowledge, or the open web.

The project specifically requires the system to implement the underlying agent and retrieval mechanisms from scratch:

- Thought → Action → Observation → Finish loop
- Parsing raw LLM output into structured actions
- A simple tool registry
- TF-IDF + cosine-similarity retrieval
- Stopping conditions and loop protection
- Full trajectory logging
- Evaluation of retrieval and answer quality

---

## 🧠 System Architecture

```text
                         User Question
                               │
                               ▼
                    ┌─────────────────────┐
                    │     ReAct Agent     │
                    │                     │
                    │  Thought → Action   │
                    └──────────┬──────────┘
                               │
                     ┌─────────┴─────────┐
                     │                   │
                  Search               Finish
                     │                   │
                     ▼                   ▼
              ┌─────────────┐       Final Answer
              │ Search Tool │
              └──────┬──────┘
                     │
                     ▼
             ┌─────────────────┐
             │  TF-IDF         │
             │  Retriever      │
             └────────┬────────┘
                      │
               Cosine Similarity
                      │
                      ▼
                 Top-1 Document
                      │
                      ▼
                 Observation
                      │
                      ▼
                    Memory
                      │
                      ▼
                 ReAct Agent
                      │
                      └──────► Repeat
```

The system follows an explicit iterative ReAct workflow rather than using a fixed single retrieval step.

---

## 🔄 ReAct Agent Loop

The agent has two available actions:

```text
Search(query)
Finish(answer)
```

At each iteration, the LLM generates a thought and chooses an action.

### Search

```text
Thought: I need information about X.
Action: Search(X)
```

The search tool retrieves the highest-ranked document and returns it as the observation.

The observation is then stored in memory and made available to the agent for the next reasoning step.

### Multi-Hop Iteration

```text
Question
   ↓
Thought
   ↓
Search(query)
   ↓
Observation
   ↓
Memory
   ↓
Thought
   ↓
Search(reformulated query)
   ↓
Observation
   ↓
...
```

The agent can reformulate subsequent queries based on previously retrieved information instead of being forced to execute a fixed number of retrieval steps.

### Finish

When the agent determines that it has sufficient evidence:

```text
Thought: I have enough information to answer.
Action: Finish(answer)
```

The search process is bounded to prevent uncontrolled execution.

---

## 🔎 Retrieval

The retriever is implemented from scratch using:

- **TF-IDF (`TfidfVectorizer`)**
- **Cosine similarity**
- **Top-1 retrieval**

Each candidate document is represented by concatenating its title and associated sentence:

```text
Title: Sentence
```

For a search query:

1. The query is vectorized using TF-IDF.
2. Cosine similarity is calculated between the query and candidate documents.
3. Documents are ranked by similarity.
4. The **Top-1** document is returned as the observation.

This keeps the retrieval mechanism simple and transparent while allowing the ReAct agent to perform iterative searches.

---

## 🧰 Tool Use

The agent interacts with the retriever through a search tool:

```text
Search(query)
```

Only the information returned by the search tool is supplied as retrieved evidence to the agent. The full collection of candidate documents is not directly exposed to the LLM.

This forces the agent to decide **what information it needs to retrieve next**.

---

## 🧠 Agent Memory

The system maintains memory of previous search steps.

Each step records:

```text
Action
Query
Observation
```

This history enables the agent to:

- Use information obtained from earlier searches
- Reformulate queries
- Avoid repeating unsuccessful searches
- Perform multi-hop reasoning across retrieved evidence

---

## 🛡️ Loop Guard

The ReAct controller includes bounded execution to prevent an agent from searching indefinitely.

The system uses a maximum search budget per question and tracks previous search activity as part of its loop-control logic.

This provides a practical stopping mechanism while allowing the agent to decide when it has enough information to finish.

---

## 🤖 LLM

The system uses:

| Component | Choice |
|---|---|
| LLM | Qwen3-32B |
| API | Groq |
| Temperature | 0 |

The prompt is designed to encourage:

- Retrieval before answering
- Use of retrieved observations as evidence
- No reliance on external knowledge
- Query reformulation when retrieval is unsuccessful
- Use of previous search history
- Finishing only after sufficient evidence has been retrieved

---

## 📊 Evaluation

The project evaluation includes the following metrics:

### 1. Exact Match (EM)

Measures whether the generated answer matches the expected answer after normalization.

### 2. Retrieval Recall

Measures whether at least one gold supporting paragraph was retrieved at any point during the agent trajectory.

### 3. Average Steps-to-Answer

Measures the mean number of `Search` calls made before the agent produces `Finish`.

### 4. Premature Stops

Among incorrect answers, identifies whether the agent stopped without ever retrieving a gold supporting paragraph or stopped despite having already retrieved the correct evidence.

---

## 📁 Repository Structure

```text
agentic-rag/
│
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
│
├── notebooks/
│   ├── 01_Dataset.ipynb
│   ├── 02_Retriever.ipynb
│   ├── 03_Prompt_and_LLM_Integration.ipynb
│   ├── 04_ReAct_Loop.ipynb
│   ├── 05_Metrics_and_Logger.ipynb
│   └── 06_Main_Pipeline.ipynb
│
├── data/
│   └── ...
│
├── results/
│   ├── evaluation_metrics.csv
│   └── agent_trajectories.txt
│
└── Agentic_RAG_Project_Report.pdf
```

The notebooks document the development and experimentation process, while the modular Python files provide a cleaner representation of the implementation.

---

## 🛠️ Tech Stack

- **Python**
- **Qwen3-32B**
- **Groq API**
- **scikit-learn**
- **TF-IDF**
- **Cosine Similarity**
- **NumPy**
- **Pandas**
- **Jupyter Notebook / Google Colab**

No pre-built agent or RAG framework was used.

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd agentic-rag
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure the API key

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key_here
```

**Do not commit `.env` to GitHub.**

Use `.env.example` as the template for required environment variables.

---

## ▶️ Running the Project

The complete pipeline can be explored through:

```text
notebooks/06_Main_Pipeline.ipynb
```

For understanding the individual components, the notebooks can be followed in this order:

```text
01 → Dataset
02 → Retriever
03 → Prompt & LLM Integration
04 → ReAct Loop
05 → Metrics & Logger
06 → Main Pipeline
```

---

## 📈 Development & Error Analysis

During development, the system demonstrated both retrieval failures and retrieval recovery.

In some cases, the initial TF-IDF retrieval returned an irrelevant document. The agent could then reformulate the query and perform another search, allowing it to recover the relevant evidence.

The development analysis highlighted retrieval quality as an important factor because the system deliberately uses **Top-1 retrieval** and a bounded search budget.

---

## 🔮 Future Improvements

Potential improvements include:

- Replace TF-IDF with a semantic retriever such as Sentence Transformers
- Compare TF-IDF against embedding-based retrieval
- Introduce Top-k retrieval
- Add a reranking stage
- Improve query reformulation
- Strengthen stopping and evidence-verification criteria

---

## 👤 My Contributions

My primary contributions focused on the **agent control layer**:

- Implemented the **ReAct agent loop**
- Implemented **tool/action parsing** from raw LLM output
- Implemented **agent memory** for maintaining previous queries and observations
- Implemented the **loop guard** and bounded search behavior
- Worked on integrating retrieved observations into subsequent reasoning iterations

---

## 📄 Documentation

The project report contains details about the retrieval and prompt design, development-time failure analysis, and potential future improvements.

---

## 🏁 Outcome

This project provided a from-scratch implementation of the core mechanics behind an agentic RAG system:

**retrieval → tool use → reasoning → memory → query reformulation → bounded agent execution → final answer**

The system achieved **93.34% answer correctness on the professor's unseen evaluation**, ranking **first among the three teams**.
