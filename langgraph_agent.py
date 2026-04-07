import time
import os
import json
from typing import Annotated, List, TypedDict, Literal
from langgraph.graph import END, StateGraph
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_chroma import Chroma
from termcolor import colored
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import get_embeddings, get_llm, DATA_FOLDER, DB_FOLDER, FILES

# Generic Retry Logic (Provider agnostic)
retry_logic = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception)
)


def initialize_vector_dbs():
    embeddings = get_embeddings()
    retrievers = {}
    
    for key in FILES.keys():
        persist_dir = os.path.join(DB_FOLDER, key)

        if os.path.exists(persist_dir):
            vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
            retrievers[key] = vectorstore.as_retriever(search_kwargs={"k": 5})
        else:
            print(colored(f"❌ Error: Database for '{key}' not found!", "red"))
            print(colored(f"⚠️ Please run 'python build_rag.py' first.", "yellow"))
            continue
    
    return retrievers

RETRIEVERS = initialize_vector_dbs()


class AgentState(TypedDict):
    question: str
    documents: List[any] # <--- 改成 List，用來存放檢索到的 Document 物件
    generation: str
    search_count: int
    needs_rewrite: str
    thought: str  # 新增：用來存儲 AI 的思考過程



# Task B: Router 階段 (對應 Thought/Action)
@retry_logic
def retrieve_node(state: AgentState):
    print(colored(f"Question: {state['question']}", "blue"))
    question = state["question"]
    llm = get_llm()
    
    # --- [START] Improved Routing Logic ---
    options = list(FILES.keys()) + ["both", "none"]
    router_prompt = f"""
        You are a financial analyst. Analyze the user question.

        1. First, think about what info you need and which company's database to check.
        2. Then, choose one datasource from: {options}.
        

        Output ONLY valid JSON:
        {{
            "thought": "Your reasoning here in English...",
            "datasource": "..."
        }}
        User Question: {question}
    """
    
    # 預設值，防止解析失敗時程式崩潰
    thought = "Analyzing the query to determine the best data source."
    target = "both"

    try:
        response = llm.invoke(router_prompt)
        content = response.content.strip()
        # Handle cases where LLM might wrap JSON in backticks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        res_json = json.loads(content)
        # 從 JSON 中提取值
        thought = res_json.get("thought", thought)
        target = res_json.get("datasource", "both")

    except Exception as e:
        # 先印出錯誤，再給予明確的字串賦值
        print(colored(f"⚠️ Router parsing failed, defaulting to 'both'. Error: {e}", "red"))
        thought = f"Defaulting to broad search for: {question}"
        target = "both"
    
    
    # 按照作業格式輸出
    print(colored(f"Thought: {thought}", "yellow"))
    print(colored(f"Action: search_{target}_financials", "cyan"))
    print(colored(f"Action Input: {question}", "cyan"))

    # --- [END] ---

    docs_content = [] # 改用列表儲存
    targets_to_search = []
    if target == "both":
        targets_to_search = list(FILES.keys())
    elif target in FILES:
        targets_to_search = [target]
    
    for t in targets_to_search:
        if t in RETRIEVERS:
            docs = RETRIEVERS[t].invoke(question)
            source_name = t.capitalize()
            docs_content.extend(docs) # 收集 Document 物件

    return {"documents": docs_content, "search_count": state["search_count"] + 1}

# Task C: Grader 階段 (對應 Observation 處理)
@retry_logic
def grade_documents_node(state: AgentState): 
    print(colored("Observation: Retrieved documents from vector store.", "magenta"))
    question = state["question"]
    documents = state["documents"]
    
    filtered_docs = []
    needs_rewrite = "no" # 預設不需要重寫

    llm = get_llm() # LLM 放在迴圈外，不用每次都初始化

    for d in documents:
        time.sleep(5)  # API Rate Limit 考量，避免一次性送出太多請求
        system_prompt = """You are a grader assessing relevance. 
        Does the retrieved document contain information related to the user question?
        Answer with ONLY one word: 'yes' or 'no'."""
        
        # 注意：這裡傳入的是 d.page_content (單個段落)，不是 documents (全體)
        msg = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Document context: {d.page_content} \n\n Question: {question}")
        ]
        
        try:
            response = llm.invoke(msg)
            grade = response.content.strip().lower()
            
            if "yes" in grade:
                print(colored("  ✅ Relevant", "green"))
                filtered_docs.append(d)
            else:
                print(colored("  ❌ Irrelevant", "red"))
        except Exception as e:
            print(colored(f"⚠️ Grading failed: {e}", "red"))

    # 如果最後一個有用的文件都沒有，就標記需要重寫
    if not filtered_docs:
        needs_rewrite = "yes"

    # 最後才回傳結果，更新 state
    return {"documents": filtered_docs, "needs_rewrite": needs_rewrite}

# Task E: Generator 階段 (對應 Final Answer)
@retry_logic
def generate_node(state: AgentState):
    print(colored("Thought: I have sufficient information to answer.", "yellow"))
    
    question = state["question"]
    documents = state["documents"]
    llm = get_llm() 

    # 強化版 System Prompt，直接將指令寫死在 Template 中
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a financial analyst. 
        STRICT REQUIREMENT: YOUR RESPONSE MUST BE ENTIRELY IN ENGLISH.
        Even if the user asks in Chinese, you MUST answer in English.
        
        Use the provided context to answer. 
        1. Distinguish between 2024, 2023, and 2022 data.
        2. If the answer is not found, say 'I don't know'.
        3. ALWAYS cite the source, e.g., [Source: Apple 10-K]."""),
        ("human", "User Question: {question}\n\nContext:\n{context}"),
    ])  
    
    chain = prompt | llm
    response = chain.invoke({"context": str(documents), "question": question})
    return {"generation": response.content}

@retry_logic
def rewrite_node(state: AgentState): 
    print(colored("--- 🔄 REWRITING QUERY ---", "red"))
    question = state["question"]
    llm = get_llm()
    
    msg = [ 
        HumanMessage(content=f"The previous search for '{question}' yielded irrelevant results. \n"
                             f"Please rephrase this question to be more specific or use better keywords for a financial search engine. \n"
                             f"Output ONLY the new question text.")
    ]
    response = llm.invoke(msg)
    new_query = response.content.strip()
    print(f"   New Question: {new_query}")
    return {"question": new_query}

def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("rewrite", rewrite_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")

    def decide_to_generate(state):
        # 如果 needs_rewrite 是 "yes"，應該要去 "rewrite" 節點，而不是 "generate"
        if state["needs_rewrite"] == "yes":
            if state["search_count"] > 2: 
                print("   (Max retries reached, generating anyway...)")
                return "generate"
            return "rewrite"
        else:
            return "generate" # 資料合格，去生成答案
        
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "generate": "generate",
            "rewrite": "rewrite"
        },
    )

    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("generate", END)

    return workflow.compile()

def run_graph_agent(question: str):
    app = build_graph()
    inputs = {"question": question, "search_count": 0, "needs_rewrite": "no", "documents": "", "generation": ""}
    # Using stream to see progress if needed, but invoke is fine for simple return
    result = app.invoke(inputs)
    return result["generation"]

# --- Legacy ReAct Agent ---
def run_legacy_agent(question: str):
    print(colored("--- 🤖 RUNNING LEGACY AGENT (ReAct) ---", "magenta"))
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain.tools.retriever import create_retriever_tool
    from langchain.tools.render import render_text_description

    tools = []
    for key, retriever in RETRIEVERS.items():
        tools.append(create_retriever_tool(
            retriever, 
            f"search_{key}_financials", 
            f"Searches {key.capitalize()}'s financial data."
        ))

    if not tools:
        return "System Error: No tools available."

    llm = get_llm()

    template = """Answer the following questions as best you can. 
    You have access to the following tools:
    {tools}

    Use the following format:
    Question: {input}
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question in English.

    Constraints:
    - Respond ONLY in English.
    - Distinguish between 2024, 2023, and 2022 data.
    - If you cannot find the exact 2024 figure, say "I don't know"[cite: 34].

    Question: {input}
    Thought: {agent_scratchpad}"""

    prompt = PromptTemplate.from_template(template)
    prompt = prompt.partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools])
    )

    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True,
        max_iterations=5
    )

    try:
        result = agent_executor.invoke({"input": question})
        return result["output"]
    except Exception as e:
        return f"Legacy Agent Error: {e}"
    
# 在 langgraph_agent.py 最底下加入
if __name__ == "__main__":
    response = run_graph_agent("Apple 2024 年的總營收是多少？")
    print(response)