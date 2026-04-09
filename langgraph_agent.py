import time
import os
import json
from typing import Annotated, List, TypedDict, Literal
from langchain_core import documents
from langchain_core.runnables import chain
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
    # 新增：存放針對個別公司的查詢詞
    apple_query: str 
    tesla_query: str


# Task B: Router 階段 (對應 Thought/Action)
def retrieve_node(state: AgentState):
    print(colored(f"Question: {state['question']}", "blue"))
    question = state["question"]
    llm = get_llm()
    
    # 1. 初始化變數
    docs_content = []
    thought = "Analyzing the query to determine the best data source."
    target = "both" # 預設值
    q_apple = None
    q_tesla = None

    # 2. 讓 LLM 進行路由分類 (Task B)
    options = ["apple", "tesla", "both", "none"]
    router_prompt = f"""
        You are a financial analyst. Analyze this financial question: "{question}"
        Choose one datasource from: {options}.
        
        If the question involves comparison or both companies, set datasource to "both" 
        and generate two targeted English search queries for our vector database.

        Output ONLY valid JSON:
        {{
            "thought": "Your reasoning in English...",
            "datasource": "apple" | "tesla" | "both" | "none"
            "apple_query": "Specific English query for Apple's 10-K (or null)",
            "tesla_query": "Specific English query for Tesla's 10-K (or null)"
        }}
        User Question: {question}
    """

    try:
        response = llm.invoke(router_prompt)
        content = response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        res_json = json.loads(content)
        thought = res_json.get("thought", thought)
        target = res_json.get("datasource", "both")

        # 這裡從 JSON 抓取 LLM 自己想出來的關鍵字
        q_apple = res_json.get("apple_query") 
        q_tesla = res_json.get("tesla_query")

    except Exception as e:
        target = "both"

    # 3. 按照助教格式列印 Log (修正您的 if 語法)
    print(colored(f"Thought: {thought}", "yellow"))
    print(colored(f"Action: search_{target}_financials", "cyan"))

    # --- 關鍵修正區 ---
    if target == "both" and q_apple and q_tesla:
        # 只有在兩者都存在時，印出合併的 Query，並執行拆分檢索
        print(colored(f"Action Input: {q_apple} | {q_tesla}", "cyan"))
        
        apple_docs = RETRIEVERS["apple"].invoke(q_apple) if "apple" in RETRIEVERS else []
        tesla_docs = RETRIEVERS["tesla"].invoke(q_tesla) if "tesla" in RETRIEVERS else []
        
        # Fallback：用各自財報的正確術語
        apple_fallback = RETRIEVERS["apple"].invoke(
            "Total net sales Total cost of sales gross margin 2024 Statement of Operations"
        )
        tesla_fallback = RETRIEVERS["tesla"].invoke(
            "Total revenues Total cost of revenues gross profit 2024 consolidated"
            # ^^^ 關鍵：Tesla 用 revenues/cost of revenues，要加 consolidated 避免撈到分部數字
        )
        
        docs_content = apple_docs + tesla_docs + apple_fallback + tesla_fallback
    elif target in RETRIEVERS:
        # 單一公司則使用原始問題或單一 Query
        actual_query = q_apple if target == "apple" and q_apple else (q_tesla if target == "tesla" and q_tesla else question)
        print(colored(f"Action Input: {actual_query}", "cyan"))
        docs_content = RETRIEVERS[target].invoke(actual_query)
    else:
        print(colored(f"Action Input: {question}", "cyan"))
        docs_content = []
    return {"documents": docs_content, "search_count": state["search_count"] + 1}

# Task C: Grader 階段 (對應 Observation 處理)
@retry_logic
def grade_documents_node(state: AgentState): 
    print(colored("Observation: Retrieved documents from vector store.", "magenta"))
    question = state["question"]
    documents = state["documents"]
    
    # --- Debug: 確認是否有抓到資料 ---
    if not documents:
        print(colored("  ⚠️ No documents found to grade!", "red"))
        return {"documents": [], "needs_rewrite": "yes"}

    filtered_docs = []
    needs_rewrite = "no" # 預設不需要重寫

    llm = get_llm() # LLM 放在迴圈外，不用每次都初始化

    for d in documents:
        system_prompt = """You are a grader assessing relevance. 
        If the user is comparing two companies, a document is RELEVANT if it contains data for EITHER company mentioned.
        As long as the document provides financial facts that can contribute to the final answer, output 'yes'.
        Answer with ONLY 'yes' or 'no'."""
        
        # 注意：這裡傳入的是 d.page_content (單個段落)，不是 documents (全體)
        msg = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Document context: {d.page_content} \n\n Question: {question}")
        ]
        
        try:
            response = llm.invoke(msg)
            grade = response.content.strip().lower()
            
            # 確保印出判定結果，方便你觀察
            if "yes" in grade:
                print(colored("  ✅ Relevant", "green"))
                filtered_docs.append(d)
            else:
                print(colored("  ❌ Irrelevant", "red"))
        except Exception as e:
            print(colored(f"⚠️ Grading failed: {e}", "red"))

    # 如果最後一個有用的文件都沒有，就標記需要重寫
    if not filtered_docs:
        # 新增：先判斷問題類型
        llm = get_llm()
        scope_check = llm.invoke(f"""
            Is this question answerable from a company's annual financial report (10-K)?
            Question: {question}
            Answer ONLY 'yes' or 'no'.
        """)
        if "no" in scope_check.content.lower():
            return {"documents": [], "needs_rewrite": "out_of_scope"}  # 新增狀態
        return {"documents": [], "needs_rewrite": "yes"}

# Task E: Generator 階段 (對應 Final Answer)
@retry_logic
def generate_node(state: AgentState):
    print(colored("Thought: Generating final answer.", "yellow"))
    
    question = state["question"]
    documents = state["documents"]
    llm = get_llm() 

    # 格式化 context 包含來源資訊，方便 LLM 引用 
    context_text = ""
    for i, d in enumerate(documents):
        # 這裡手動加入來源標記，幫助 LLM 符合 CITATION 規定
        source = d.metadata.get('source', 'Financial Report')
        context_text += f"--- Document {i+1} (Source: {source}) ---\n{d.page_content}\n\n"
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a professional financial analyst. 
        STRICT REQUIREMENTS:
        1. LANGUAGE: Response must be ENTIRELY in English.
        2. PRECISION: Distinguish between 2024, 2023, and 2022. Use 'Year ended' or 'Twelve months ended' for annual data.
        3. HONESTY: If figures are missing after checking ALL context, say 'I don't know'.
        4. CITATION: Strictly cite every claim, e.g., [Source: Apple 10-K].
        5. CALCULATION: If 'Gross Margin %' is not explicitly stated, you MUST calculate it: (Revenue - Cost of Revenue) / Revenue. Use 'Total Net Sales' and 'Total Cost of Sales' from the Statements of Operations.
        6. DATA PRIORITY: Prioritize 'Consolidated Total' figures over 'Segment Change' or 'Increase' amounts.
        7. COMPARISON: For comparison questions, you MUST list figures for BOTH companies before concluding who spent/earned more.
        8. NO TIMIDITY: If these two numbers exist anywhere in the context, use them to calculate the percentage. Do not say 'I don't know' if the raw numbers for revenue and cost are available.
        9. MANDATORY DATA SEARCH: You MUST look for 'Total net sales' and 'Total cost of sales' in the context. They are often in a table format.
        10. OUT OF SCOPE: If no relevant documents were found AND the question asks about future projections or information not typically in annual reports, respond clearly with "This information is not available in the financial reports provided."
        11. 11. TERMINOLOGY: Tesla uses 'Total revenues' and 'Total cost of revenues' 
    (NOT 'net sales'). Apple uses 'Total net sales' and 'Total cost of sales'. 
    For Tesla gross margin %, use the CONSOLIDATED total figures, 
    NOT segment-level (Automotive only) figures.
         """),
        
        ("human", "User Question: {question}\n\nContext:\n{context}"),
    ])  
    
    chain = prompt | llm
    response = chain.invoke({"context": context_text, "question": question})    
    return {"generation": response.content}

@retry_logic
def rewrite_node(state: AgentState): 
    print(colored("--- 🔄 REWRITING QUERY ---", "red"))
    question = state["question"]
    llm = get_llm()
    
    msg = [ 
        HumanMessage(content=f"""
        The previous search for '{question}' failed. 
        Please provide a concise, PROFESSIONAL ENGLISH search query. 
        - Convert '研發費用' to 'Research and development expenses'.
        - Convert '總營收' to 'Total net sales'.
        - Convert '毛利' to 'Gross margin'.
        - Include 'fiscal year 2024' or 'year ended 2024'.
        Output ONLY the new English query text.""")
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
        if state["needs_rewrite"] == "out_of_scope":
            return "generate"  # 直接生成「我不知道」
        if state["needs_rewrite"] == "yes":
            if state["search_count"] > 2:
                return "generate"
            return "rewrite"
        return "generate"
        
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

    template = """Answer the following questions as best you can. You have access to the following tools:
    {tools}

    Use the following format strictly:
    Question: {input}
    Thought: {agent_scratchpad}
    Action: the action to take, must be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result returned by the tool
    ... (this Thought/Action/Action Input/Observation can repeat)
    Thought: I now know the final answer
    Final Answer: the conclusion in English.

    Constraints:
    - English Only: The Final Answer must be in English[cite: 32].
    - Year Precision: Distinguish between 2024, 2023, and 2022 data.
    - Honesty: If the exact 2024 figure is not found, say "I don't know".
    """

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
    response = run_graph_agent("Which company had a higher Total Gross Margin percentage in 2024, Apple or Tesla? Please provide the approximate percentages.")
    print(response)