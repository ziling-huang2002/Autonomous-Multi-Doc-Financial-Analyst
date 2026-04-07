# Assignment 3: Autonomous Multi-Doc Financial Analyst

本專案實作了一個基於 **LangGraph** 的自動化多文檔財務分析代理系統。透過構建狀態感知（State-aware）的 RAG 流程，實現了具備自我修正與智慧路由功能的金融分析工具。

---
## Execution
* Step1: `python build_rag.py`
* Step2: `python evaluator.py`
---

## 📊 實驗結果與模型比較 (Q1)

針對不同的 Embedding 模型進行測試，結果顯示多語言適配性是處理跨國財報（Apple/Tesla）的關鍵因素：

* **Model A: `paraphrase-multilingual-MiniLM-L12-v2`** 
    * **Final Score**: 5/14
    * **分析**：雖然參數較小，但作為多語言模型，能有效將中文問題映射至英文財報片段。
* **Model B: `BAAI/bge-small-en-v1.5`** 
    * **Final Score**: 1/14 
    * **分析**：純英文優化模型，面對中文提問時檢索相關性大幅下降，導致後續節點失效。

**核心發現**：輕量級模型在區分「季度數據」與「全年總計」時仍具局限性。模型選擇應優先考慮問題語系與專業領域的適配，而非僅看排名。

---

## 🔍 LangGraph vs. LangChain 詳細對照 (Q2)

本專案將流程從線性 Chain 轉向狀態圖（State Graph），主要差異如下：

| 特性 | LangChain | LangGraph |
| :--- | :--- | :--- |
| **結構模型** | **DAG (有向無環圖)**：流程線性，難以回頭。 |**Cyclic Graph**：支援循環與疊代。 |
| **狀態管理** | **無狀態 (Stateless)**：僅傳遞前一棒輸出 。 | **內置狀態 (State-aware)**：全域共享 `AgentState`。 |
| **控制權** | **高層級抽象**：ReAct Agent 決定一切（黑盒子） 。 | **精細控制**：開發者自定義節點與跳轉邏輯。 |
| **錯誤修復** | 依賴 LLM 自行修正，容易陷入死循環。 | **強制修復**：可設計專門的 `Rewrite` 節點。 |

### 關鍵優勢
* **Cyclic Iteration**：引入 `grade_documents` 節點，實現「不合格就重來」的顯式自我修正邏輯。
* **State Awareness**：利用 `search_count` 監控重試次數，避免 API 無止盡消耗。
* **Intelligent Routing**：預先判斷查詢目標（Apple/Tesla），提高檢索精準度 (Task B: Router)。

---

## ⚙️ Chunk Size 對大型表格的影響 (Q3)

針對資產負債表（Balance Sheet）等大型表格，`chunk_size` 的設定會產生顯著影響：

### 1. 調整影響
* **增加至 2000**：能封裝完整表格，確保數據與行列標籤（項目、年份）不被切斷，提升比較類問題的準確率。
* **減少至 500**：表格擷取不完整，導致 LLM 缺乏上下文而產生幻覺或回答 "I don't know"。

### 2. Trade-off 權衡
* **Context Precision (小語塊)**：精確定位、節省 Token、減少噪音，但資訊容易破碎化。
* **Context Completeness (大語塊)**：保留報表結構與邏輯關係，對財務推理至關重要，但易受噪音干擾且增加成本。

---

## ⚠️ 實作挑戰：API 限額問題
由於 LangGraph 將流程拆解為多個節點，並引入循環與自我修正機制，API 調用次數會呈指數級增長。在執行評估器（Evaluator）時，頻繁觸發 `Rate limit (429)` 錯誤是實作中最大的資源瓶頸。
