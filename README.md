# Assignment 3: Autonomous Multi-Doc Financial Analyst

本專案實作了一個基於 **LangGraph** 的自動化多文檔財務分析代理系統。透過構建狀態感知（State-aware）的 RAG 流程，實現了具備自我修正與智慧路由功能的金融分析工具。


## Execution
* Step1: `python build_rag.py`
* Step2: `python evaluator.py`

## Note
只要有換模型或是改chunk_size都要先build_rag，才能進行evaluator
