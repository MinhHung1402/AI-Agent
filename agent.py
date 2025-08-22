import os
import asyncio
import json
import requests
from typing import Annotated, List, TypedDict
import operator
from dotenv import load_dotenv

from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, AnyMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache

from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.checkpoint.memory import MemorySaver

from vectorstore import load as load_vectorstore
from toolbox_langchain import ToolboxClient

# -----------------------------
# 0) Environment & Caching
# -----------------------------
load_dotenv()
set_llm_cache(InMemoryCache())

REQUIRED_VARS = [
    "GOOGLE_API_KEY",
    "GOOGLE_CUSTOM_SEARCH_API_KEY",
    "GOOGLE_CUSTOM_SEARCH_CX",
]
for var in REQUIRED_VARS:
    if not os.getenv(var):
        raise RuntimeError(f"Missing env var: {var}")

# -----------------------------
# 1) Graph State
# -----------------------------
class AgentState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage | ToolMessage], operator.add]

# -----------------------------
# 2) Utilities
# -----------------------------
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "vi-VN,vi;q=0.9"})

def _http_get(url: str, *, timeout: int = 12) -> str:
    """Fetch page text with conservative timeouts & small cap to save tokens."""
    try:
        r = session.get(url, timeout=timeout)
        r.raise_for_status()
        return r.text[:8_000]  # tight cap to lower token usage drastically
    except Exception as e:
        return f"[fetch_error] {e}"

# -----------------------------
# 3) Tools
# -----------------------------
@tool("web_search", return_direct=False)
def web_search(query: str) -> str:
    """Tìm kiếm web độ chính xác cao với số lượng kết quả giới hạn để tiết kiệm quota.

    Args:
        query: Chuỗi truy vấn của người dùng.
    Returns:
        JSON string với các keys: { query, results: [ {rank, title, link, snippet, body_preview?} ] }
    """

    api_key = os.getenv("GOOGLE_CUSTOM_SEARCH_API_KEY")
    cx = os.getenv("GOOGLE_CUSTOM_SEARCH_CX")

    params = {
        "key": api_key,
        "cx": cx,
        "q": query,
        "num": 4,  # fewer results to save tokens
        "gl": "vn",
        "hl": "vi",
        "safe": "off",
    }

    try:
        r = session.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        items = data.get("items", [])
        if not items:
            return json.dumps({"error": "no_results"}, ensure_ascii=False)

        results = []
        for i, it in enumerate(items):
            results.append(
                {
                    "rank": i + 1,
                    "title": it.get("title"),
                    "link": it.get("link"),
                    "snippet": it.get("snippet"),
                }
            )

        # Only one preview page body to save tokens
        for i in range(min(1, len(results))):
            results[i]["body_preview"] = _http_get(results[i]["link"])[:8000]

        return json.dumps({"query": query, "results": results}, ensure_ascii=False)
    except requests.exceptions.RequestException as e:
        return json.dumps({"error": f"http_error: {e}"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"unexpected: {e}"}, ensure_ascii=False)

@tool("search_internal_document", return_direct=False)
def search_internal_document(query: str) -> str:
    """Tìm kiếm trong internal document bằng phương pháp MMR và gộp các đoạn liền kề để có ngữ cảnh dài hơn.

    Args:
        query: Câu truy vấn tìm kiếm.
    Returns:
        JSON string với các keys: { query, passages: [ {source, page, text} ] }
    """

    try:
        vs = load_vectorstore()
        try:
            retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 16, "lambda_mult": 0.6})
            docs = retriever.get_relevant_documents(query)
        except Exception:
            docs = [d for d, _ in vs.similarity_search_with_score(query, k=4)]

        merged: dict[tuple, List[str]] = {}
        for d in docs:
            src = d.metadata.get("source", "Unknown")
            page = d.metadata.get("page", d.metadata.get("page_number", "?"))
            key = (src, page)
            merged.setdefault(key, []).append(d.page_content.strip())

        merged_blocks = []
        for (src, page), chunks in merged.items():
            text = "".join(chunks)
            # hard cap each merged block to save tokens
            if len(text) > 1200:
                text = text[:1200] + "…"
            merged_blocks.append({"source": src, "page": page, "text": text})

        return json.dumps({"query": query, "passages": merged_blocks[:4]}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"internal_search_error: {e}"}, ensure_ascii=False)

# -----------------------------
# 4) Core LLM Call
# -----------------------------
def call_llm(state: AgentState, llm_with_tools):
    sys = SystemMessage(
    content=(
        """Bạn là trợ lý AI thông minh, có quyền sử dụng công cụ mà không cần xin phép người dùng.
            - Luôn luôn dùng công cụ `search_internal_document` trước khi nhắc đến tìm kiếm.
            - Nếu thông tin không tìm thấy hoặc không đủ, **bạn PHẢI tự động thực hiện `web_search`**, không hỏi người dùng.
            - Trả lời ngắn gọn, tối ưu token, tối đa khoảng 150 từ.
            - Không xin phép, không hỏi lại. Hãy tự hành động nếu bạn thấy cần thêm thông tin từ web."""
    )
)

    resp = llm_with_tools.invoke([sys] + state["messages"])
    return {"messages": [resp]}

# -----------------------------
# 5) Build Graph
# -----------------------------
def build_graph() -> StateGraph:
    client = ToolboxClient("http://127.0.0.1:5000")
    mcp_tools = client.load_toolset()

    tools = [*mcp_tools, web_search, search_internal_document]

    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        max_output_tokens=2048,
        top_p=0.9,
        timeout=20,
    ).bind_tools(tools)

    def _call(state):
        return call_llm(state, model)

    tool_node = ToolNode(tools)

    builder = StateGraph(AgentState)
    builder.add_node("assistant", _call)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition, {"tools": "tools", END: END})
    builder.add_edge("tools", "assistant")

    return builder

# -----------------------------
# 6) Init
# -----------------------------
memory = MemorySaver()
GRAPH = build_graph().compile(checkpointer=memory)
THREAD_ID = "gradio-thread"

# -----------------------------
# 7) App glue
# -----------------------------
async def agent_respond(message: str, history: List[List[str]]):
    # keep only the last N turns to reduce prompt tokens
    HISTORY_WINDOW = 6
    trimmed = history[-HISTORY_WINDOW:] if HISTORY_WINDOW else history

    msgs: List[HumanMessage | AIMessage] = []
    for u, a in trimmed:
        if u:
            msgs.append(HumanMessage(content=u))
        if a:
            msgs.append(AIMessage(content=a))
    msgs.append(HumanMessage(content=message))

    config = {
        "configurable": {"thread_id": THREAD_ID},
        "recursion_limit": 50,
    }
    final_state = await GRAPH.ainvoke({"messages": msgs}, config)

    for m in reversed(final_state["messages"]):
        if isinstance(m, AIMessage) and m.content:
            return m.content
    return "(Không có phản hồi)"

# -----------------------------
# 8) Gradio UI
# -----------------------------
import gradio as gr

with open("chat.css", "r", encoding="utf-8") as f:
    CUSTOM_CSS = f.read()


with gr.Blocks(css=CUSTOM_CSS) as demo:
    gr.Markdown('<div id="main-title">🤖 Gemini AI Tech Assistant</div>')
    chatbot = gr.Chatbot([], elem_id="chatbox", height=470, bubble_full_width=False, avatar_images=(None, None))
    with gr.Row():
        msg = gr.Textbox(show_label=False, placeholder="Type your message and press Enter...", elem_id="input-box", scale=8)
        send_btn = gr.Button("Send", elem_id="send-btn", scale=1)
        clear_btn = gr.Button("Clear", elem_id="clear-btn", scale=1)

    async def _on_submit(user_message, chat_history):
        try:
            reply = await agent_respond(user_message, chat_history)
        except Exception as e:
            reply = f"[ERROR]: {e}"
        chat_history.append((user_message, reply))
        return "", chat_history

    msg.submit(_on_submit, inputs=[msg, chatbot], outputs=[msg, chatbot])
    send_btn.click(_on_submit, inputs=[msg, chatbot], outputs=[msg, chatbot])
    clear_btn.click(lambda: ([], ""), None, [chatbot, msg])

if __name__ == "__main__":
    demo.launch(share=False)
