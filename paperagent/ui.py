import json
import streamlit as st

from paperagent.config import (
    DEFAULT_MODEL_PROFILE,
    MODEL_PROFILE_LABELS,
    MODEL_PROFILES,
    PAPER_INDEX_PATH,
    get_model_name,
)
from paperagent.engine import PaperAgentEngine
from paperagent.rag_chain import wants_source_details


EXAMPLE_QUESTIONS = {
    "作者论文": "张会彦是谁？当前论文库收录了哪些相关论文？",
    "数据来源": "哪些论文使用了 Gaia DR2？请按文件列出来源。",
    "对象对比": "对比 Triton 和 Himalia 相关论文的数据来源差异。",
    "方法总结": "总结 2021_AJ_New Positions of Triton Based on Gaia DR2 这篇论文的观测数据、处理方法和结论，请列出来源。",
    "误差分析": "论文中 Triton 卫星的星表拟合位置残差和精度是多少？",
}


def main():
    st.set_page_config(
        page_title="PaperAgent 学术文献智能体",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="collapsed",
        menu_items={
            "Get Help": None,
            "Report a bug": None,
            "About": "PaperAgent：融合确定性检索与大模型反向代理的学术文献智能体",
        },
    )
    inject_style()
    init_state()
    auto_load_engine()

    render_topbar()
    render_chat_surface()


def inject_style():
    st.markdown(
        """
        <style>
        #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"],
        [data-testid="stStatusWidget"], [data-testid="stSidebar"], [data-testid="collapsedControl"],
        [data-testid="stHeader"], .stAppHeader {
            display: none !important;
        }
        .block-container {
            max-width: 1020px;
            padding-top: 0.3rem;
            padding-bottom: 5.2rem;
            overflow-x: hidden;
        }
        .stApp {
            overflow-x: hidden;
            background-color: #fafbfc;
        }
        .pa-topline {
            display: flex;
            align-items: baseline;
            gap: 0.65rem;
            min-height: 2.35rem;
        }
        .pa-brand {
            color: #0f172a;
            font-size: 1.28rem;
            font-weight: 900;
            letter-spacing: -0.02em;
        }
        .pa-tagline {
            font-size: 0.76rem;
            font-weight: 600;
            padding: 2px 7px;
            border-radius: 999px;
            background: #e0f2fe;
            color: #0369a1;
        }
        .pa-identity {
            color: #64748b;
            font-size: 0.82rem;
            line-height: 1.35;
            margin-top: -0.25rem;
        }
        .pa-tech {
            color: #334155;
            font-weight: 650;
        }
        .pa-status {
            color: #64748b;
            font-size: 0.82rem;
            white-space: nowrap;
        }
        .pa-status-dot {
            display: inline-block;
            width: 0.5rem;
            height: 0.5rem;
            border-radius: 999px;
            margin-right: 0.35rem;
            background: #94a3b8;
        }
        .pa-status-dot.ready {
            background: #10b981;
            box-shadow: 0 0 6px rgba(16, 185, 129, 0.5);
        }
        .pa-status-dot.error {
            background: #ef4444;
        }
        .pa-control-row {
            padding: 0.4rem 0.2rem 0.6rem;
            border-bottom: 1px solid #e2e8f0;
            background: #ffffff;
            border-radius: 12px;
            margin-bottom: 0.6rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.02);
        }
        [data-testid="column"] {
            min-width: 0;
        }
        div[data-testid="stButton"] > button {
            min-height: 2.2rem;
            border-radius: 8px;
            font-weight: 550;
            font-size: 0.88rem;
            padding-left: 0.75rem;
            padding-right: 0.75rem;
        }
        div[data-testid="stSelectbox"] label {
            display: none;
        }
        .st-key-chat_panel {
            height: min(700px, calc(100dvh - 190px)) !important;
            overflow-y: auto;
            overflow-x: hidden;
            border: 0 !important;
            padding: 0.5rem 0.2rem 0.8rem;
        }
        .st-key-chat_panel [data-testid="stVerticalBlock"] {
            gap: 0.7rem;
        }
        div[data-testid="stChatInput"] {
            max-width: 1020px;
            margin: 0 auto;
            padding-left: 0.85rem;
            padding-right: 0.85rem;
        }
        div[data-testid="stChatMessage"] {
            border-radius: 0;
            background: transparent;
        }
        div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            flex-direction: row-reverse;
            text-align: right;
        }
        div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) div[data-testid="stMarkdownContainer"] {
            margin-left: auto;
            max-width: 78%;
            min-width: 0;
            color: #ffffff;
            background: linear-gradient(135deg, #2563eb, #1d4ed8);
            border-radius: 16px;
            padding: 0.68rem 0.95rem;
            box-shadow: 0 2px 5px rgba(37, 99, 235, 0.15);
            font-size: 0.95rem;
        }
        div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) div[data-testid="stMarkdownContainer"] {
            max-width: 88%;
            min-width: 0;
            color: #0f172a;
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 0.8rem 1.05rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
            font-size: 0.95rem;
            line-height: 1.6;
        }
        .pa-empty {
            max-width: 750px;
            margin: 2.2rem auto 0;
            text-align: center;
        }
        .pa-empty-title {
            color: #0f172a;
            font-size: 2rem;
            font-weight: 900;
            letter-spacing: -0.02em;
            margin-bottom: 0.35rem;
        }
        .pa-empty-subtitle {
            color: #64748b;
            font-size: 0.96rem;
            margin-bottom: 1.2rem;
        }
        .source-card {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 9px;
            padding: 0.55rem 0.75rem;
            margin-top: 0.45rem;
            font-size: 0.84rem;
        }
        .source-header {
            font-weight: 600;
            color: #1e293b;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .source-page {
            background: #e2e8f0;
            color: #334155;
            padding: 1px 6px;
            border-radius: 4px;
            font-size: 0.75rem;
        }
        .source-snippet {
            color: #475569;
            margin-top: 0.3rem;
            font-size: 0.8rem;
            line-height: 1.45;
        }
        .reasoning-box {
            background: #f8fafc;
            border-left: 3px solid #3b82f6;
            padding: 0.6rem 0.8rem;
            border-radius: 0 8px 8px 0;
            color: #475569;
            font-size: 0.86rem;
            margin-bottom: 0.6rem;
            line-height: 1.5;
        }
        div[data-testid="stMarkdownContainer"] table {
            width: 100%;
            border-collapse: collapse;
            margin: 0.6rem 0;
            font-size: 0.9rem;
        }
        div[data-testid="stMarkdownContainer"] th {
            background: #f1f5f9;
            padding: 0.45rem 0.6rem;
            border: 1px solid #cbd5e1;
            font-weight: 650;
        }
        div[data-testid="stMarkdownContainer"] td {
            padding: 0.45rem 0.6rem;
            border: 1px solid #e2e8f0;
        }
        @media (max-width: 768px) {
            .block-container {
                padding: 0.4rem 0.5rem 4.9rem;
            }
            .pa-brand {
                font-size: 1.1rem;
            }
            .st-key-chat_panel {
                height: calc(100dvh - 200px) !important;
            }
            div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) div[data-testid="stMarkdownContainer"],
            div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) div[data-testid="stMarkdownContainer"] {
                max-width: 95%;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state():
    st.session_state.setdefault("model_profile", DEFAULT_MODEL_PROFILE)
    st.session_state.setdefault("rebuild_token", 0)
    st.session_state.setdefault("engine", None)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("loaded", False)
    st.session_state.setdefault("load_error", None)
    st.session_state.setdefault("pending_question", None)
    st.session_state.setdefault("example_choice", None)


def auto_load_engine():
    if st.session_state.loaded or st.session_state.load_error:
        return
    model_name = get_model_name(st.session_state.model_profile)
    if PaperAgentEngine.status(model_name).has_vectorstore:
        load_engine(rebuild=False)


@st.cache_resource(show_spinner=False)
def cached_engine(model_name, rebuild_token):
    engine = PaperAgentEngine(model_name)
    engine.initialize(rebuild=bool(rebuild_token))
    return engine


def load_engine(rebuild=False):
    try:
        if rebuild:
            st.session_state.rebuild_token += 1
        model_name = get_model_name(st.session_state.model_profile)
        st.session_state.engine = cached_engine(model_name, st.session_state.rebuild_token)
        st.session_state.loaded = True
        st.session_state.load_error = None
    except Exception as exc:
        st.session_state.loaded = False
        st.session_state.load_error = str(exc)


def switch_model_profile(profile):
    st.session_state.model_profile = profile
    model_name = get_model_name(profile)
    engine = st.session_state.get("engine")
    if engine is not None:
        engine.set_model(model_name)


def render_topbar():
    model_name = get_model_name(st.session_state.model_profile)
    status = PaperAgentEngine.status(model_name)
    state_label, dot_class = current_status_label()

    st.markdown('<div class="pa-control-row">', unsafe_allow_html=True)
    title_col, model_col, action_col = st.columns([1.35, 1.35, 1.3], gap="small", vertical_alignment="center")

    with title_col:
        st.markdown(
            f"""
            <div class="pa-topline">
              <span class="pa-brand">PaperAgent</span>
              <span class="pa-tagline">工位反向代理</span>
              <span class="pa-status"><span class="pa-status-dot {dot_class}"></span>{state_label}</span>
            </div>
            <div class="pa-identity">文献库：<span class="pa-tech">{status.paper_count} 篇</span> · 驱动：<span class="pa-tech">Gemini / Claude / FAISS</span></div>
            """,
            unsafe_allow_html=True,
        )

    with model_col:
        profiles = list(MODEL_PROFILES)
        selected_profile = st.selectbox(
            "选择推理模型",
            options=profiles,
            index=profiles.index(st.session_state.model_profile)
            if st.session_state.model_profile in profiles
            else 0,
            format_func=lambda key: MODEL_PROFILE_LABELS.get(key, f"{key} · {MODEL_PROFILES[key]}"),
            width="stretch",
        )
        if selected_profile != st.session_state.model_profile:
            switch_model_profile(selected_profile)
            st.toast(f"已切换为：{MODEL_PROFILE_LABELS.get(selected_profile, selected_profile)}", icon=":material/check_circle:")
            st.rerun()

    with action_col:
        with st.container(horizontal=True, horizontal_alignment="right", gap="small"):
            if st.button("文献库", icon=":material/menu_book:", help="浏览收录的 61 篇学术论文清单"):
                paper_library_dialog()
            if st.button("重建", icon=":material/refresh:", help="从 MinerU 文本重建 FAISS 向量库"):
                confirm_rebuild_dialog()
            if st.button("清空", icon=":material/delete:", help="清空当前会话聊天记录"):
                confirm_clear_dialog()

    if st.session_state.load_error:
        st.error(f"加载失败：{st.session_state.load_error}", icon=":material/error:")
    st.markdown("</div>", unsafe_allow_html=True)


def current_status_label():
    if st.session_state.load_error:
        return "加载失败", "error"
    if st.session_state.loaded:
        return "知识库与代理就绪", "ready"
    return "未就绪", ""


@st.dialog("已收录学术论文库（61 篇）", width="large")
def paper_library_dialog():
    st.markdown("当前知识库收录了团队在**天体测量、木星海王星卫星精密历表、Gaia DR2/EDR3 星表比对**等领域的 61 篇核心论文。")
    records = []
    if PAPER_INDEX_PATH.exists():
        try:
            with open(PAPER_INDEX_PATH, "r", encoding="utf-8") as f:
                records = json.load(f)
        except Exception:
            pass

    search_kw = st.text_input("搜索论文关键词 / 年份 / 作者", placeholder="例如：Gaia, Triton, Himalia, 张会彦, 2021")
    filtered = records
    if search_kw:
        kw = search_kw.lower().strip()
        filtered = [
            r for r in records
            if kw in r.get("filename", "").lower()
            or kw in r.get("title", "").lower()
            or kw in r.get("first_pages_text", "").lower()
        ]

    st.caption(f"共显示 {len(filtered)} / {len(records)} 篇文献")
    with st.container(height=380):
        for idx, rec in enumerate(filtered, start=1):
            fname = rec.get("filename", "未知文件")
            title = rec.get("title") or fname
            st.markdown(f"**{idx}. {fname}**")
            if title and title != fname:
                st.caption(f"📌 {title}")
            st.divider()


@st.dialog("重建向量库")
def confirm_rebuild_dialog():
    st.markdown(
        "重建会从 `data/mineru/` 中的 MinerU Markdown 重新清洗、切片、计算 BGE-M3 embedding，并覆盖 `vectorstore/` 中的 FAISS 索引。"
    )
    st.warning("重建需要数分钟计算，原有 PDF 文件保持不变。", icon=":material/warning:")
    with st.container(horizontal=True, horizontal_alignment="right", gap="small"):
        if st.button("取消", key="cancel_rebuild"):
            st.rerun()
        if st.button("确定重建", type="primary", key="confirm_rebuild"):
            load_engine(rebuild=True)
            st.rerun()


@st.dialog("清空对话")
def confirm_clear_dialog():
    st.markdown("清空当前聊天记录，不影响论文库和本地索引文件。")
    with st.container(horizontal=True, horizontal_alignment="right", gap="small"):
        if st.button("取消", key="cancel_clear"):
            st.rerun()
        if st.button("确定", type="primary", key="confirm_clear"):
            st.session_state.messages = []
            st.session_state.pending_question = None
            st.session_state.example_choice = None
            st.rerun()


def render_chat_surface():
    with st.container(height=700, border=False, key="chat_panel"):
        if not st.session_state.messages:
            render_empty_state()
        render_chat_history()

        if st.session_state.pending_question:
            question = st.session_state.pending_question
            st.session_state.pending_question = None
            ask(question)
            st.rerun()

    question = st.chat_input("输入论文科研问题（支持星表对比、观测方法、作者论文查询）...", disabled=not st.session_state.loaded)
    if question:
        ask(question)
        st.rerun()


def render_empty_state():
    st.markdown(
        """
        <div class="pa-empty">
          <div class="pa-empty-title">PaperAgent 学术文献智能体</div>
          <div class="pa-empty-subtitle">工位反向代理大模型驱动 · 融合确定性结构化索引与高精学术 RAG</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.pills(
        "快捷提问示例",
        options=list(EXAMPLE_QUESTIONS),
        key="example_choice",
        on_change=queue_example_question,
        label_visibility="collapsed",
        disabled=not st.session_state.loaded,
        width="stretch",
    )


def queue_example_question():
    selected = st.session_state.get("example_choice")
    if selected:
        st.session_state.pending_question = EXAMPLE_QUESTIONS[selected]
        st.session_state.example_choice = None


def render_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            render_message_content(message)


def ask(question):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("正在检索文献知识库并推理生成..."):
                response = st.write_stream(st.session_state.engine.answer_stream(question))
            result = getattr(st.session_state.engine, "last_result", None)
            reasoning = getattr(result, "reasoning", None) if result else None
            sources = getattr(result, "sources", []) if result else []
            diagnostics = result.diagnostics() if result else ""

            if reasoning:
                with st.expander("🧠 思考与推演过程 (Reasoning)", expanded=False):
                    st.markdown(f'<div class="reasoning-box">{reasoning}</div>', unsafe_allow_html=True)

            if sources and wants_source_details(question):
                render_sources_cards(sources)

            render_diagnostics(diagnostics)
        except Exception as exc:
            response = format_error_message(exc)
            reasoning = None
            sources = []
            diagnostics = ""
            st.error(response, icon=":material/error:")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
            "reasoning": reasoning,
            "sources": sources,
            "diagnostics": diagnostics,
        }
    )


def render_message_content(message):
    content = message["content"]
    if message["role"] != "assistant":
        st.markdown(content)
        return

    reasoning = message.get("reasoning")
    if reasoning:
        with st.expander("🧠 思考与推演过程 (Reasoning)", expanded=False):
            st.markdown(f'<div class="reasoning-box">{reasoning}</div>', unsafe_allow_html=True)

    if "\n来源：" in content:
        answer, sources_text = content.split("\n来源：", 1)
        st.markdown(answer)
        with st.expander("📄 来源引用片段", expanded=False):
            st.markdown(f"来源：{sources_text}")
    else:
        st.markdown(content)

    sources = message.get("sources")
    if sources and not ("\n来源：" in content):
        render_sources_cards(sources)

    render_diagnostics(message.get("diagnostics"))


def render_sources_cards(sources: list[dict]):
    if not sources:
        return
    with st.expander("📄 溯源文献与原文卡片", expanded=False):
        for s in sources:
            fname = s.get("filename", "未知文件")
            page = s.get("page", "?")
            snippet = s.get("snippet", "")
            st.markdown(
                f"""
                <div class="source-card">
                  <div class="source-header">
                    <span>📄 {fname}</span>
                    <span class="source-page">P{page}</span>
                  </div>
                  <div class="source-snippet">{snippet}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_diagnostics(diagnostics):
    if not diagnostics:
        return
    with st.expander("📊 运行诊断与性能监控", expanded=False):
        st.caption(diagnostics)


def format_error_message(exc):
    message = str(exc)
    if "Cannot connect to" in message or "Connection refused" in message:
        return "无法连接学术推理内核服务。请确认服务已启动且网络端口连通正常。"
    if "请先加载论文知识库" in message:
        return "请先点击顶部“加载”。"
    return f"请求处理失败：{message}"
