import streamlit as st

from paperagent.config import DEFAULT_MODEL_PROFILE, MODEL_PROFILES, get_model_name
from paperagent.engine import PaperAgentEngine


EXAMPLE_QUESTIONS = {
    "作者论文": "张会彦是谁？当前论文库收录了哪些相关论文？",
    "数据来源": "哪些论文使用了 Gaia DR2？请按文件列出来源。",
    "对象对比": "对比 Triton 和 Himalia 相关论文的数据来源差异。",
    "方法总结": "总结 2021_AJ_New Positions of Triton Based on Gaia DR2 这篇论文的观测数据、处理方法和结论，请列出来源。",
}


def main():
    st.set_page_config(
        page_title="PaperAgent",
        page_icon="PA",
        layout="wide",
        initial_sidebar_state="collapsed",
        menu_items={
            "Get Help": None,
            "Report a bug": None,
            "About": "PaperAgent：本地论文知识库智能体",
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
            max-width: 980px;
            padding-top: 0.25rem;
            padding-bottom: 5.2rem;
            overflow-x: hidden;
        }
        .stApp {
            overflow-x: hidden;
        }
        .pa-topline {
            display: flex;
            align-items: baseline;
            gap: 0.55rem;
            min-height: 2.35rem;
        }
        .pa-brand {
            color: #111827;
            font-size: 1.16rem;
            font-weight: 900;
            line-height: 1.1;
        }
        .pa-identity {
            color: #6b7280;
            font-size: 0.82rem;
            line-height: 1.35;
            margin-top: -0.28rem;
        }
        .pa-tech {
            color: #374151;
            font-weight: 620;
        }
        .pa-status {
            color: #6b7280;
            font-size: 0.82rem;
            white-space: nowrap;
        }
        .pa-status-dot {
            display: inline-block;
            width: 0.48rem;
            height: 0.48rem;
            border-radius: 999px;
            margin-right: 0.32rem;
            background: #9ca3af;
        }
        .pa-status-dot.ready {
            background: #16a34a;
        }
        .pa-status-dot.error {
            background: #dc2626;
        }
        .pa-control-row {
            padding-bottom: 0.35rem;
            border-bottom: 1px solid #eef2f7;
            overflow-x: hidden;
        }
        [data-testid="column"] {
            min-width: 0;
        }
        div[data-testid="stButton"] > button {
            min-height: 2.18rem;
            border-radius: 8px;
            padding-left: 0.72rem;
            padding-right: 0.72rem;
        }
        div[data-testid="stSelectbox"] label {
            display: none;
        }
        div[data-testid="stDialog"] {
            align-items: flex-start !important;
        }
        div[data-testid="stDialog"] div[role="dialog"] {
            margin-top: 7vh !important;
        }
        .st-key-chat_panel {
            height: min(680px, calc(100dvh - 178px)) !important;
            overflow-y: auto;
            overflow-x: hidden;
            border: 0 !important;
            padding: 0.45rem 0.1rem 0.75rem;
        }
        .st-key-chat_panel [data-testid="stVerticalBlock"] {
            gap: 0.6rem;
        }
        div[data-testid="stChatInput"] {
            max-width: 980px;
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
            max-width: 74%;
            min-width: 0;
            color: #ffffff;
            background: #2563eb;
            border-radius: 14px;
            padding: 0.62rem 0.82rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.08);
        }
        div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) div[data-testid="stMarkdownContainer"] {
            max-width: 84%;
            min-width: 0;
            color: #111827;
            background: #f9fafb;
            border: 1px solid #edf0f4;
            border-radius: 14px;
            padding: 0.68rem 0.88rem;
        }
        .pa-empty {
            max-width: 720px;
            margin: 2.6rem auto 0;
            text-align: center;
        }
        .pa-empty-title {
            color: #111827;
            font-size: 1.85rem;
            font-weight: 900;
            line-height: 1.15;
            margin-bottom: 0.35rem;
        }
        div[data-testid="stMarkdownContainer"] {
            max-width: 100%;
            overflow-wrap: anywhere;
            word-break: break-word;
        }
        div[data-testid="stMarkdownContainer"] table {
            display: table;
            width: 100%;
            max-width: 100%;
            table-layout: fixed;
            overflow-wrap: anywhere;
            word-break: break-word;
        }
        div[data-testid="stMarkdownContainer"] th,
        div[data-testid="stMarkdownContainer"] td {
            white-space: normal !important;
            overflow-wrap: anywhere;
            word-break: break-word;
        }
        div[data-testid="stMarkdownContainer"] pre {
            white-space: pre-wrap;
            overflow-x: hidden;
        }
        .pa-empty-subtitle {
            color: #6b7280;
            font-size: 0.96rem;
            margin-bottom: 1rem;
        }
        .pa-empty div[data-testid="stPills"] {
            justify-content: center;
        }
        @media (max-width: 768px) {
            .block-container {
                padding: 0.45rem 0.58rem 4.9rem;
            }
            .pa-topline {
                min-height: auto;
                margin-bottom: 0.2rem;
            }
            .pa-brand {
                font-size: 1.02rem;
            }
            .pa-identity {
                font-size: 0.74rem;
                margin-top: -0.08rem;
            }
            .pa-status {
                font-size: 0.75rem;
            }
            .pa-control-row {
                padding-bottom: 0.22rem;
            }
            div[data-testid="stButton"] > button {
                min-height: 2rem;
                padding-left: 0.48rem;
                padding-right: 0.48rem;
                font-size: 0.82rem;
            }
            .st-key-chat_panel {
                height: calc(100dvh - 196px) !important;
                min-height: 430px;
                padding-top: 0.15rem;
            }
            div[data-testid="stChatInput"] {
                padding-left: 0.55rem;
                padding-right: 0.55rem;
            }
            div[data-testid="stChatMessage"] {
                gap: 0.25rem;
            }
            div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) div[data-testid="stMarkdownContainer"],
            div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) div[data-testid="stMarkdownContainer"] {
                max-width: 94%;
                padding: 0.56rem 0.68rem;
                font-size: 0.94rem;
            }
            .pa-empty {
                margin-top: 1.6rem;
            }
            .pa-empty-title {
                font-size: 1.38rem;
            }
            .pa-empty-subtitle {
                font-size: 0.86rem;
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
    title_col, model_col, action_col = st.columns([1.45, 0.85, 1.25], gap="small", vertical_alignment="center")

    with title_col:
        st.markdown(
            f"""
            <div class="pa-topline">
              <span class="pa-brand">PaperAgent</span>
              <span class="pa-status"><span class="pa-status-dot {dot_class}"></span>{state_label} · PDF {status.paper_count}</span>
            </div>
            <div class="pa-identity">By 杨哲 · <span class="pa-tech">Streamlit / Ollama / BGE-M3 / FAISS</span></div>
            """,
            unsafe_allow_html=True,
        )

    with model_col:
        profiles = list(MODEL_PROFILES)
        selected_profile = st.selectbox(
            "模型",
            options=profiles,
            index=profiles.index(st.session_state.model_profile)
            if st.session_state.model_profile in profiles
            else 0,
            format_func=lambda key: f"{key} · {MODEL_PROFILES[key]}",
            width="stretch",
        )
        if selected_profile != st.session_state.model_profile:
            switch_model_profile(selected_profile)
            st.toast(f"已切换到 {selected_profile}", icon=":material/check_circle:")
            st.rerun()

    with action_col:
        with st.container(horizontal=True, horizontal_alignment="right", gap="small"):
            if st.button("加载", icon=":material/play_arrow:", help="读取已有知识库。"):
                confirm_load_dialog()
            if st.button("重建", icon=":material/refresh:", help="从 MinerU 文本重建向量库。"):
                confirm_rebuild_dialog()
            if st.button("清空", icon=":material/delete:", help="清空当前对话。"):
                confirm_clear_dialog()

    if st.session_state.load_error:
        st.error(f"加载失败：{st.session_state.load_error}", icon=":material/error:")
    st.markdown("</div>", unsafe_allow_html=True)


def current_status_label():
    if st.session_state.load_error:
        return "加载失败", "error"
    if st.session_state.loaded:
        return "知识库已连接", "ready"
    return "知识库未加载", ""


@st.dialog("加载知识库")
def confirm_load_dialog():
    st.markdown(
        "加载会读取当前已有的 FAISS 向量库、论文索引和本地配置，不会重新切片，也不会改动论文文件。"
    )
    with st.container(horizontal=True, horizontal_alignment="right", gap="small"):
        if st.button("取消", key="cancel_load"):
            st.rerun()
        if st.button("确定", type="primary", key="confirm_load"):
            load_engine(rebuild=False)
            st.rerun()


@st.dialog("重建向量库")
def confirm_rebuild_dialog():
    st.markdown(
        "重建会从 `data/mineru/` 中的 MinerU Markdown 重新清洗、切片、计算 BGE-M3 embedding，并覆盖 `vectorstore/` 中的 FAISS 索引。"
    )
    st.warning("这会比普通加载更慢；论文 PDF 不会被删除。", icon=":material/warning:")
    with st.container(horizontal=True, horizontal_alignment="right", gap="small"):
        if st.button("取消", key="cancel_rebuild"):
            st.rerun()
        if st.button("确定", type="primary", key="confirm_rebuild"):
            load_engine(rebuild=True)
            st.rerun()


@st.dialog("清空对话")
def confirm_clear_dialog():
    st.markdown("清空只会删除当前浏览器会话里的聊天记录，不影响论文库、向量库和本地文件。")
    with st.container(horizontal=True, horizontal_alignment="right", gap="small"):
        if st.button("取消", key="cancel_clear"):
            st.rerun()
        if st.button("确定", type="primary", key="confirm_clear"):
            st.session_state.messages = []
            st.session_state.pending_question = None
            st.session_state.example_choice = None
            st.rerun()


def render_chat_surface():
    with st.container(height=680, border=False, key="chat_panel"):
        if not st.session_state.messages:
            render_empty_state()
        render_chat_history()

        if st.session_state.pending_question:
            question = st.session_state.pending_question
            st.session_state.pending_question = None
            ask(question)
            st.rerun()

    question = st.chat_input("问一个论文相关问题", disabled=not st.session_state.loaded)
    if question:
        ask(question)
        st.rerun()


def render_empty_state():
    st.markdown(
        """
        <div class="pa-empty">
          <div class="pa-empty-title">Ask the corpus</div>
          <div class="pa-empty-subtitle">杨哲构建的本地论文知识库智能体 · Streamlit / Ollama / BGE-M3 / FAISS</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.pills(
        "示例问题",
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
            with st.spinner("正在检索论文..."):
                response = st.write_stream(st.session_state.engine.answer_stream(question))
            result = getattr(st.session_state.engine, "last_result", None)
            diagnostics = result.diagnostics() if result else ""
            render_diagnostics(diagnostics)
        except Exception as exc:
            response = format_error_message(exc)
            diagnostics = ""
            st.error(response, icon=":material/error:")

    st.session_state.messages.append(
        {"role": "assistant", "content": response, "diagnostics": diagnostics}
    )


def render_message_content(message):
    content = message["content"]
    if message["role"] != "assistant":
        st.markdown(content)
        return

    if "\n来源：" in content:
        answer, sources = content.split("\n来源：", 1)
        st.markdown(answer)
        with st.expander(":material/source: 来源片段", expanded=False):
            st.markdown(f"来源：{sources}")
    else:
        st.markdown(content)

    render_diagnostics(message.get("diagnostics"))


def render_diagnostics(diagnostics):
    if not diagnostics:
        return
    with st.expander(":material/query_stats: 运行信息", expanded=False):
        st.caption(diagnostics)


def format_error_message(exc):
    message = str(exc)
    if "Cannot connect to Ollama" in message or "Connection refused" in message:
        return "无法连接 Ollama。请确认已运行 `ollama serve`，并且本机 11434 端口可访问。"
    if "model" in message.lower() and ("not found" in message.lower() or "pull" in message.lower()):
        return "当前 Ollama 模型不可用。请先运行 `ollama list` 确认模型存在，或切换模型档位。"
    if "请先加载论文知识库" in message:
        return "请先点击顶部“加载”。"
    return f"请求处理失败：{message}"
