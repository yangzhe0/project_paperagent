import html
import json
import os
import sys
from pathlib import Path

# 确保项目根目录在 Python 模块搜索路径中
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import streamlit as st

from paperagent.config import (
    DEFAULT_MODEL_PROFILE,
    MODEL_PROFILES,
    PAPER_INDEX_PATH,
    get_backend_for_model,
    get_model_name,
    get_model_profile,
)
from paperagent.core.engine import PaperAgentEngine
from paperagent.generation.ollama import OllamaClient
from paperagent.proxy_client import ProxyClient
from paperagent.rag_chain import wants_source_details
from paperagent.retrieval.knowledge_base import KnowledgeBase


AUTHOR_NAME = "杨哲"
AUTHOR_URL = "https://zheoi.com"

EXAMPLE_QUESTIONS = {
    "👤 作者论文": "张会彦是谁？当前论文库收录了哪些相关论文？",
    "🔭 数据来源": "哪些论文使用了 Gaia DR2？请按文件列出来源。",
    "🪐 对象对比": "对比 Triton 和 Himalia 相关论文的数据来源差异。",
    "📐 方法总结": "总结 2021_AJ_New Positions of Triton Based on Gaia DR2 这篇论文的观测数据、处理方法和结论，请列出来源。",
    "📊 精度残差": "论文中 Triton 卫星的星表拟合位置残差和测角精度是多少？",
}


def main():
    st.set_page_config(
        page_title="PaperAgent · 杨哲 (zheoi.com)",
        page_icon="🪐",
        layout="wide",
        initial_sidebar_state="collapsed",
        menu_items={
            "Get Help": AUTHOR_URL,
            "Report a bug": AUTHOR_URL,
            "About": f"""
            ### PaperAgent 学术文献智能体
            **作者**：[{AUTHOR_NAME}]({AUTHOR_URL})
            
            **架构**：融合天体测量多维结构化索引与新一代长上下文学术推理内核（Gemini 3.8 Flash / Claude Thinking / BGE-M3 / FAISS）。
            收录 61 篇深空天体测量与精密卫星历表核心科研论文。
            """,
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
        /* 隐藏 Streamlit 原生杂项 */
        #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"],
        [data-testid="stStatusWidget"], [data-testid="stSidebar"], [data-testid="collapsedControl"],
        [data-testid="stHeader"], .stAppHeader {
            display: none !important;
        }

        /* 全局排版与现代质感 */
        .stApp {
            background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Microsoft YaHei", sans-serif;
            color: #0f172a;
        }
        .block-container {
            max-width: 1040px;
            padding-top: 0.5rem;
            padding-bottom: 5rem;
            overflow-x: hidden;
        }

        /* 顶部导航卡片 (支持桌面端与移动端弹性自适应) */
        .pa-header-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 0.55rem 0.95rem;
            margin-bottom: 0.5rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
        }
        .pa-header-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.6rem;
        }
        .pa-header-left {
            display: flex;
            align-items: center;
            gap: 0.65rem;
        }
        .pa-header-right {
            display: flex;
            align-items: center;
            gap: 0.65rem;
        }
        .pa-brand-title {
            font-size: 1.32rem;
            font-weight: 900;
            letter-spacing: -0.03em;
            white-space: nowrap;
            background: linear-gradient(135deg, #0f172a 0%, #2563eb 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .pa-tag-astro {
            font-size: 0.76rem;
            font-weight: 650;
            white-space: nowrap;
            padding: 2px 8px;
            border-radius: 999px;
            background: #eff6ff;
            color: #1d4ed8;
            border: 1px solid #dbeafe;
        }
        .pa-author-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            font-size: 0.8rem;
            font-weight: 700;
            white-space: nowrap;
            color: #1e293b;
            background: #f8fafc;
            padding: 3px 10px;
            border-radius: 999px;
            text-decoration: none !important;
            border: 1px solid #cbd5e1;
            box-shadow: 0 1px 2px rgba(0,0,0,0.03);
            transition: all 0.2s ease;
        }
        .pa-author-badge:hover {
            color: #2563eb;
            background: #eff6ff;
            border-color: #93c5fd;
            transform: translateY(-1px);
        }
        .pa-author-badge .avatar {
            background: linear-gradient(135deg, #3b82f6, #1d4ed8);
            color: white;
            font-size: 0.68rem;
            font-weight: 800;
            width: 1.25rem;
            height: 1.25rem;
            border-radius: 50%;
            display: inline-flex;
            align-items: center;
            justify-content: center;
        }
        .pa-mobile-sub {
            display: none;
        }
        .pa-mobile-dot {
            display: none;
        }

        /* 状态指示药丸 */
        .pa-status-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            font-size: 0.76rem;
            font-weight: 650;
            white-space: nowrap;
            padding: 3px 9px;
            border-radius: 999px;
            background: #ecfdf5;
            color: #065f46;
            border: 1px solid #a7f3d0;
        }
        .pa-status-pill.error {
            background: #fef2f2;
            color: #991b1b;
            border-color: #fecaca;
        }
        .pa-dot {
            width: 6px;
            height: 6px;
            border-radius: 999px;
            background: #10b981;
            box-shadow: 0 0 6px rgba(16, 185, 129, 0.6);
        }
        .pa-dot.error {
            background: #ef4444;
            box-shadow: 0 0 6px rgba(239, 68, 68, 0.6);
        }

        /* 模型状态指示卡片 (桌面端展示) */
        .pa-model-meta-bar {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 0.35rem 0.85rem;
            font-size: 0.82rem;
            margin-bottom: 0.6rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.02);
        }
        .pa-badge-mode {
            font-size: 0.74rem;
            font-weight: 700;
            padding: 2px 7px;
            border-radius: 6px;
            background: #dbeafe;
            color: #1e40af;
            white-space: nowrap;
        }
        .pa-badge-model-name {
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-weight: 600;
            color: #0f172a;
            white-space: nowrap;
        }
        .pa-model-desc {
            color: #64748b;
            flex-grow: 1;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        /* 按钮与输入框精致美化 */
        div[data-testid="stButton"] > button {
            border-radius: 9px !important;
            font-weight: 600 !important;
            font-size: 0.84rem !important;
            padding: 0.35rem 0.7rem !important;
            border: 1px solid #cbd5e1 !important;
            background: #ffffff !important;
            transition: all 0.2s ease;
        }
        div[data-testid="stButton"] > button:hover {
            border-color: #3b82f6 !important;
            color: #2563eb !important;
            background: #f8fafc !important;
            box-shadow: 0 2px 6px rgba(37, 99, 235, 0.1);
        }
        div[data-testid="stSelectbox"] label {
            display: none;
        }

        /* 聊天主界面容器 */
        .st-key-chat_panel {
            height: min(690px, calc(100dvh - 210px)) !important;
            overflow-y: auto;
            overflow-x: hidden;
            border: 0 !important;
            padding: 0.4rem 0.2rem 0.8rem;
        }
        .st-key-chat_panel [data-testid="stVerticalBlock"] {
            gap: 0.8rem;
        }

        /* 对话气泡 (Perplexity / Linear 风格) */
        div[data-testid="stChatMessage"] {
            border-radius: 0;
            background: transparent;
            padding: 0.35rem 0;
        }
        div[data-testid="stChatMessage"]:has([data-testid*="AvatarUser"]) {
            flex-direction: row-reverse !important;
            text-align: right;
        }
        div[data-testid="stChatMessage"]:has([data-testid*="AvatarUser"]) [data-testid="stChatMessageContent"] {
            margin-left: auto !important;
            max-width: 80% !important;
            min-width: 0 !important;
            color: #ffffff !important;
            background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%) !important;
            border-radius: 16px 16px 4px 16px !important;
            padding: 0.75rem 1.1rem !important;
            box-shadow: 0 2px 6px rgba(30, 64, 175, 0.18) !important;
            font-size: 0.94rem !important;
            line-height: 1.55 !important;
        }
        div[data-testid="stChatMessage"]:has([data-testid*="AvatarUser"]) [data-testid="stChatMessageContent"] * {
            color: #ffffff !important;
        }
        div[data-testid="stChatMessage"]:has([data-testid*="AvatarAssistant"]) [data-testid="stChatMessageContent"] {
            max-width: 90% !important;
            min-width: 0 !important;
            color: #0f172a !important;
            background: #ffffff !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 16px 16px 16px 4px !important;
            padding: 0.9rem 1.25rem !important;
            box-shadow: 0 2px 8px -2px rgba(15, 23, 42, 0.05) !important;
            font-size: 0.95rem !important;
            line-height: 1.65 !important;
        }

        /* 首页空状态设计 (Hero Section) */
        .pa-hero {
            text-align: center;
            max-width: 760px;
            margin: 2.2rem auto 1.5rem;
        }
        .pa-hero-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 3px 12px;
            border-radius: 999px;
            background: #eff6ff;
            color: #1d4ed8;
            font-size: 0.78rem;
            font-weight: 700;
            margin-bottom: 0.75rem;
            border: 1px solid #dbeafe;
        }
        .pa-hero-title {
            font-size: 2.2rem;
            font-weight: 900;
            letter-spacing: -0.03em;
            color: #0f172a;
            margin-bottom: 0.35rem;
            line-height: 1.2;
        }
        .pa-hero-author {
            font-size: 0.92rem;
            color: #475569;
            margin-bottom: 0.6rem;
        }
        .pa-hero-author a {
            color: #2563eb;
            font-weight: 700;
            text-decoration: none;
            border-bottom: 1px dashed #93c5fd;
            padding-bottom: 1px;
        }
        .pa-hero-author a:hover {
            color: #1d4ed8;
            border-bottom-style: solid;
        }
        .pa-hero-desc {
            font-size: 0.95rem;
            color: #64748b;
            line-height: 1.55;
            margin-bottom: 1.2rem;
        }

        /* 思考推理链容器 (Thinking Box) */
        .thinking-card {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-left: 3px solid #6366f1;
            border-radius: 0 10px 10px 0;
            padding: 0.75rem 0.95rem;
            margin-bottom: 0.65rem;
            font-size: 0.86rem;
            color: #475569;
            line-height: 1.55;
            white-space: pre-wrap;
        }

        /* 来源文献卡片 (Perplexity Citation Card) */
        .citation-card {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 0.6rem 0.85rem;
            margin-top: 0.5rem;
            transition: all 0.2s ease;
        }
        .citation-card:hover {
            background: #ffffff;
            border-color: #cbd5e1;
            box-shadow: 0 2px 6px rgba(0,0,0,0.03);
        }
        .citation-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-weight: 650;
            font-size: 0.85rem;
            color: #1e293b;
        }
        .citation-tag {
            font-size: 0.72rem;
            font-weight: 700;
            background: #e2e8f0;
            color: #334155;
            padding: 2px 6px;
            border-radius: 4px;
        }
        .citation-snippet {
            margin-top: 0.35rem;
            font-size: 0.81rem;
            color: #475569;
            line-height: 1.45;
        }

        /* ==================== 移动端深度适配 (<768px) ==================== */
        @media (max-width: 768px) {
            .block-container {
                padding: 0.3rem 0.5rem 4.5rem !important;
            }
            .pa-header-card {
                padding: 0.45rem 0.65rem !important;
                border-radius: 10px !important;
                margin-bottom: 0.35rem !important;
            }
            .pa-header-row {
                flex-direction: row !important;
                align-items: center !important;
                justify-content: space-between !important;
            }
            .pa-header-left {
                gap: 0.35rem !important;
            }
            .pa-brand-title {
                font-size: 1.15rem !important;
            }
            .pa-tag-astro {
                display: none !important;
            }
            .pa-status-pill {
                display: none !important;
            }
            .pa-mobile-dot {
                display: inline-block !important;
                width: 7px;
                height: 7px;
                border-radius: 999px;
                background: #10b981;
                box-shadow: 0 0 6px rgba(16, 185, 129, 0.7);
                margin-left: 2px;
            }
            .pa-mobile-dot.error {
                background: #ef4444;
                box-shadow: 0 0 6px rgba(239, 68, 68, 0.7);
            }
            .pa-mobile-sub {
                display: block !important;
                font-size: 0.72rem;
                color: #64748b;
                margin-top: 2px;
            }
            .pa-author-badge {
                font-size: 0.72rem !important;
                padding: 2px 8px !important;
            }
            .pa-author-badge .avatar {
                width: 1.15rem !important;
                height: 1.15rem !important;
                font-size: 0.62rem !important;
            }
            /* 移动端隐藏冗余的模型详情条，为对话释放宝贵的垂直屏幕空间 */
            .pa-model-meta-bar {
                display: none !important;
            }
            /* 移动端顶层控制器容器：将下拉框与按钮组拆分为上下两行，各自 100% 满宽 */
            div[data-testid="stHorizontalBlock"]:has([data-testid="stSelectbox"]) {
                flex-direction: column !important;
                gap: 0.35rem !important;
            }
            div[data-testid="stHorizontalBlock"]:has([data-testid="stSelectbox"]) > div.stColumn {
                width: 100% !important;
                min-width: 100% !important;
                max-width: 100% !important;
                flex: 1 1 100% !important;
            }
            /* 移动端三个操作按钮水平单行排列，平分 100% 宽度 */
            div[data-testid="stHorizontalBlock"]:has(button):not(:has([data-testid="stSelectbox"])) {
                flex-direction: row !important;
                flex-wrap: nowrap !important;
                gap: 0.35rem !important;
            }
            div[data-testid="stHorizontalBlock"]:has(button):not(:has([data-testid="stSelectbox"])) > div.stColumn {
                flex: 1 1 0% !important;
                min-width: 0 !important;
                width: 0 !important;
            }
            div[data-testid="stButton"] > button {
                width: 100% !important;
                font-size: 0.78rem !important;
                padding: 0.32rem 0.2rem !important;
                min-height: 2.15rem !important;
                white-space: nowrap !important;
                overflow: hidden !important;
                text-overflow: ellipsis !important;
            }
            /* 移动端 Hero 区域紧凑化 */
            .pa-hero {
                margin: 1.2rem auto 0.8rem !important;
                padding: 0 0.5rem !important;
            }
            .pa-hero-title {
                font-size: 1.45rem !important;
                margin-bottom: 0.25rem !important;
            }
            .pa-hero-badge {
                font-size: 0.72rem !important;
                padding: 2px 8px !important;
                margin-bottom: 0.4rem !important;
            }
            .pa-hero-author {
                font-size: 0.82rem !important;
                margin-bottom: 0.35rem !important;
            }
            .pa-hero-desc {
                font-size: 0.82rem !important;
                line-height: 1.45 !important;
                margin-bottom: 0.75rem !important;
            }
            /* 移动端聊天窗口与气泡 */
            .st-key-chat_panel {
                height: calc(100dvh - 250px) !important;
                padding: 0.5rem 0.2rem 1.5rem !important;
                margin-top: 0.4rem !important;
            }
            div[data-testid="stChatMessage"]:has([data-testid*="AvatarUser"]) [data-testid="stChatMessageContent"] {
                max-width: 88% !important;
                padding: 0.65rem 0.85rem !important;
                font-size: 0.88rem !important;
            }
            div[data-testid="stChatMessage"]:has([data-testid*="AvatarAssistant"]) [data-testid="stChatMessageContent"] {
                max-width: 98% !important;
                padding: 0.75rem 0.85rem !important;
                font-size: 0.88rem !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state():
    default_profile = os.environ.get("PAPERAGENT_DEFAULT_PROFILE", "flash")
    if default_profile not in MODEL_PROFILES:
        default_profile = DEFAULT_MODEL_PROFILE
    st.session_state.setdefault("model_profile", default_profile)
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
        with st.spinner("正在连接天体测量文献知识库..."):
            load_engine(rebuild=False)


@st.cache_resource(show_spinner=False)
def cached_knowledge_base(rebuild_token):
    return KnowledgeBase(rebuild=bool(rebuild_token))


def load_engine(rebuild=False):
    try:
        if rebuild:
            st.session_state.rebuild_token += 1
        model_name = get_model_name(st.session_state.model_profile)
        knowledge_base = cached_knowledge_base(st.session_state.rebuild_token)
        engine = PaperAgentEngine(model_name)
        engine.initialize(knowledge_base=knowledge_base)
        st.session_state.engine = engine
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


@st.cache_data(ttl=15, show_spinner=False)
def check_model_availability(model_name: str, backend: str):
    if backend == "proxy":
        try:
            healthy = ProxyClient(model_name).check_health()
            return healthy, "云端推理引擎在线" if healthy else "云端推理引擎离线"
        except Exception as exc:
            return False, f"引擎连接异常: {exc}"
    else:
        try:
            models = OllamaClient(model_name).list_models()
            available = model_name in models
            return available, "本地推理引擎就绪" if available else "模型未安装"
        except Exception as exc:
            return False, f"本地引擎未连接: {exc}"


def render_topbar():
    model_name = get_model_name(st.session_state.model_profile)
    profile = get_model_profile(st.session_state.model_profile)
    status = PaperAgentEngine.status(model_name)

    model_available, health_text = check_model_availability(model_name, profile.backend)
    status_class = "" if model_available else "error"

    # 第一行：顶部导航卡片（内部通过 Flex 自适应，移动端与桌面端均完美自适应）
    st.markdown(
        f"""
        <div class="pa-header-card">
          <div class="pa-header-row">
            <div class="pa-header-left">
              <span class="pa-brand-title">🪐 PaperAgent</span>
              <span class="pa-mobile-dot {status_class}" title="{health_text}"></span>
              <span class="pa-tag-astro">天体测量文献库 · {status.paper_count} 篇</span>
            </div>
            <div class="pa-header-right">
              <a href="{AUTHOR_URL}" target="_blank" class="pa-author-badge" title="访问作者杨哲个人主页">
                <span class="avatar">YZ</span>
                <span>{AUTHOR_NAME} · {AUTHOR_URL.replace('https://', '')}</span>
              </a>
              <span class="pa-status-pill {status_class}">
                <span class="pa-dot {status_class}"></span>
                <span>{health_text}</span>
              </span>
            </div>
          </div>
          <div class="pa-mobile-sub">
            收录 <b>{status.paper_count} 篇</b> 深空天体测量论文 · 云端前沿模型驱动
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 第二行：推理模型选择与文献操作工具栏
    col_model, col_actions = st.columns([1.6, 1.4], gap="small", vertical_alignment="center")
    with col_model:
        profiles = ["flash", "thinking", "pro"]
        selected_profile = st.selectbox(
            "选择推理模型",
            options=profiles,
            index=profiles.index(st.session_state.model_profile)
            if st.session_state.model_profile in profiles
            else 0,
            format_func=lambda key: f"{get_model_profile(key).label} · {MODEL_PROFILES[key]}",
            help="云端学术推理内核具备超大上下文与长思维链推演；本地模型支持离线运行。",
            width="stretch",
        )
        if selected_profile != st.session_state.model_profile:
            switch_model_profile(selected_profile)
            selected = get_model_profile(selected_profile)
            st.toast(f"已切换为：{selected.label}模式", icon=":material/check_circle:")
            st.rerun()

    with col_actions:
        btn_c1, btn_c2, btn_c3 = st.columns(3, gap="small")
        with btn_c1:
            if st.button("文献库", icon=":material/menu_book:", help="浏览收录的 61 篇学术论文清单", width="stretch"):
                paper_library_dialog()
        with btn_c2:
            if st.button("重建", icon=":material/refresh:", help="从 MinerU 文本重建向量库", width="stretch"):
                confirm_rebuild_dialog()
        with btn_c3:
            if st.button("清空", icon=":material/delete:", help="清空当前会话聊天记录", width="stretch"):
                confirm_clear_dialog()

    # 3. 桌面端专属：模型详情提示条（移动端通过 CSS 自动隐藏，不占垂直高度）
    st.markdown(
        f"""
        <div class="pa-model-meta-bar">
          <span class="pa-badge-mode">{html.escape(profile.label)}模式</span>
          <span class="pa-badge-model-name">{html.escape(model_name)}</span>
          <span class="pa-model-desc">{html.escape(profile.description)}</span>
          <a href="{AUTHOR_URL}" target="_blank" style="text-decoration:none; color:#2563eb; font-size:0.78rem; font-weight:700;">
            访问作者主页 {AUTHOR_NAME} ↗
          </a>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.dialog("已收录天体测量学术论文库（61 篇）", width="large")
def paper_library_dialog():
    st.markdown(
        f"""
        ### 🪐 深空天体测量文献知识库
        收录由 **[{AUTHOR_NAME}]({AUTHOR_URL})** 整理归纳的天体测量学、木星与海王星不规则卫星精密历表、Gaia DR2/EDR3 星表改正等领域的 61 篇核心论文。
        作者主页：[{AUTHOR_URL}]({AUTHOR_URL})
        """
    )
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

    st.caption(f"共检索到 {len(filtered)} / {len(records)} 篇文献")
    with st.container(height=400):
        for idx, rec in enumerate(filtered, start=1):
            fname = rec.get("filename", "未知文件")
            title = rec.get("title") or fname
            st.markdown(f"**{idx}. {fname}**")
            if title and title != fname:
                st.caption(f"📌 {title}")
            st.divider()


@st.dialog("重建向量库")
def confirm_rebuild_dialog():
    st.markdown("重建会从 `data/mineru/` 中的 MinerU Markdown 重新切片并计算 BGE-M3 embedding，覆盖 FAISS 索引。")
    st.warning("重建需要数分钟计算，原有 PDF 保持不变。", icon=":material/warning:")
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
    with st.container(height=690, border=False, key="chat_panel"):
        if not st.session_state.messages:
            render_empty_state()
        render_chat_history()

        if st.session_state.pending_question:
            question = st.session_state.pending_question
            st.session_state.pending_question = None
            ask(question)
            st.rerun()

    question = st.chat_input("向天体测量论文库提问（支持星表对比、观测方法、精度残差查询）...", disabled=not st.session_state.loaded)
    if question:
        ask(question)
        st.rerun()


def render_empty_state():
    profile = get_model_profile(st.session_state.model_profile)
    st.markdown(
        f"""
        <div class="pa-hero">
          <div class="pa-hero-badge">🪐 空间天体测量 · 行星卫星历表 · 学术知识库</div>
          <div class="pa-hero-title">向论文知识库提问</div>
          <div class="pa-hero-author">
            由 <b><a href="{AUTHOR_URL}" target="_blank">{AUTHOR_NAME}</a></b> 构建与维护 · 个人主页：<a href="{AUTHOR_URL}" target="_blank">{AUTHOR_URL.replace('https://', '')}</a>
          </div>
          <div class="pa-hero-desc">
            结合高精天体测量结构化索引与新一代推理大模型，支持精密历表比对、观测处理方法总结与高信度学术溯源
          </div>
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
    history = st.session_state.messages[-4:]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("正在检索学术论文库并推演思考..."):
                response = st.write_stream(
                    st.session_state.engine.answer_stream(question, history=history)
                )
            result = getattr(st.session_state.engine, "last_result", None)
            reasoning = getattr(result, "reasoning", None) if result else None
            sources = getattr(result, "sources", []) if result else []
            diagnostics = result.diagnostics() if result else ""

            if reasoning:
                with st.expander("🧠 深度推演与思考过程 (Reasoning)", expanded=False):
                    st.markdown(f'<div class="thinking-card">{html.escape(reasoning)}</div>', unsafe_allow_html=True)

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
        with st.expander("🧠 深度推演与思考过程 (Reasoning)", expanded=False):
            st.markdown(f'<div class="thinking-card">{html.escape(reasoning)}</div>', unsafe_allow_html=True)

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
    with st.expander("📄 溯源文献与原文卡片 (Citations)", expanded=False):
        for s in sources:
            fname = s.get("filename", "未知文件")
            page = s.get("page", "?")
            snippet = s.get("snippet", "")
            st.markdown(
                f"""
                <div class="citation-card">
                  <div class="citation-header">
                    <span>📄 {html.escape(fname)}</span>
                    <span class="citation-tag">P{html.escape(str(page))}</span>
                  </div>
                  <div class="citation-snippet">{html.escape(snippet)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_diagnostics(diagnostics):
    if not diagnostics:
        return
    with st.expander("📊 运行诊断与性能监控", expanded=False):
        st.caption(f"{diagnostics} · 作者：{AUTHOR_NAME} ({AUTHOR_URL})")


def format_error_message(exc):
    message = str(exc)
    if "Cannot connect to Reverse Proxy" in message or "10.126.126.2" in message:
        return "无法连接云端学术推理内核。请确认网络连通与服务端口状态。"
    if "Cannot connect to Ollama" in message or "Connection refused" in message:
        return "无法连接本地推理服务。请确认已运行 `ollama serve`，并且本机 11434 端口可访问。"
    if "model" in message.lower() and ("not found" in message.lower() or "pull" in message.lower()):
        return "当前模型不可用，请确认模型名称或切换模型档位。"
    if "format is outdated" in message or "schema changed" in message:
        return "知识库索引版本已更新，请点击顶部“重建”完成迁移。"
    if "Knowledge base is not built" in message:
        return "知识库尚未构建，请点击顶部“重建”。"
    if "请先加载论文知识库" in message:
        return "请先点击顶部“加载”。"
    return f"请求处理失败：{message}"


if __name__ == "__main__":
    main()

