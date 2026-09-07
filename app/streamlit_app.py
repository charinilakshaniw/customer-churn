import html
import streamlit as st

from agent.graph import customeriq_graph


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CustomerIQ",
    page_icon="✦",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL
       ======================================================== */

    .stApp {
        background: #ffffff;
    }

    .block-container {
        max-width: 900px;
        padding-top: 1.5rem;
        padding-bottom: 8rem;
    }

    #MainMenu,
    footer,
    header {
        visibility: hidden;
    }


    /* ========================================================
       HEADER
       ======================================================== */

    .customeriq-header {
        display: flex;
        align-items: center;
        justify-content: space-between;

        padding: 0.5rem 0 1.25rem 0;
        margin-bottom: 1rem;

        border-bottom: 1px solid #eeeeee;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 11px;
    }

    .brand-icon {
        width: 38px;
        height: 38px;

        display: flex;
        align-items: center;
        justify-content: center;

        border-radius: 11px;

        background: #171717;
        color: #ffffff;

        font-size: 18px;
        font-weight: 600;
    }

    .brand-name {
        color: #171717;

        font-size: 20px;
        font-weight: 650;

        line-height: 1.1;
    }

    .brand-subtitle {
        color: #858585;

        font-size: 12px;
        font-weight: 400;

        margin-top: 3px;
    }

    .status {
        display: flex;
        align-items: center;
        gap: 6px;

        color: #777777;

        font-size: 12px;
    }

    .status-dot {
        width: 7px;
        height: 7px;

        border-radius: 50%;

        background: #22c55e;
    }


    /* ========================================================
       WELCOME
       ======================================================== */

    .welcome {
        text-align: center;

        padding: 4.5rem 1rem 2rem 1rem;
    }

    .welcome-icon {
        width: 58px;
        height: 58px;

        display: flex;
        align-items: center;
        justify-content: center;

        margin: 0 auto 1.5rem auto;

        border-radius: 18px;

        background: #f5f5f5;

        color: #171717;

        font-size: 27px;
        font-weight: 600;
    }

    .welcome-title {
        margin: 0;

        color: #171717;

        font-size: 32px;
        font-weight: 650;

        letter-spacing: -0.8px;
    }

    .welcome-description {
        max-width: 600px;

        margin: 12px auto 0 auto;

        color: #777777;

        font-size: 15px;
        line-height: 1.6;
    }


    /* ========================================================
       SUGGESTIONS
       ======================================================== */

    .suggestion-title {
        text-align: center;

        margin: 1.5rem 0 0.8rem 0;

        color: #999999;

        font-size: 12px;
        font-weight: 500;

        text-transform: uppercase;
        letter-spacing: 0.06em;
    }


    /* ========================================================
       STREAMLIT BUTTONS
       ======================================================== */

    .stButton > button {
        min-height: 62px;

        border: 1px solid #e7e7e7;
        border-radius: 14px;

        background: #ffffff;

        color: #333333;

        font-size: 14px;
        font-weight: 450;

        transition: all 0.15s ease;
    }

    .stButton > button:hover {
        border-color: #cccccc;

        background: #fafafa;

        color: #171717;
    }


    /* ========================================================
       CHAT
       ======================================================== */

    .user-message {
        display: flex;
        justify-content: flex-end;

        margin: 1.5rem 0;
    }

    .user-bubble {
        max-width: 75%;

        padding: 12px 16px;

        border-radius: 18px;

        background: #f2f2f2;

        color: #222222;

        font-size: 15px;
        line-height: 1.5;
    }

    .assistant-label {
        margin-top: 1.5rem;
        margin-bottom: 8px;

        color: #333333;

        font-size: 13px;
        font-weight: 600;
    }

    .assistant-message {
        color: #242424;

        font-size: 15px;
        line-height: 1.7;
    }


    /* ========================================================
       AGENT ACTIVITY
       ======================================================== */

    .agent-trace {
        margin-top: 1.5rem;
        padding-top: 1rem;

        border-top: 1px solid #eeeeee;
    }

    .trace-title {
        margin-bottom: 8px;

        color: #999999;

        font-size: 11px;
        font-weight: 600;

        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .trace-step {
        display: inline-block;

        margin-right: 6px;
        margin-bottom: 5px;

        padding: 5px 10px;

        border: 1px solid #eeeeee;
        border-radius: 20px;

        background: #fafafa;

        color: #666666;

        font-size: 11px;
    }


    /* ========================================================
       CHAT INPUT
       ======================================================== */

    div[data-testid="stChatInput"] {
        padding-bottom: 1rem;
    }

    div[data-testid="stChatInput"] textarea {
        border-radius: 18px !important;

        border: 1px solid #dddddd !important;

        background: #ffffff !important;

        box-shadow:
            0 2px 10px rgba(0, 0, 0, 0.04);

        padding: 12px 16px !important;
    }

    div[data-testid="stChatInput"] textarea:focus {
        border-color: #bbbbbb !important;

        box-shadow:
            0 2px 12px rgba(0, 0, 0, 0.06);
    }


    /* ========================================================
       MOBILE
       ======================================================== */

    @media (max-width: 640px) {

        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .welcome {
            padding-top: 3rem;
        }

        .welcome-title {
            font-size: 26px;
        }

        .welcome-description {
            font-size: 14px;
        }

        .user-bubble {
            max-width: 88%;
        }

        .brand-subtitle {
            display: none;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# HEADER
# ============================================================

st.html(
    """
    <div class="customeriq-header">

        <div class="brand">

            <div class="brand-icon">
                ✦
            </div>

            <div>
                <div class="brand-name">
                    CustomerIQ
                </div>

                <div class="brand-subtitle">
                    AI Customer Retention Intelligence
                </div>
            </div>

        </div>

        <div class="status">
            <span class="status-dot"></span>
            Online
        </div>

    </div>
    """
)


# ============================================================
# WELCOME SCREEN
# ============================================================

if not st.session_state.messages:

    st.html(
        """
        <div class="welcome">

            <div class="welcome-icon">
                ✦
            </div>

            <h1 class="welcome-title">
                How can I help with your customers?
            </h1>

            <p class="welcome-description">
                Analyse churn risk, understand customer behaviour,
                and get AI-powered retention recommendations.
            </p>

        </div>
        """
    )

    st.html(
        """
        <div class="suggestion-title">
            Try asking
        </div>
        """
    )

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "Analyse a customer",
            use_container_width=True,
        ):

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": (
                        "Analyse customer C0021 "
                        "and recommend what we should do."
                    ),
                }
            )

            st.rerun()

        if st.button(
            "Explain churn risk",
            use_container_width=True,
        ):

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": (
                        "Explain the churn risk "
                        "of customer C0021."
                    ),
                }
            )

            st.rerun()

    with col2:

        if st.button(
            "Find high-risk customers",
            use_container_width=True,
        ):

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": (
                        "What should we do about "
                        "high-risk customers?"
                    ),
                }
            )

            st.rerun()

        if st.button(
            "Recommend retention actions",
            use_container_width=True,
        ):

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": (
                        "What retention strategies "
                        "should we use for high-risk customers?"
                    ),
                }
            )

            st.rerun()


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    if message["role"] == "user":

        safe_content = html.escape(
            message["content"]
        )

        st.html(
            f"""
            <div class="user-message">

                <div class="user-bubble">
                    {safe_content}
                </div>

            </div>
            """
        )

    else:

        st.html(
            """
            <div class="assistant-label">
                ✦ CustomerIQ
            </div>
            """
        )

        st.markdown(
            message["content"]
        )

        if message.get("tools_used"):

            st.html(
                """
                <div class="agent-trace">

                    <div class="trace-title">
                        Agent activity
                    </div>

                </div>
                """
            )

            tool_labels = {
                "get_customer_data":
                    "✓ Customer data",

                "search_business_knowledge":
                    "✓ Business knowledge (RAG)",
            }

            for tool in message["tools_used"]:

                label = tool_labels.get(
                    tool,
                    f"✓ {tool}",
                )

                st.html(
                    f"""
                    <span class="trace-step">
                        {html.escape(label)}
                    </span>
                    """
                )


# ============================================================
# CHAT INPUT
# ============================================================

prompt = st.chat_input(
    "Ask CustomerIQ anything..."
)


if prompt:

    # --------------------------------------------------------
    # Store user message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    # --------------------------------------------------------
    # Display user message
    # --------------------------------------------------------

    safe_prompt = html.escape(prompt)

    st.html(
        f"""
        <div class="user-message">

            <div class="user-bubble">
                {safe_prompt}
            </div>

        </div>
        """
    )

    # --------------------------------------------------------
    # Generate response
    # --------------------------------------------------------

    with st.spinner(
        "CustomerIQ is analysing..."
    ):

        try:

            result = customeriq_graph.invoke(
                {
                    "user_query": prompt
                }
            )

            response = result.get(
                "final_response",
                "I couldn't generate a response.",
            )

            tools_used = result.get(
                "tools_used",
                [],
            )

        except Exception as error:

            st.error("CustomerIQ encountered an error:")
            st.exception(error)

            response = (
                "I couldn't process that request. "
                "Please try again."
            )

            tools_used = []

    # --------------------------------------------------------
    # Store assistant response
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
            "tools_used": tools_used,
        }
    )

    # --------------------------------------------------------
    # Display assistant response
    # --------------------------------------------------------

    st.html(
        """
        <div class="assistant-label">
            ✦ CustomerIQ
        </div>
        """
    )

    st.markdown(response)

    # --------------------------------------------------------
    # Agent trace
    # --------------------------------------------------------

    if tools_used:

        st.html(
            """
            <div class="agent-trace">

                <div class="trace-title">
                    Agent activity
                </div>

            </div>
            """
        )

        tool_labels = {
            "get_customer_data":
                "✓ Customer data",

            "search_business_knowledge":
                "✓ Business knowledge (RAG)",
        }

        for tool in tools_used:

            label = tool_labels.get(
                tool,
                f"✓ {tool}",
            )

            st.html(
                f"""
                <span class="trace-step">
                    {html.escape(label)}
                </span>
                """
            )