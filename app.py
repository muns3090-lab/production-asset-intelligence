import streamlit as st
import anthropic
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
import os

load_dotenv()

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Production Asset Intelligence",
    page_icon="🎬",
    layout="wide"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    /* Base background */
    .stApp {
        background: linear-gradient(160deg, #0d1117 0%, #0f1923 50%, #0d1f2d 100%);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0a1628;
    }

    /* All default text → light */
    html, body, [class*="css"], .stMarkdown, p, li, span, label {
        color: #e2e8f0 !important;
    }

    /* Headers */
    h1 { color: #f0c040 !important; font-size: 2.4em !important; letter-spacing: 2px; }
    h2 { color: #f0c040 !important; }
    h3 { color: #7dd3c8 !important; }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.04);
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8 !important;
        background: transparent;
        border-radius: 8px;
        font-weight: 600;
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #f0c040, #d4a017) !important;
        color: #0d1117 !important;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(240,192,64,0.1), rgba(125,211,200,0.06));
        border: 1px solid rgba(240,192,64,0.25);
        border-radius: 14px;
        padding: 22px 16px;
        text-align: center;
    }
    .metric-card h2 { font-size: 2.2em !important; margin: 0 !important; }
    .metric-card p { color: #94a3b8 !important; margin: 4px 0 0 !important; font-size: 0.9em; }

    /* Ticket cards */
    .ticket-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 14px 18px;
        margin: 8px 0;
    }
    .ticket-card strong { color: #f0c040 !important; }
    .ticket-card span { color: #94a3b8 !important; }
    .ticket-card small { color: #64748b !important; }

    .priority-critical { border-left: 4px solid #f87171; }
    .priority-high     { border-left: 4px solid #fb923c; }
    .priority-medium   { border-left: 4px solid #facc15; }
    .priority-low      { border-left: 4px solid #4ade80; }

    /* Form inputs */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div {
        background: rgba(255,255,255,0.07) !important;
        border: 1px solid rgba(240,192,64,0.3) !important;
        color: #e2e8f0 !important;
        border-radius: 8px !important;
    }

    /* Primary button */
    .stButton > button, .stFormSubmitButton > button {
        background: linear-gradient(135deg, #f0c040, #d4a017) !important;
        color: #0d1117 !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 28px !important;
        font-size: 1em !important;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        background: linear-gradient(135deg, #ffd966, #f0c040) !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(240,192,64,0.35);
    }

    /* Chat bubbles */
    .stChatMessage {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.07) !important;
        border-radius: 12px !important;
    }

    /* Dataframe */
    .stDataFrame { border-radius: 10px; overflow: hidden; }
    .stDataFrame thead { background: rgba(240,192,64,0.15) !important; }

    /* Success / info boxes */
    .stSuccess { background: rgba(74,222,128,0.12) !important; border-color: #4ade80 !important; color: #bbf7d0 !important; }
    .stInfo    { background: rgba(125,211,200,0.1) !important; border-color: #7dd3c8 !important; color: #ccfbf1 !important; }

    /* Divider */
    hr { border-color: rgba(240,192,64,0.2) !important; }

    /* Spinner */
    .stSpinner > div { border-top-color: #f0c040 !important; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0d1117; }
    ::-webkit-scrollbar-thumb { background: rgba(240,192,64,0.3); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Load Data ────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("data/sample_assets.csv")

df = load_data()

# ── Anthropic Client ─────────────────────────────────────────
api_key = st.secrets.get("ANTHROPIC_API_KEY") if hasattr(st, "secrets") else None
api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=api_key)

def analyze_asset_request(title, category, department, priority, description):
    prompt = f"""You are an AI assistant for a major Hollywood studio's production asset management system.

Analyze this asset request and provide:
1. **Triage Assessment** (2-3 sentences on urgency and risk if not fulfilled)
2. **Routing Recommendation** (which team/vendor should handle this)
3. **Estimated Lead Time** (realistic timeline for procurement/fulfillment)
4. **Risk Flags** (any compliance, safety, or scheduling concerns)
5. **Similar Past Requests** (suggest if this seems like a recurring need)

Asset Request:
- Title: {title}
- Category: {category}
- Department: {department}
- Priority: {priority}
- Description: {description}

Be concise, practical, and production-aware. Use film industry terminology."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

def chat_with_ai(messages, user_message, df_context):
    system_prompt = f"""You are PAI Assistant — an AI for a Hollywood studio's Production Asset Intelligence system.

You have access to the current asset database summary:
- Total Requests: {len(df_context)}
- Open: {len(df_context[df_context['status']=='Open'])}
- In Progress: {len(df_context[df_context['status']=='In Progress'])}
- Resolved: {len(df_context[df_context['status']=='Resolved'])}
- Critical Items: {len(df_context[df_context['priority']=='Critical'])}

Recent open requests:
{df_context[df_context['status']=='Open'][['id','title','category','priority']].to_string()}

Answer questions about production assets, request status, priorities, and studio operations.
Be helpful, concise, and use film industry knowledge."""

    messages_for_api = [{"role": m["role"], "content": m["content"]} for m in messages]
    messages_for_api.append({"role": "user", "content": user_message})

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        system=system_prompt,
        messages=messages_for_api
    )
    return response.content[0].text

# ── Header ───────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 28px 0 10px;'>
    <div style='font-size:3em; margin-bottom:6px;'>🎬</div>
    <h1 style='margin:0; font-size:2.4em; letter-spacing:3px; 
               background: linear-gradient(135deg, #f0c040, #7dd3c8);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
        PRODUCTION ASSET INTELLIGENCE
    </h1>
    <p style='color:#64748b !important; font-size:1.05em; margin-top:8px; letter-spacing:1px;'>
        AI-powered asset request management for studio productions
    </p>
    <div style='width:80px; height:2px; background:linear-gradient(90deg,#f0c040,#7dd3c8); 
                margin:14px auto 0;'></div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Navigation ───────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊  Dashboard", "📋  Asset Requests", "🤖  Submit & Analyze", "💬  PAI Assistant"])

# ══════════════════════════════════════════
# TAB 1: DASHBOARD
# ══════════════════════════════════════════
with tab1:
    st.markdown("### 📊 Production Overview")
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        (len(df), "#f0c040", "Total Requests", "📁"),
        (len(df[df['status']=='Open']), "#fb923c", "Open", "🔓"),
        (len(df[df['priority']=='Critical']), "#f87171", "Critical", "🚨"),
        (len(df[df['status']=='Resolved']), "#4ade80", "Resolved", "✅"),
    ]
    for col, (val, color, label, icon) in zip([col1,col2,col3,col4], metrics):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div style='font-size:1.8em; margin-bottom:4px;'>{icon}</div>
                <h2 style='color:{color} !important;'>{val}</h2>
                <p>{label}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        status_counts = df['status'].value_counts().reset_index()
        fig1 = px.pie(
            status_counts, values='count', names='status',
            title="Requests by Status",
            color_discrete_sequence=['#f0c040','#fb923c','#4ade80','#7dd3c8']
        )
        fig1.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'), title_font_color='#7dd3c8',
            legend=dict(font=dict(color='#e2e8f0'))
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col_b:
        cat_counts = df['category'].value_counts().reset_index()
        fig2 = px.bar(
            cat_counts, x='category', y='count',
            title="Requests by Category",
            color='count',
            color_continuous_scale=[[0,'#0d3b5e'],[0.5,'#f0c040'],[1,'#f87171']]
        )
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'), title_font_color='#7dd3c8',
            xaxis=dict(tickfont=dict(color='#94a3b8'), gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(tickfont=dict(color='#94a3b8'), gridcolor='rgba(255,255,255,0.05)')
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 🚨 Critical & High Priority Items")
    urgent = df[df['priority'].isin(['Critical','High'])].sort_values('priority')
    st.dataframe(
        urgent[['id','title','category','department','status','priority']],
        use_container_width=True, hide_index=True
    )

# ══════════════════════════════════════════
# TAB 2: ASSET REQUESTS
# ══════════════════════════════════════════
with tab2:
    st.markdown("### 📋 All Asset Requests")
    st.markdown("<br>", unsafe_allow_html=True)

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        status_filter = st.selectbox("Filter by Status", ["All"] + list(df['status'].unique()))
    with col_f2:
        priority_filter = st.selectbox("Filter by Priority", ["All"] + list(df['priority'].unique()))
    with col_f3:
        category_filter = st.selectbox("Filter by Category", ["All"] + list(df['category'].unique()))

    filtered = df.copy()
    if status_filter != "All":
        filtered = filtered[filtered['status'] == status_filter]
    if priority_filter != "All":
        filtered = filtered[filtered['priority'] == priority_filter]
    if category_filter != "All":
        filtered = filtered[filtered['category'] == category_filter]

    st.markdown(f"<p style='color:#64748b; font-size:0.9em;'>Showing {len(filtered)} of {len(df)} requests</p>", unsafe_allow_html=True)

    for _, row in filtered.iterrows():
        priority_class = f"priority-{row['priority'].lower()}"
        status_colors = {"Open": "#fb923c", "In Progress": "#7dd3c8", "Resolved": "#4ade80"}
        s_color = status_colors.get(row['status'], '#94a3b8')
        st.markdown(f"""
        <div class='ticket-card {priority_class}'>
            <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;'>
                <strong style='color:#f0c040; font-size:1em;'>{row['id']}</strong>
                <span style='background:rgba(255,255,255,0.08); padding:2px 10px; border-radius:20px;
                             color:{s_color}; font-size:0.8em; font-weight:600;'>{row['status']}</span>
            </div>
            <div style='color:#e2e8f0; font-weight:600; margin-bottom:5px;'>{row['title']}</div>
            <div style='color:#94a3b8; font-size:0.85em; margin-bottom:6px;'>
                📁 {row['category']} &nbsp;|&nbsp; 🏢 {row['department']} &nbsp;|&nbsp;
                ⚡ <span style='color:#f0c040'>{row['priority']}</span>
            </div>
            <div style='color:#64748b; font-size:0.82em;'>{row['description'][:130]}...</div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════
# TAB 3: SUBMIT & ANALYZE
# ══════════════════════════════════════════
with tab3:
    st.markdown("### 🤖 Submit New Asset Request")
    st.markdown("<p style='color:#94a3b8;'>Fill out the form below and get instant AI triage, routing, and risk analysis.</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    with st.form("asset_form"):
        col1, col2 = st.columns(2)
        with col1:
            title = st.text_input("Asset Title *", placeholder="e.g., Hero Motorcycle - Scene 8")
            category = st.selectbox("Category *", ["Props","Equipment","Costumes","Locations",
                                                    "Vehicles","Special Effects","Set Dressing","Other"])
            department = st.selectbox("Department *", ["Art Department","Cinematography","Wardrobe",
                                                        "Production","Stunt Coordination","Transportation",
                                                        "VFX/SFX","Set Construction","Other"])
        with col2:
            priority = st.selectbox("Priority *", ["Critical","High","Medium","Low"])
            submitted_by = st.text_input("Your Name *", placeholder="e.g., Jordan Lee")
            date_needed = st.date_input("Date Needed")

        description = st.text_area("Description *",
            placeholder="Describe the asset, specific requirements, scene context, quantities, etc.",
            height=130)
        submitted = st.form_submit_button("🤖 Analyze with AI →")

    if submitted and title and description:
        st.markdown("---")
        st.markdown("### 🎯 AI Triage Analysis")
        with st.spinner("Analyzing your request..."):
            analysis = analyze_asset_request(title, category, department, priority, description)
        st.markdown(f"""
        <div style='background:rgba(125,211,200,0.08); border:1px solid rgba(125,211,200,0.25);
                    border-radius:12px; padding:20px; color:#e2e8f0;'>
            {analysis.replace(chr(10), "<br>")}
        </div>
        """, unsafe_allow_html=True)
        st.success("✅ AI analysis complete. Request logged.")

# ══════════════════════════════════════════
# TAB 4: PAI ASSISTANT
# ══════════════════════════════════════════
with tab4:
    st.markdown("### 💬 PAI Assistant")
    st.markdown("<p style='color:#94a3b8;'>Ask anything about production assets, request status, or studio operations.</p>", unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Starter prompts
    if not st.session_state.chat_history:
        st.markdown("<p style='color:#64748b; font-size:0.85em; margin-bottom:8px;'>💡 Try asking:</p>", unsafe_allow_html=True)
        cols = st.columns(3)
        starters = [
            "What are our critical open requests?",
            "Which department has the most requests?",
            "What's the risk on PAI-007?"
        ]
        for col, prompt in zip(cols, starters):
            with col:
                st.markdown(f"""
                <div style='background:rgba(240,192,64,0.08); border:1px solid rgba(240,192,64,0.2);
                            border-radius:8px; padding:10px 12px; color:#94a3b8; font-size:0.85em;
                            cursor:pointer;'>💬 {prompt}</div>
                """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask about assets, priorities, or production needs...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("PAI thinking..."):
                reply = chat_with_ai(st.session_state.chat_history[:-1], user_input, df)
            st.markdown(reply)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})

# ── Footer ────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding:40px 0 20px; margin-top:40px;
            border-top:1px solid rgba(240,192,64,0.15);'>
    <p style='color:#334155 !important; font-size:0.8em; letter-spacing:1px;'>
        🎬 PRODUCTION ASSET INTELLIGENCE &nbsp;·&nbsp; Built by Sunil Sukumar &nbsp;·&nbsp; Powered by Claude AI
    </p>
</div>
""", unsafe_allow_html=True)