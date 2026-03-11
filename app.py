import streamlit as st
import anthropic
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
import os
from datetime import datetime

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
    .main { background-color: #0a0a0f; }
    .stApp { background: linear-gradient(135deg, #0a0a0f 0%, #1a0a2e 100%); }
    h1, h2, h3 { color: #e8c97a !important; }
    .metric-card {
        background: rgba(232, 201, 122, 0.08);
        border: 1px solid rgba(232, 201, 122, 0.3);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 5px;
    }
    .ticket-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 15px;
        margin: 8px 0;
    }
    .priority-critical { border-left: 4px solid #ff4444; }
    .priority-high { border-left: 4px solid #ff8800; }
    .priority-medium { border-left: 4px solid #ffcc00; }
    .stButton > button {
        background: linear-gradient(135deg, #e8c97a, #c9a227);
        color: #0a0a0f;
        font-weight: bold;
        border: none;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Data ────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("data/sample_assets.csv")

df = load_data()

# ── Anthropic Client ─────────────────────────────────────────
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

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
<div style='text-align:center; padding: 20px 0;'>
    <h1 style='font-size:2.5em; letter-spacing:2px;'>🎬 Production Asset Intelligence</h1>
    <p style='color:#888; font-size:1.1em;'>AI-powered asset request management for studio productions</p>
</div>
""", unsafe_allow_html=True)

# ── Navigation ───────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📋 Asset Requests", "🤖 Submit & Analyze", "💬 PAI Assistant"])

# ══════════════════════════════════════════
# TAB 1: DASHBOARD
# ══════════════════════════════════════════
with tab1:
    st.markdown("### Production Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class='metric-card'>
            <h2 style='color:#e8c97a'>{len(df)}</h2>
            <p style='color:#aaa'>Total Requests</p></div>""", unsafe_allow_html=True)
    with col2:
        open_count = len(df[df['status']=='Open'])
        st.markdown(f"""<div class='metric-card'>
            <h2 style='color:#ff8800'>{open_count}</h2>
            <p style='color:#aaa'>Open</p></div>""", unsafe_allow_html=True)
    with col3:
        crit_count = len(df[df['priority']=='Critical'])
        st.markdown(f"""<div class='metric-card'>
            <h2 style='color:#ff4444'>{crit_count}</h2>
            <p style='color:#aaa'>Critical</p></div>""", unsafe_allow_html=True)
    with col4:
        resolved_count = len(df[df['status']=='Resolved'])
        st.markdown(f"""<div class='metric-card'>
            <h2 style='color:#44ff88'>{resolved_count}</h2>
            <p style='color:#aaa'>Resolved</p></div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        status_counts = df['status'].value_counts().reset_index()
        fig1 = px.pie(status_counts, values='count', names='status',
                      title="Requests by Status",
                      color_discrete_sequence=['#e8c97a','#ff8800','#44ff88'])
        fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           font_color='white')
        st.plotly_chart(fig1, use_container_width=True)

    with col_b:
        cat_counts = df['category'].value_counts().reset_index()
        fig2 = px.bar(cat_counts, x='category', y='count',
                      title="Requests by Category",
                      color='count', color_continuous_scale='Oranges')
        fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           font_color='white', xaxis_tickangle=-30)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 🚨 Critical & High Priority Items")
    urgent = df[df['priority'].isin(['Critical','High'])].sort_values('priority')
    st.dataframe(urgent[['id','title','category','department','status','priority']],
                 use_container_width=True, hide_index=True)

# ══════════════════════════════════════════
# TAB 2: ASSET REQUESTS
# ══════════════════════════════════════════
with tab2:
    st.markdown("### All Asset Requests")

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

    for _, row in filtered.iterrows():
        priority_class = f"priority-{row['priority'].lower()}"
        st.markdown(f"""
        <div class='ticket-card {priority_class}'>
            <strong style='color:#e8c97a'>{row['id']}</strong> — {row['title']}<br>
            <span style='color:#aaa'>📁 {row['category']} | 🏢 {row['department']} |
            🔴 {row['priority']} | ⚡ {row['status']}</span><br>
            <small style='color:#666'>{row['description'][:120]}...</small>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════
# TAB 3: SUBMIT & ANALYZE
# ══════════════════════════════════════════
with tab3:
    st.markdown("### Submit New Asset Request")
    st.markdown("Fill out the form and get instant AI triage and routing analysis.")

    with st.form("asset_form"):
        col1, col2 = st.columns(2)
        with col1:
            title = st.text_input("Asset Title *", placeholder="e.g., Hero Motorcycle - Scene 8")
            category = st.selectbox("Category *", ["Props", "Equipment", "Costumes", "Locations",
                                                    "Vehicles", "Special Effects", "Set Dressing", "Other"])
            department = st.selectbox("Department *", ["Art Department", "Cinematography", "Wardrobe",
                                                        "Production", "Stunt Coordination", "Transportation",
                                                        "VFX/SFX", "Set Construction", "Other"])
        with col2:
            priority = st.selectbox("Priority *", ["Critical", "High", "Medium", "Low"])
            submitted_by = st.text_input("Your Name *", placeholder="e.g., Jordan Lee")
            date_needed = st.date_input("Date Needed")

        description = st.text_area("Description *", placeholder="Describe the asset, any specific requirements, scene context, quantities, etc.",
                                   height=120)
        submitted = st.form_submit_button("🤖 Submit & Analyze with AI")

    if submitted and title and description:
        st.markdown("---")
        st.markdown("### 🤖 AI Triage Analysis")
        with st.spinner("Analyzing your request..."):
            analysis = analyze_asset_request(title, category, department, priority, description)
        st.markdown(analysis)
        st.success("✅ Request submitted! AI analysis complete.")

# ══════════════════════════════════════════
# TAB 4: PAI ASSISTANT
# ══════════════════════════════════════════
with tab4:
    st.markdown("### 💬 Ask PAI Assistant")
    st.markdown("Ask anything about your production assets, request status, or studio operations.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

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
<div style='text-align:center; padding:30px 0 10px; color:#555; font-size:0.85em;'>
    Production Asset Intelligence | Built by Sunil Sukumar | Powered by Claude AI
</div>
""", unsafe_allow_html=True)