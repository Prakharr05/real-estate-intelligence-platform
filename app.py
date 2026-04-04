import os
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from database import SessionLocal
from models import Society, BuilderFloor, Plot
from openai import OpenAI
from ingest import (
    ingest_document, extract_text_from_bytes,
    retrieve_chunks, retrieve_from_all,
    get_collection_stats, COLLECTION_MAP
)
import re

# ============================================================
# 1. PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Alpha Intelligence | Deal Finder", layout="wide")
st.title("🏙️ Real Estate Intelligence Platform")
st.markdown("Transforming raw property data into actionable investment **Alpha**.")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ============================================================
# 2. TABS
# ============================================================
tab1, tab2 = st.tabs(["📊 Market Intelligence", "🤖 Document Chat"])

# ============================================================
# 3. SHARED UTILITIES
# ============================================================
@st.cache_resource
def load_model_artifacts(category_key):
    try:
        model_path   = os.path.join(MODELS_DIR, f"{category_key}_model.pkl")
        encoder_path = os.path.join(MODELS_DIR, f"{category_key}_encoder.pkl")
        avg_path     = os.path.join(MODELS_DIR, f"{category_key}_sector_avg.pkl")
        with open(model_path,   'rb') as f: model      = pickle.load(f)
        with open(encoder_path, 'rb') as f: encoder    = pickle.load(f)
        with open(avg_path,     'rb') as f: sector_avg = pickle.load(f)
        return model, encoder, sector_avg
    except FileNotFoundError:
        return None, None, None

@st.cache_data(ttl=60)
def get_data(category):
    db = SessionLocal()
    if category == "Societies":        query = db.query(Society)
    elif category == "Builder Floors": query = db.query(BuilderFloor)
    else:                              query = db.query(Plot)
    df = pd.read_sql(query.statement, db.bind)
    db.close()
    return df

def extract_number(val):
    if pd.isna(val): return 0.0
    nums = re.findall(r'[\d\.]+', str(val).replace(',', ''))
    try:    return float(nums[0]) if nums else 0.0
    except: return 0.0

def categorize_phase(score, category):
    thresholds = {
        "Plots":          (7.0, 4.0, 2.0),
        "Builder Floors": (7.5, 5.0, 2.5),
        "Societies":      (8.0, 5.5, 3.0),
    }
    e, p, g = thresholds.get(category, (8.0, 5.0, 2.5))
    if score >= e:    return "🌟 ELITE"
    elif score >= p:  return "🔥 PRIME"
    elif score >= g:  return "📈 GROWTH"
    else:             return "🌱 EMERGING"

CATEGORY_KEY_MAP = {
    "Plots": "plots", "Builder Floors": "floors", "Societies": "societies"}

# ============================================================
# TAB 1 — MARKET INTELLIGENCE (unchanged from previous version)
# ============================================================
with tab1:

    # ── Sidebar filters ──────────────────────────────────────
    st.sidebar.header("Market Filters")
    cat_choice = st.sidebar.selectbox(
        "Select Asset Class", ["Plots", "Societies", "Builder Floors"])
    df           = get_data(cat_choice)
    category_key = CATEGORY_KEY_MAP[cat_choice]

    if not df.empty:
        min_alpha       = st.sidebar.slider("Minimum Connectivity Score", 0.0, 10.0, 0.0)
        alpha_filter    = st.sidebar.multiselect(
            "Alpha Rating", ["HIGH", "STABLE", "VALUE"],
            default=["HIGH", "STABLE", "VALUE"])
        sectors         = ["All"] + sorted(df['sector'].dropna().unique().tolist())
        selected_sector = st.sidebar.selectbox("Sector", sectors)
        valid_prices    = df['price'].dropna()
        if not valid_prices.empty:
            min_p, max_p = int(valid_prices.min()), int(valid_prices.max())
            price_range  = st.sidebar.slider(
                "Price Range (₹)", min_p, max_p, (min_p, max_p),
                step=500000, format="₹%d")
        else:
            price_range = (0, 999999999)
        bhk_filter = None
        if cat_choice in ["Societies", "Builder Floors"] and 'bhk_type' in df.columns:
            bhk_options = ["All"] + sorted(df['bhk_type'].dropna().unique().tolist())
            bhk_filter  = st.sidebar.selectbox("BHK Type", bhk_options)
    else:
        min_alpha, alpha_filter = 0.0, ["HIGH", "STABLE", "VALUE"]
        selected_sector, price_range, bhk_filter = "All", (0, 999999999), None

    # ── Data engineering ─────────────────────────────────────
    if not df.empty:
        df['market_phase']  = df['connectivity_score'].apply(
            lambda x: categorize_phase(x, cat_choice))
        df['price_numeric'] = df['price'].apply(extract_number)
        df['display_price'] = df['price_numeric'].apply(
            lambda x: f"₹ {x:,.0f}" if x > 0 else "Price Hidden")
        if cat_choice == "Plots" and 'plot_area_sqyd' in df.columns:
            df['area_numeric']      = df['plot_area_sqyd'].apply(extract_number)
            df['price_per_sqyd']    = np.where(
                df['area_numeric'] > 0, df['price_numeric'] / df['area_numeric'], 0)
            df['efficiency_metric'] = df['price_per_sqyd'].apply(
                lambda x: f"₹ {x:,.0f} / sq.yd" if x > 0 else "Area Not Listed")
        else:
            df['efficiency_metric'] = df['price_per_sqft'].apply(
                lambda x: f"₹ {x:,.0f} / sq.ft"
                if pd.notna(x) and x > 0 else "Rate Not Listed")

    # ── Apply filters ─────────────────────────────────────────
    if not df.empty:
        filtered_df = df[
            (df['connectivity_score'] >= min_alpha) &
            (df['price'].fillna(0) >= price_range[0]) &
            (df['price'].fillna(0) <= price_range[1]) &
            (df['alpha_rating'].isin(alpha_filter))
        ]
        if selected_sector != "All":
            filtered_df = filtered_df[filtered_df['sector'] == selected_sector]
        if bhk_filter and bhk_filter != "All":
            filtered_df = filtered_df[filtered_df['bhk_type'] == bhk_filter]
    else:
        filtered_df = df.copy()

    # ── Summary metrics ───────────────────────────────────────
    if not filtered_df.empty:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Listings", len(filtered_df))
        m2.metric("Avg Connectivity Score",
                  f"{filtered_df['connectivity_score'].mean():.2f}")
        m3.metric("Avg Price",
                  f"₹ {filtered_df['price'].mean():,.0f}"
                  if filtered_df['price'].notna().any() else "N/A")
        if cat_choice == "Plots" and 'price_per_sqyd' in filtered_df:
            avg_r = filtered_df['price_per_sqyd'].mean()
            m4.metric("Avg Rate", f"₹ {avg_r:,.0f} / sq.yd" if avg_r > 0 else "N/A")
        elif 'price_per_sqft' in filtered_df.columns:
            avg_r = filtered_df['price_per_sqft'].mean()
            m4.metric("Avg Rate", f"₹ {avg_r:,.0f} / sq.ft" if avg_r > 0 else "N/A")

    # ── Map + Top Alpha ───────────────────────────────────────
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader(f"Investment Heatmap: {cat_choice}")
        map_df = filtered_df.dropna(subset=['latitude', 'longitude'])
        if not map_df.empty:
            layer = pdk.Layer(
                "ColumnLayer", data=map_df,
                get_position="[longitude, latitude]",
                get_elevation="connectivity_score * 300", radius=250,
                get_fill_color=["255 - (connectivity_score * 25)",
                                "connectivity_score * 25", 150, 200],
                pickable=True, auto_highlight=True,
            )
            st.pydeck_chart(pdk.Deck(
                layers=[layer],
                initial_view_state=pdk.ViewState(
                    latitude=28.3670, longitude=77.3450, zoom=10.5, pitch=45),
                tooltip={"text": "{name}\n{title}\nPhase: {market_phase}\n"
                                 "Score: {connectivity_score}"}
            ))
        else:
            st.info("⚠️ No lat/lon data yet — run scoring.py to populate coordinates.")

    with col2:
        st.subheader("Top Alpha Opportunities")
        top_props = filtered_df.sort_values(
            by="connectivity_score", ascending=False).head(8)
        for _, row in top_props.iterrows():
            with st.expander(
                f"{row['market_phase']} | {row['sector']} "
                f"(Score: {row['connectivity_score']})"):
                st.write(f"**Name:** {row.get('name', 'Unknown')}")
                st.write(f"**Title:** {row['title']}")
                st.write(f"**Price:** {row['display_price']}")
                st.write(f"**Rate:** {row['efficiency_metric']}")
                st.write(f"**Alpha Rating:** {row.get('alpha_rating', 'N/A')}")
                st.write(f"**Infra Proximity:** {row.get('infra_proximity_km', 0.0)} km")
                if cat_choice in ["Societies", "Builder Floors"] and 'bhk_type' in row:
                    st.write(f"**BHK:** {row['bhk_type']}")
                st.progress(float(row['connectivity_score']) / 10)

    # ── Market inventory table ────────────────────────────────
    st.divider()
    st.subheader("Market Inventory (Live Deal Finder)")
    if not filtered_df.empty:
        base_cols = ['name', 'title', 'sector', 'display_price',
                     'efficiency_metric', 'connectivity_score',
                     'alpha_rating', 'market_phase', 'infra_proximity_km']
        if cat_choice == "Plots":
            base_cols.insert(4, 'plot_area_sqyd')
        elif cat_choice in ["Societies", "Builder Floors"]:
            if 'bhk_type'  in filtered_df.columns: base_cols.insert(4, 'bhk_type')
            if 'area_sqft' in filtered_df.columns: base_cols.insert(5, 'area_sqft')
        display_cols = filtered_df[[c for c in base_cols if c in filtered_df.columns]]

        def highlight_alpha(row):
            phase = row['market_phase']
            if "ELITE"   in phase: return ['background-color: rgba(16, 185, 129, 0.2)'] * len(row)
            elif "PRIME"  in phase: return ['background-color: rgba(56, 189, 248, 0.2)'] * len(row)
            elif "GROWTH" in phase: return ['background-color: rgba(245, 158, 11, 0.2)'] * len(row)
            else:                   return [''] * len(row)

        st.dataframe(display_cols.style.apply(highlight_alpha, axis=1),
                     use_container_width=True, hide_index=True)
        st.download_button("⬇️ Export to CSV",
                           data=display_cols.to_csv(index=False),
                           file_name=f"{category_key}_alpha_data.csv",
                           mime="text/csv")
    else:
        st.info("No properties match the current filters.")

    # ── Deal Scorer ───────────────────────────────────────────
    st.divider()
    st.subheader("🔍 Deal Scorer — Is This a Good Deal?")
    st.markdown("Enter a listing you've found and we'll tell you how it compares to market.")

    model, encoder, sector_avg = load_model_artifacts(category_key)

    if model is None:
        st.warning(f"⚠️ No trained model found for **{cat_choice}**. "
                   f"Run `python train.py` first.")
    else:
        known_sectors = list(encoder.classes_)
        with st.form("deal_scorer_form"):
            fc1, fc2 = st.columns(2)
            with fc1:
                input_sector = st.selectbox("Sector", known_sectors)
                sector_scores = df[df['sector'] == input_sector]['connectivity_score']
                input_conn    = float(sector_scores.mean()) if not sector_scores.empty else 5.0
                st.info(f"📡 Connectivity Score for {input_sector}: **{input_conn:.2f}** (auto)")
                input_price  = st.number_input(
                    "Total Price (₹)", min_value=100000,
                    max_value=500000000, value=5000000, step=100000)
            with fc2:
                if cat_choice == "Plots":
                    input_area   = st.number_input(
                        "Plot Area (sq.yd)", min_value=10.0,
                        max_value=5000.0, value=100.0, step=5.0)
                    input_corner = st.selectbox("Corner Plot?", ["No", "Yes"])
                else:
                    input_area   = st.number_input(
                        "Area (sq.ft)", min_value=100.0,
                        max_value=10000.0, value=1000.0, step=50.0)
                    input_bhk    = st.selectbox(
                        "BHK Type", [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
                    if cat_choice == "Societies":
                        input_possession = st.selectbox(
                            "Possession Status",
                            ["Ready to Move", "Under Construction"])
            submitted = st.form_submit_button("🧮 Score This Deal")

        if submitted:
            sector_enc = encoder.transform([input_sector])[0]
            if cat_choice == "Plots":
                features  = np.array([[input_area, input_conn,
                                        1 if input_corner == "Yes" else 0, sector_enc]])
                user_rate = input_price / (input_area * 9)
            elif cat_choice == "Builder Floors":
                features  = np.array([[input_area, float(input_bhk), input_conn, sector_enc]])
                user_rate = input_price / input_area
            else:
                features  = np.array([[input_area, float(input_bhk), input_conn,
                                        1 if input_possession == "Ready to Move" else 0,
                                        sector_enc]])
                user_rate = input_price / input_area

            predicted_rate = float(model.predict(features)[0])
            sector_rate    = sector_avg.get(input_sector, predicted_rate)
            diff_ml        = ((user_rate - predicted_rate) / predicted_rate) * 100
            diff_sec       = ((user_rate - sector_rate)    / sector_rate)    * 100
            DEAL_THRESHOLD = 8.0

            if diff_ml < -DEAL_THRESHOLD:   label, label_color = "🟢 Good Deal", "green"
            elif diff_ml > DEAL_THRESHOLD:  label, label_color = "🔴 Overpriced", "red"
            else:                           label, label_color = "🟡 Fair Price", "orange"

            st.divider()
            rate_unit = "sq.yd" if cat_choice == "Plots" else "sq.ft"
            r1, r2, r3 = st.columns(3)
            r1.metric("Your Rate",       f"₹ {user_rate:,.0f} / {rate_unit}")
            r2.metric("ML Fair Rate",    f"₹ {predicted_rate:,.0f} / {rate_unit}",
                      delta=f"{diff_ml:+.1f}% vs fair value",  delta_color="inverse")
            r3.metric("Sector Avg Rate", f"₹ {sector_rate:,.0f} / {rate_unit}",
                      delta=f"{diff_sec:+.1f}% vs sector avg", delta_color="inverse")

            st.markdown(f"""
            <div style="border:2px solid {label_color}; border-radius:12px;
                        padding:20px; margin-top:16px; text-align:center;
                        background-color:rgba(0,0,0,0.03)">
                <h2 style="color:{label_color}; margin:0">{label}</h2>
                <p style="margin:8px 0 0 0; font-size:15px">
                    This property is priced
                    <strong>{abs(diff_ml):.1f}%
                    {'below' if diff_ml < 0 else 'above'}</strong>
                    the ML-predicted fair market rate<br>and
                    <strong>{abs(diff_sec):.1f}%
                    {'below' if diff_sec < 0 else 'above'}</strong>
                    the {input_sector} average.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### 📊 Comparable Listings in Dataset")
            similar = filtered_df[filtered_df['sector'] == input_sector].copy()
            if cat_choice in ["Societies", "Builder Floors"] and 'bhk_type' in similar.columns:
                similar = similar[similar['bhk_type'] == str(input_bhk)]
            if similar.empty:
                st.info(f"No comparable listings found in {input_sector}.")
            else:
                show_cols = ['name', 'title', 'display_price',
                             'efficiency_metric', 'connectivity_score', 'alpha_rating']
                st.dataframe(similar[[c for c in show_cols if c in similar.columns]].head(8),
                             use_container_width=True, hide_index=True)

# ============================================================
# TAB 2 — DOCUMENT CHAT (RAG)
# ============================================================
with tab2:
    st.subheader("🤖 Property Document Chat")
    st.markdown(
        "Upload property documents (brochures, RERA certificates, legal/title docs) "
        "and ask questions about them.")

    # ── Session state init ────────────────────────────────────
    if "chat_history"    not in st.session_state: st.session_state.chat_history    = []
    if "ingested_docs"   not in st.session_state: st.session_state.ingested_docs   = []
    if "active_doc_type" not in st.session_state: st.session_state.active_doc_type = "all"

    # ── Layout: uploader left, chat right ────────────────────
    up_col, chat_col = st.columns([1, 2])

    with up_col:
        st.markdown("#### 📁 Upload Documents")

        doc_type_label = st.selectbox(
            "Document Type",
            ["Property Brochure", "RERA Certificate", "Legal / Title Document"],
            key="doc_type_select"
        )
        DOC_TYPE_MAP = {
            "Property Brochure":        "brochure",
            "RERA Certificate":         "rera",
            "Legal / Title Document":   "legal",
        }
        selected_doc_type = DOC_TYPE_MAP[doc_type_label]

        property_name = st.text_input(
            "Property / Project Name (optional)",
            placeholder="e.g. Adore Happy Homes",
            key="property_name_input"
        )

        uploaded_file = st.file_uploader(
            "Upload PDF", type=["pdf"], key="pdf_uploader")

        if uploaded_file and st.button("📥 Ingest Document"):
            with st.spinner("Extracting and embedding document..."):
                try:
                    pdf_bytes = uploaded_file.read()
                    pages     = extract_text_from_bytes(pdf_bytes)

                    if not pages:
                        st.error("❌ Could not extract text from this PDF. "
                                 "It may be scanned/image-based.")
                    else:
                        result = ingest_document(
                            doc_type=selected_doc_type,
                            filename=uploaded_file.name,
                            pages=pages,
                            property_name=property_name or "Unknown"
                        )
                        if result["status"] == "success":
                            st.session_state.ingested_docs.append({
                                "filename":      uploaded_file.name,
                                "doc_type":      selected_doc_type,
                                "property_name": property_name or "Unknown",
                                "chunks":        result["chunks"],
                                "pages":         result["pages"],
                            })
                            st.success(
                                f"✅ Ingested **{uploaded_file.name}** — "
                                f"{result['chunks']} chunks across "
                                f"{result['pages']} pages")
                        else:
                            st.error(f"❌ {result['message']}")
                except Exception as e:
                    st.error(f"❌ Error during ingestion: {e}")

        # ── Ingested docs list ────────────────────────────────
        if st.session_state.ingested_docs:
            st.markdown("---")
            st.markdown("**Ingested this session:**")
            for doc in st.session_state.ingested_docs:
                st.markdown(
                    f"- 📄 `{doc['filename']}` "
                    f"({doc['doc_type'].upper()}) — "
                    f"{doc['chunks']} chunks")

        # ── Collection stats ──────────────────────────────────
        st.markdown("---")
        st.markdown("**Vector Store Status:**")
        stats = get_collection_stats()
        for dtype, count in stats.items():
            icon  = "🟢" if count > 0 else "⚪"
            st.markdown(f"{icon} `{dtype}`: {count} chunks")

        # ── Search scope selector ─────────────────────────────
        st.markdown("---")
        st.markdown("**Search Scope:**")
        scope_options = ["All Documents"] + [
            f"{k.capitalize()} only"
            for k, v in stats.items() if v > 0
        ]
        scope_choice = st.selectbox(
            "Search in", scope_options, key="scope_select")
        if scope_choice == "All Documents":
            st.session_state.active_doc_type = "all"
        else:
            st.session_state.active_doc_type = scope_choice.split(" ")[0].lower()

        # ── Clear chat ────────────────────────────────────────
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

    # ── Chat interface ────────────────────────────────────────
    with chat_col:
        st.markdown("#### 💬 Ask About Your Documents")

        # Check if any docs are ingested
        stats = get_collection_stats()
        total_chunks = sum(stats.values())

        if total_chunks == 0:
            st.info("👈 Upload and ingest a document first to start chatting.")
        else:
            # Display chat history
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    # Show sources for assistant messages
                    if msg["role"] == "assistant" and "sources" in msg:
                        with st.expander("📎 Source Chunks Used"):
                            for i, src in enumerate(msg["sources"]):
                                st.markdown(
                                    f"**Chunk {i+1}** — "
                                    f"`{src['metadata'].get('filename', 'unknown')}` "
                                    f"(Page {src['metadata'].get('page_num', '?')}, "
                                    f"Type: {src['metadata'].get('doc_type', '?').upper()}) "
                                    f"| Similarity: {1 - src['distance']:.2f}")
                                st.caption(src["text"][:300] + "...")
                                st.divider()

            # Chat input
            user_input = st.chat_input(
                "Ask anything about your uploaded documents...")

            if user_input:
                # Add user message to history
                st.session_state.chat_history.append({
                    "role": "user", "content": user_input})

                with st.chat_message("user"):
                    st.markdown(user_input)

                with st.chat_message("assistant"):
                    with st.spinner("Searching documents..."):

                        # ── Retrieve relevant chunks ──────────
                        active_type = st.session_state.active_doc_type
                        if active_type == "all":
                            chunks = retrieve_from_all(user_input, n_results=5)
                        else:
                            chunks = retrieve_chunks(
                                user_input, active_type, n_results=5)

                        if not chunks:
                            response = (
                                "I couldn't find relevant information in the "
                                "uploaded documents for your question. Please "
                                "make sure the relevant document is ingested.")
                            st.markdown(response)
                            st.session_state.chat_history.append({
                                "role": "assistant", "content": response})
                        else:
                            # ── Build context from chunks ─────
                            context_parts = []
                            for i, chunk in enumerate(chunks):
                                meta = chunk["metadata"]
                                context_parts.append(
                                    f"[Source {i+1} | "
                                    f"File: {meta.get('filename', 'unknown')} | "
                                    f"Page: {meta.get('page_num', '?')} | "
                                    f"Type: {meta.get('doc_type', '?').upper()}]\n"
                                    f"{chunk['text']}"
                                )
                            context = "\n\n---\n\n".join(context_parts)

                            # ── System prompt ─────────────────
                            system_prompt = """You are an expert real estate analyst 
specializing in the Faridabad property market, with deep knowledge of:
- Property valuation and investment analysis
- RERA regulations and compliance in Haryana
- Legal aspects of property transactions in India
- Builder floor and society market dynamics in NCR

You answer questions based ONLY on the provided document context.
If the answer is not in the context, say so clearly.
Always cite which source/page your answer comes from.
Keep answers concise, factual and investment-focused."""

                            # ── Call GPT-4o ───────────────────
                            messages = [
                                {"role": "system", "content": system_prompt},
                            ]
                            # Add last 6 messages for context
                            for h in st.session_state.chat_history[-6:]:
                                messages.append({
                                    "role":    h["role"],
                                    "content": h["content"]
                                })
                            # Add retrieved context
                            messages.append({
                                "role": "user",
                                "content": (
                                    f"Context from documents:\n\n{context}\n\n"
                                    f"Question: {user_input}"
                                )
                            })

                            completion = openai_client.chat.completions.create(
                                model="gpt-4o",
                                messages=messages,
                                temperature=0.2,
                                max_tokens=800,
                            )
                            response = completion.choices[0].message.content

                            st.markdown(response)

                            # Show sources inline
                            with st.expander("📎 Source Chunks Used"):
                                for i, src in enumerate(chunks):
                                    st.markdown(
                                        f"**Chunk {i+1}** — "
                                        f"`{src['metadata'].get('filename', 'unknown')}` "
                                        f"(Page {src['metadata'].get('page_num', '?')}) "
                                        f"| Similarity: {1 - src['distance']:.2f}")
                                    st.caption(src["text"][:300] + "...")
                                    st.divider()

                            # Save to history with sources
                            st.session_state.chat_history.append({
                                "role":    "assistant",
                                "content": response,
                                "sources": chunks,
                            })