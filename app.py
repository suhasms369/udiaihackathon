# ============================================================
# Aadhaar Operational Intelligence Dashboard - FIXED
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
try:
    from google import genai
    NEW_GENAI = True
except ImportError:
    import google.generativeai as genai
    NEW_GENAI = False
import json
import os
import warnings
warnings.filterwarnings('ignore')

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Aadhaar Operational Intelligence",
    layout="wide"
)

px.defaults.template = "plotly_dark"

st.title("🇮🇳 Aadhaar Operational Intelligence Dashboard")
st.caption(
    "An interactive, decision-support dashboard for analysing "
    "Aadhaar enrolment and update operations using aggregated UIDAI datasets."
)

# ====================================================
# DATA LOADING & NORMALIZATION
# ====================================================
@st.cache_data
def load_data():
    files = [f for f in os.listdir('.') if f.endswith('.csv')]
    frames = []

    for f in files:
        df = pd.read_csv(f)
        df.columns = df.columns.str.strip().str.lower()

        if not {'date', 'state', 'district'}.issubset(df.columns):
            continue

        # ENROLMENT
        if {'age_0_5', 'age_5_17', 'age_18_greater'}.issubset(df.columns):
            df['child_ops'] = df['age_0_5'] + df['age_5_17']
            df['adult_ops'] = df['age_18_greater']
            df['data_type'] = 'New Enrolment'

        # BIOMETRIC UPDATE
        elif {'bio_age_5_17', 'bio_age_17_'}.issubset(df.columns):
            df['child_ops'] = df['bio_age_5_17']
            df['adult_ops'] = df['bio_age_17_']
            df['data_type'] = 'Biometric Update'

        # DEMOGRAPHIC UPDATE
        elif {'demo_age_5_17', 'demo_age_17_'}.issubset(df.columns):
            df['child_ops'] = df['demo_age_5_17']
            df['adult_ops'] = df['demo_age_17_']
            df['data_type'] = 'Demographic Update'
        else:
            continue

        df = df[['date', 'state', 'district', 'child_ops', 'adult_ops', 'data_type']]
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    df['total_ops'] = df['child_ops'] + df['adult_ops']
    return df


df = load_data()
if df.empty:
    st.error("No valid Aadhaar CSV files found.")
    st.stop()

# ====================================================
# GEOJSON LOADERS
# ====================================================
@st.cache_data
def load_geojson(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

india_states_geo = load_geojson("file2_normalized.geojson")
india_districts_geo = load_geojson("india (2).geojson")

# ====================================================
# STATE NAME NORMALIZATION (ENHANCED)
# ====================================================
def norm(s):
    """Enhanced normalization for better matching"""
    return (
        str(s).lower()
        .replace("&", "and")
        .replace(".", "")
        .replace("-", " ")
        .replace("  ", " ")
        .strip()
    )

# Detect correct state property key
STATE_KEY = list(india_states_geo['features'][0]['properties'].keys())[0]

# Build lookup for GeoJSON state names
geo_state_lookup = {
    norm(f['properties'][STATE_KEY]): f['properties'][STATE_KEY]
    for f in india_states_geo['features']
}

# Normalize data state names and map to GeoJSON
df['state_norm'] = df['state'].apply(norm)
df['state_geo'] = df['state_norm'].map(geo_state_lookup)

# ====================================================
# DISTRICT NAME NORMALIZATION (NEW - CRITICAL FIX)
# ====================================================
if india_districts_geo:
    # First, identify the district property key
    sample_district_props = india_districts_geo['features'][0]['properties']
    
    # Try to find district name key (common variations)
    DISTRICT_KEY = None
    for key in ['district', 'District', 'DISTRICT', 'dist_name', 'dtname']:
        if key in sample_district_props:
            DISTRICT_KEY = key
            break
    
    if not DISTRICT_KEY:
        DISTRICT_KEY = 'district'  # fallback
    
    # Build district-to-state mapping from GeoJSON
    district_state_map = {}
    for feature in india_districts_geo['features']:
        props = feature['properties']
        dist_name = props.get(DISTRICT_KEY, '')
        state_name = props.get('st_nm', props.get('state', ''))
        
        if dist_name and state_name:
            district_state_map[norm(dist_name)] = {
                'district_geo': dist_name,
                'state_geo': state_name,
                'state_norm': norm(state_name)
            }
    
    # Map districts in data
    df['district_norm'] = df['district'].apply(norm)
    df['district_info'] = df['district_norm'].map(district_state_map)
    
    # Extract mapped values
    df['district_geo'] = df['district_info'].apply(lambda x: x['district_geo'] if pd.notna(x) else None)
else:
    DISTRICT_KEY = 'district'
    df['district_norm'] = df['district'].apply(norm)
    df['district_geo'] = df['district']

# Debug info
matched = df['state_geo'].notna().sum()
total = len(df)
unique_states_in_data = df['state'].nunique()
unique_matched = df['state_geo'].nunique()

st.sidebar.info(f"""
**State Matching:**
- CSV states: {unique_states_in_data}
- Matched: {unique_matched}
- Rows: {matched}/{total}
""")

# Remove rows that can't be matched
df = df.dropna(subset=['state_geo'])

# ====================================================
# SIDEBAR CONTROLS
# ====================================================
st.sidebar.header("Context Filters")

state_choice = st.sidebar.selectbox(
    "Region",
    ["All India"] + sorted(df['state_geo'].unique())
)

metric_map = {
    "Total Operations": "total_ops",
    "Child Operations (0–17)": "child_ops",
    "Adult Operations (18+)": "adult_ops"
}
metric_label = st.sidebar.selectbox("Metric", list(metric_map.keys()))
metric_col = metric_map[metric_label]

filtered_df = df if state_choice == "All India" else df[df['state_geo'] == state_choice]

# ====================================================
# KPI CARDS
# ====================================================
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Operations", f"{filtered_df[metric_col].sum():,}")
k2.metric("Child Operations", f"{filtered_df['child_ops'].sum():,}")
k3.metric("Adult Operations", f"{filtered_df['adult_ops'].sum():,}")
k4.metric("Active Districts", filtered_df['district'].nunique())

st.markdown("---")

# ====================================================
# MAP VISUALIZATION (FIXED FOR DISTRICTS)
# ====================================================
st.subheader("🗺️ Regional Distribution")

if state_choice == "All India":
    # --- STATE VIEW (No changes needed - already working) ---
    state_view = (
        df.groupby('state_geo', as_index=False)
        .agg(total_ops=(metric_col, 'sum'))
    )
    
    state_data_dict = dict(zip(state_view['state_geo'], state_view['total_ops']))
    
    locations = []
    z_values = []
    text_values = []
    
    for feature in india_states_geo['features']:
        state_name = feature['properties'].get(STATE_KEY)
        if state_name in state_data_dict:
            locations.append(state_name)
            z_values.append(state_data_dict[state_name])
            text_values.append(f"{state_name}<br>{state_data_dict[state_name]:,}")
    
    fig_map = go.Figure(go.Choroplethmap(
        geojson=india_states_geo,
        locations=locations,
        z=z_values,
        featureidkey=f'properties.{STATE_KEY}',
        colorscale="YlOrRd",
        text=text_values,
        hovertemplate='<b>%{text}</b><extra></extra>',
        marker_line_width=1.5,
        marker_line_color='white',
        colorbar=dict(title=metric_label)
    ))
    
    fig_map.update_layout(
        geo_style="carto-darkmatter",
        geo_scope='asia',
        geo_center={"lat": 23.5, "lon": 78.5},
        geo_projection_scale=4,
        margin={"r":0,"t":40,"l":0,"b":0},
        height=600,
        title="State-wise Aadhaar Operations (All India)"
    )

else:
    # --- DISTRICT VIEW (FIXED VERSION) ---
    dist_view = (
        filtered_df.groupby('district', as_index=False)
        .agg(total_ops=(metric_col, 'sum'))
    )
    
    if india_districts_geo:
        # Filter GeoJSON features for selected state
        state_norm_selected = norm(state_choice)
        
        feats = [
            f for f in india_districts_geo['features']
            if norm(f['properties'].get('st_nm', f['properties'].get('state', ''))) == state_norm_selected
        ]
        
        if feats and len(feats) > 0:
            # Create district lookup
            dist_data_dict = dict(zip(
                dist_view['district'].apply(norm),
                dist_view['total_ops']
            ))
            
            # Prepare data for mapbox
            locations = []
            z_values = []
            text_values = []
            
            for feature in feats:
                dist_name = feature['properties'].get(DISTRICT_KEY, '')
                dist_norm_key = norm(dist_name)
                
                if dist_norm_key in dist_data_dict:
                    locations.append(dist_name)
                    z_values.append(dist_data_dict[dist_norm_key])
                    text_values.append(f"{dist_name}<br>{dist_data_dict[dist_norm_key]:,}")
            
            if locations:
                # Use Choroplethmapbox for better rendering
                fig_map = go.Figure(go.Choroplethmapbox(
                    geojson={"type": "FeatureCollection", "features": feats},
                    locations=locations,
                    z=z_values,
                    featureidkey=f'properties.{DISTRICT_KEY}',
                    colorscale="Inferno",
                    text=text_values,
                    hovertemplate='<b>%{text}</b><extra></extra>',
                    marker_line_width=1,
                    marker_line_color='white',
                    colorbar=dict(title=metric_label)
                ))
                
                # Calculate bounding box from features
                all_coords = []
                for feat in feats:
                    if feat['geometry']['type'] == 'Polygon':
                        all_coords.extend(feat['geometry']['coordinates'][0])
                    elif feat['geometry']['type'] == 'MultiPolygon':
                        for poly in feat['geometry']['coordinates']:
                            all_coords.extend(poly[0])
                
                if all_coords:
                    lons = [c[0] for c in all_coords]
                    lats = [c[1] for c in all_coords]
                    
                    # Calculate range
                    lon_range = max(lons) - min(lons)
                    lat_range = max(lats) - min(lats)
                    
                    # For very small states/islands, ensure minimum range
                    min_range = 0.5  # degrees
                    if lon_range < min_range:
                        lon_range = min_range
                    if lat_range < min_range:
                        lat_range = min_range
                    
                    # Add adaptive padding - more for smaller areas
                    if max(lon_range, lat_range) < 2:  # Small state/island
                        padding_factor = 0.3  # 30% padding
                    else:
                        padding_factor = 0.1  # 10% padding for larger states
                    
                    lon_padding = lon_range * padding_factor
                    lat_padding = lat_range * padding_factor
                    
                    center_lon = sum(lons) / len(lons)
                    center_lat = sum(lats) / len(lats)
                    
                    # Calculate appropriate zoom level
                    # Larger range = smaller zoom
                    max_range = max(lon_range, lat_range)
                    if max_range < 0.5:
                        zoom = 9
                    elif max_range < 1:
                        zoom = 8
                    elif max_range < 2:
                        zoom = 7
                    elif max_range < 4:
                        zoom = 6
                    else:
                        zoom = 5
                    
                    # Use center and zoom for better control
                    fig_map.update_layout(
                        mapbox_style="carto-darkmatter",
                        mapbox_center={"lat": center_lat, "lon": center_lon},
                        mapbox_zoom=zoom,
                        margin={"r":0,"t":40,"l":0,"b":0},
                        height=600,
                        title=f"District-wise Aadhaar Operations — {state_choice}"
                    )
                else:
                    # Fallback if no coordinates found
                    fig_map.update_layout(
                        geo_scope='asia',
                        geo_center={"lat": 23.5, "lon": 78.5},
                        geo_projection_scale=5,
                        margin={"r":0,"t":40,"l":0,"b":0},
                        height=600,
                        title=f"District-wise Aadhaar Operations — {state_choice}"
                    )
            else:
                st.warning(f"No matching districts found in data for {state_choice}")
                fig_map = None
        else:
            st.warning(f"No district boundaries found for {state_choice} in GeoJSON")
            
            # Fallback: Show bar chart instead
            fig_map = px.bar(
                dist_view.sort_values('total_ops', ascending=False).head(20),
                x='total_ops',
                y='district',
                orientation='h',
                title=f"Top 20 Districts in {state_choice}",
                color='total_ops',
                color_continuous_scale='Inferno'
            )
            fig_map.update_layout(height=600, showlegend=False)
    else:
        st.error("District GeoJSON file not found.")
        fig_map = None

if fig_map:
    st.plotly_chart(fig_map, width='stretch')

st.caption(
    "Map highlights regional concentration of Aadhaar operations, "
    "helping identify high-load and under-served areas."
)

st.markdown("---")

# ====================================================
# ANALYTICS - 6 COMPREHENSIVE GRAPHS
# ====================================================
st.subheader("📊 Operational Analytics")

# Row 1: Temporal Trend + Top/Bottom Districts
c1, c2 = st.columns(2)

daily = filtered_df.groupby('date')[metric_col].sum().reset_index()
fig_trend = px.area(daily, x='date', y=metric_col, 
                    title="📈 Temporal Trend of Operations")
c1.plotly_chart(fig_trend, use_container_width=True)
c1.caption("Reveals temporal surges possibly driven by campaigns or deadlines.")

district_totals = (
    filtered_df.groupby('district')[metric_col]
    .sum()
    .reset_index()
    .sort_values(metric_col, ascending=False)
)
top_bottom = pd.concat([
    district_totals.head(10).assign(category='Top 10'),
    district_totals.tail(10).assign(category='Bottom 10')
])
fig_top_bottom = px.bar(
    top_bottom,
    y='district',
    x=metric_col,
    color='category',
    orientation='h',
    title="🏆 Top 10 vs Bottom 10 Districts",
    color_discrete_map={'Top 10': '#ff6b6b', 'Bottom 10': '#4ecdc4'}
)
c2.plotly_chart(fig_top_bottom, use_container_width=True)
c2.caption("Identifies high-performing and under-served districts for resource allocation.")

st.markdown("---")

# Row 2: State-wise Operation Mix + Child vs Adult Scatter
d1, d2 = st.columns(2)

state_type_mix = (
    filtered_df.groupby(['state_geo', 'data_type'])['total_ops']
    .sum()
    .reset_index()
)
fig_state_mix = px.bar(
    state_type_mix,
    x='state_geo',
    y='total_ops',
    color='data_type',
    title="🗂️ State-wise Operation Type Distribution",
    barmode='stack'
)
fig_state_mix.update_layout(xaxis_tickangle=45)
d1.plotly_chart(fig_state_mix, width='stretch')
d1.caption("Shows whether states focus on enrolment or updates - indicates maturity.")

scatter = (
    filtered_df.groupby('district')[['child_ops', 'adult_ops', 'total_ops']]
    .sum().reset_index()
)
fig_scatter = px.scatter(
    scatter,
    x='child_ops',
    y='adult_ops',
    size='total_ops',
    log_x=True,
    log_y=True,
    opacity=0.85,
    title="👶👨 Child vs Adult Operations (District Level)",
    labels={'child_ops': 'Child Ops (0-17)', 'adult_ops': 'Adult Ops (18+)'}
)
d2.plotly_chart(fig_scatter, width='stretch')
d2.caption("Shows demographic skew - districts above diagonal have more adult operations.")

st.markdown("---")

# Row 3: Weekly Pattern + Monthly Trend
e1, e2 = st.columns(2)

filtered_df['day_of_week'] = filtered_df['date'].dt.day_name()
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dow_data = (
    filtered_df.groupby('day_of_week')[metric_col]
    .mean()
    .reindex(day_order)
    .reset_index()
)
fig_dow = px.bar(
    dow_data,
    x='day_of_week',
    y=metric_col,
    title="📅 Average Operations by Day of Week",
    color=metric_col,
    color_continuous_scale='Blues'
)
e1.plotly_chart(fig_dow, use_container_width=True)
e1.caption("Identifies weekly operational patterns - useful for staffing decisions.")

filtered_df['month'] = filtered_df['date'].dt.to_period('M').astype(str)
monthly_type = (
    filtered_df.groupby(['month', 'data_type'])['total_ops']
    .sum()
    .reset_index()
)
fig_monthly = px.line(
    monthly_type,
    x='month',
    y='total_ops',
    color='data_type',
    title="📆 Monthly Operations by Type",
    markers=True
)
fig_monthly.update_layout(xaxis_tickangle=45)
e2.plotly_chart(fig_monthly, use_container_width=True)
e2.caption("Shows seasonal patterns and growth trends across operation types.")

st.markdown("---")

# ====================================================
# GEMINI AI - ASSISTIVE INSIGHTS
# ====================================================
st.subheader("🤖 AI-Assisted Pattern Recognition")

st.warning("⚠️ **Important:** AI-generated insights are assistive only and should be validated by domain experts.")

st.info("""
📊 **Data Scope:** Analyzes aggregated summary data including top/bottom districts, state totals, and operation distributions.

⚠️ **Limitations:** Cannot answer questions requiring specific dates, pincode-level data, or districts not in rankings.
""")

use_ai = st.checkbox("Enable AI-Assisted Analysis")
if use_ai:
    # Try to get API key from secrets first, fallback to user input
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except:
        api_key = st.text_input("Gemini API Key", type="password", 
                                help="API key not configured. Enter your key or contact admin.")
    
    if api_key:
        if NEW_GENAI:
            # Use new google.genai package
            client = genai.Client(api_key=api_key)
            model_name = "gemini-1.5-flash"
        else:
            # Use old google.generativeai package
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")

        top_districts = (
            filtered_df.groupby('district')[['total_ops', 'child_ops', 'adult_ops']]
            .sum().sort_values('total_ops', ascending=False).head(15)
        )
        
        bottom_districts = (
            filtered_df.groupby('district')[['total_ops', 'child_ops', 'adult_ops']]
            .sum().sort_values('total_ops', ascending=True).head(10)
        )
        
        type_distribution = filtered_df.groupby('data_type')['total_ops'].sum()
        state_summary = (
            filtered_df.groupby('state_geo')[['total_ops', 'child_ops', 'adult_ops']]
            .sum().sort_values('total_ops', ascending=False).head(10)
        )
        
        overall_ratio = filtered_df['child_ops'].sum() / filtered_df['adult_ops'].sum()
        
        data_context = f"""
DATASET OVERVIEW:
Region: {state_choice}
Total Operations: {filtered_df['total_ops'].sum():,}
Child Operations: {filtered_df['child_ops'].sum():,}
Adult Operations: {filtered_df['adult_ops'].sum():,}
Child-to-Adult Ratio: {overall_ratio:.2f}
Active Districts: {filtered_df['district'].nunique()}

TOP 15 DISTRICTS:
{top_districts.to_string()}

BOTTOM 10 DISTRICTS:
{bottom_districts.to_string()}

OPERATION TYPES:
{type_distribution.to_string()}

TOP 10 STATES:
{state_summary.to_string()}
"""

        if st.button("🔍 Generate Pattern Analysis", type="primary"):
            with st.spinner("Analyzing patterns..."):
                prompt = f"""
Provide ASSISTIVE analysis (not directive) of this Aadhaar data:

{data_context}

Include:
1. OBSERVED PATTERNS (3-4) with confidence levels
2. QUESTIONS FOR INVESTIGATION (3-4)
3. DATA COMPARISONS (2-3)
4. CONSIDERATIONS FOR ANALYSTS (2-3)

Use assistive language, not directives. Include confidence levels with reasoning.
"""
                try:
                    if NEW_GENAI:
                        response = client.models.generate_content(
                            model=model_name,
                            contents=prompt
                        )
                        response_text = response.text
                    else:
                        response = model.generate_content(prompt)
                        response_text = response.text
                                        
                    if 'analysis_done' not in st.session_state:
                        st.session_state.analysis_done = False
                    if 'chat_history' not in st.session_state:
                        st.session_state.chat_history = []
                    
                    st.session_state.analysis_done = True
                    st.session_state.initial_analysis = response_text
                    st.session_state.data_context = data_context
                    
                    st.markdown("---")
                    st.markdown("### 📊 AI-Assisted Pattern Recognition")
                    st.markdown(response_text)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        if 'analysis_done' in st.session_state and st.session_state.analysis_done:
            st.markdown("---")
            st.markdown("### 💬 Ask Questions About the Data")
            
            user_question = st.text_input(
                "Your question:",
                placeholder="e.g., Which districts have highest child-to-adult ratio?"
            )
            
            col1, col2 = st.columns([1, 5])
            with col1:
                ask_button = st.button("Ask", type="primary")
            with col2:
                if st.button("Clear Conversation"):
                    st.session_state.chat_history = []
                    st.rerun()
            
            if ask_button and user_question:
                with st.spinner("Thinking..."):
                    qa_prompt = f"""
{st.session_state.data_context}

USER QUESTION: {user_question}

Answer based ONLY on the data above. Be specific and cite numbers.
If data doesn't contain the answer, say so clearly.
"""
                    try:
                        if NEW_GENAI:
                            qa_response = client.models.generate_content(
                                model=model_name,
                                contents=qa_prompt
                            )
                            answer = qa_response.text
                        else:
                            qa_response = model.generate_content(qa_prompt)
                            answer = qa_response.text
                        
                        st.session_state.chat_history.append({
                            'question': user_question,
                            'answer': answer
                        })
                        
                        st.markdown("**Your Question:**")
                        st.info(user_question)
                        st.markdown("**AI Response:**")
                        st.markdown(answer)
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

st.caption(
    "All insights derived from aggregated, anonymized UIDAI datasets. "
    "No personal data is used."
)