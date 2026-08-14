import os
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium

# Machine Learning & Metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Custom Module Imports
from database import init_db, insert_prediction, get_all_predictions
from export_pdf import generate_traffic_pdf
from alerts import trigger_traffic_alert

# Initialize Database
init_db()

# --- LOAD MAHARASHTRA DATASET ---
@st.cache_data
def load_maharashtra_data():
    folder_path = os.path.join("data", "real")
    
    # १. data/real/ फोल्डरमध्ये असणारी कोणतीही CSV फाईल शोधा
    if os.path.exists(folder_path):
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        if csv_files:
            # पहिली सापडलेली CSV फाईल लोड करा
            return pd.read_csv(os.path.join(folder_path, csv_files[0]))
    
    # २. जर फाईल मुख्य फोल्डरमध्ये असेल तर (Fallback Option)
    if os.path.exists("traffic_data.csv"):
        return pd.read_csv("traffic_data.csv")

    st.error("data/real/ फोल्डरमध्ये CSV फाईल सापडली नाही!")
    return pd.DataFrame()

# ग्लोबल व्हेरिएबल सेट करा
maharashtra_df = load_maharashtra_data()

# Define maharashtra_df globally
maharashtra_df = load_maharashtra_data()


# ============================================================
# REAL OPENSTREETMAP PLACES
# ============================================================

def get_real_places(place_type, district):

    import requests

    query = f"""
    [out:json][timeout:25];

    area["name"="{district}"]["boundary"="administrative"]->.searchArea;

    (
      node["amenity"="{place_type}"](area.searchArea);
      way["amenity"="{place_type}"](area.searchArea);
      relation["amenity"="{place_type}"](area.searchArea);
    );

    out center tags;
    """

    response = requests.post(
        "https://overpass-api.de/api/interpreter",
        data=query,
        headers={
            "User-Agent": "MaharashtraTrafficProject/1.0"
        }
    )

    if response.status_code != 200:
        return []

    return response.json().get("elements", [])



# ============================================================
# PAGE CONFIGURATION & SESSION STATE
# ============================================================

st.set_page_config(
    page_title="Smart Traffic Prediction System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

# ============================================================
# LOGIN PAGE
# ============================================================

if not st.session_state.logged_in:
    if not st.session_state.logged_in:
       components.html("""
        <div id="canvas-container" style="position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; z-index: -1; pointer-events: none; background: #0e1117;">
            <canvas id="bg-canvas"></canvas>
        </div>
        <script>
            const canvas = document.getElementById('bg-canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;

            let particles = [];
            for(let i = 0; i < 60; i++) {
                particles.push({
                    x: Math.random() * canvas.width,
                    y: Math.random() * canvas.height,
                    radius: Math.random() * 2 + 1,
                    vx: (Math.random() - 0.5) * 1.5,
                    vy: (Math.random() - 0.5) * 1.5
                });
            }

            function animate() {
                requestAnimationFrame(animate);
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.fillStyle = 'rgba(47, 128, 237, 0.7)';
                
                particles.forEach(p => {
                    p.x += p.vx;
                    p.y += p.vy;
                    if(p.x < 0 || p.x > canvas.width) p.vx *= -1;
                    if(p.y < 0 || p.y > canvas.height) p.vy *= -1;

                    ctx.beginPath();
                    ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
                    ctx.fill();
                });
            }
            animate();
        </script>
    """, height=0)

    # २. Glassmorphism आणि Neon UI सोबत लॉगिन कार्ड
    st.markdown("""
    <div style="
        max-width:500px;
        margin:60px auto 20px auto;
        padding:40px;
        background: rgba(27, 31, 42, 0.75);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius:20px;
        border:1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        text-align:center;
    ">
        <h1 style="color: #fff; text-shadow: 0 0 10px #2f80ed;">🚦 Smart Traffic</h1>
        <p style="color: #aab2c0;">AI & Machine Learning Traffic Prediction System</p>
    </div>
    """, unsafe_allow_html=True)

    username = st.text_input("👤 Username", placeholder="Enter username")
    password = st.text_input("🔒 Password", type="password", placeholder="Enter password")

    if st.button("🚀 Login", use_container_width=True):
        if username == "admin" and password == "admin123":
            st.session_state.logged_in = True
            st.session_state.username = username
            st.rerun()
        else:
            st.error("❌ Invalid username or password")

    st.info("Demo Login: **admin** / **admin123**")
    st.stop()

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
.main { background-color: #0e1117; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }
h1 { font-size: 42px !important; font-weight: 700 !important; }
h2 { font-size: 28px !important; }
h3 { font-size: 22px !important; }
.metric-card {
    background: #1b1f2a; padding: 20px; border-radius: 12px;
    border: 1px solid #303642; text-align: center;
}
.metric-title { color: #aab2c0; font-size: 15px; }
.metric-value { font-size: 30px; font-weight: bold; }
.feature-card {
    background: #1b1f2a; padding: 20px; border-radius: 12px;
    border-left: 5px solid #2f80ed; margin-bottom: 10px;
}
.footer { text-align: center; color: #8b93a1; padding: 30px; margin-top: 40px; }
/* Glassmorphism Effect */
.glass-card {
    background: rgba(27, 31, 42, 0.7);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 25px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    transition: all 0.3s ease-in-out;
}

/* Hover Animation Effect */
.glass-card:hover {
    transform: translateY(-8px) scale(1.02);
    border: 1px solid #2f80ed;
    box-shadow: 0 0 20px rgba(47, 128, 237, 0.6);
}

.neon-text {
    color: #fff;
    text-shadow: 0 0 10px #2f80ed, 0 0 20px #2f80ed;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATASET CREATION / LOADING
# ============================================================

@st.cache_data
def create_dataset():
    np.random.seed(42)
    n = 3000
    hour = np.random.randint(0, 24, n)
    day_of_week = np.random.randint(0, 7, n)
    weather = np.random.randint(0, 4, n)
    accidents = np.random.randint(0, 4, n)
    road_capacity = np.random.randint(500, 2000, n)

    base_traffic = np.where(
        ((hour >= 7) & (hour <= 10)) | ((hour >= 17) & (hour <= 21)),
        np.random.randint(900, 1400, n),
        np.random.randint(300, 600, n)
    )

    traffic = base_traffic + accidents * 100 + weather * 40 + np.random.normal(0, 70, n)
    traffic = np.clip(traffic, 100, 1600)

    return pd.DataFrame({
        "Hour": hour,
        "Day_of_Week": day_of_week,
        "Weather": weather,
        "Accidents": accidents,
        "Road_Capacity": road_capacity,
        "Vehicles": traffic.astype(int)
    })

if os.path.exists("data/traffic_data.csv"):
    df = pd.read_csv("data/traffic_data.csv")
else:
    df = create_dataset()

# ============================================================
# MACHINE LEARNING MODEL
# ============================================================

features = ["Hour", "Day_of_Week", "Weather", "Accidents", "Road_Capacity"]
X = df[features]
y = df["Vehicles"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

model = RandomForestRegressor(n_estimators=150, random_state=42, max_depth=15)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)
accuracy = max(0, min(100, r2 * 100))

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("🚦 Navigation")
st.sidebar.success(f"👤 Logged in as: {st.session_state.username}")

if st.sidebar.button("🚪 Logout", use_container_width=True):
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.rerun()

page = st.sidebar.radio(
    "Select Module",
    [
        "🏠 Dashboard",
        "🔮 Traffic Prediction",
        "📜 Prediction History",
        "🗺️ Maharashtra Traffic Map",
        "🌦️ Weather & Holidays",
        "📊 Analytics",
        "🤖 ML Model",
        "📁 Dataset",
        "ℹ️ About Project"
    ]
)

# ============================================================
# PROJECT BRANDING CARD
# ============================================================

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    <div style="
        padding: 18px;
        border-radius: 16px;
        background: linear-gradient(145deg, #111827, #1e293b);
        border: 1px solid rgba(255,255,255,0.12);
        box-shadow: 0 8px 25px rgba(0,0,0,0.25);
        text-align: center;
        margin-bottom: 15px;
    ">

        <div style="
            font-size: 34px;
            margin-bottom: 8px;
        ">
            🚦
        </div>

        <div style="
            font-size: 17px;
            font-weight: 700;
            color: white;
            margin-bottom: 6px;
        ">
            Maharashtra Traffic
        </div>

        <div style="
            font-size: 13px;
            color: #94a3b8;
            margin-bottom: 14px;
        ">
            Intelligence System
        </div>

        <div style="
            height: 1px;
            background: rgba(255,255,255,0.12);
            margin: 10px 0 14px 0;
        ">
        </div>

        <div style="
            font-size: 12px;
            color: #cbd5e1;
            line-height: 1.6;
        ">
            🤖 Machine Learning<br>
            🗺️ Maharashtra Monitoring<br>
            🌦️ Weather Intelligence<br>
            🎉 Holiday Analysis
        </div>

        <div style="
            margin-top: 15px;
            padding: 7px;
            border-radius: 8px;
            background: rgba(34,197,94,0.12);
            color: #86efac;
            font-size: 11px;
            font-weight: 600;
        ">
            ● SMART MOBILITY PROJECT
        </div>

    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.caption(
    "Smart Mobility • Data • AI"
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":
    st.title("🚦 Smart Traffic Prediction System")
    st.write("AI & Machine Learning based intelligent traffic monitoring and prediction platform")
    st.markdown("---")

    st.subheader("🚦 Current Traffic Intelligence")
    avg_traffic = df["Vehicles"].mean()

    if avg_traffic < 500:
        traffic_status = "🟢 LOW"
    elif avg_traffic < 900:
        traffic_status = "🟡 MEDIUM"
    else:
        traffic_status = "🔴 HIGH"

    peak_hour = df.groupby("Hour")["Vehicles"].mean().idxmax()
    peak_traffic = df.groupby("Hour")["Vehicles"].mean().max()
    capacity_utilization = (df["Vehicles"].mean() / df["Road_Capacity"].mean()) * 100

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("🚦 Traffic Status", traffic_status)
    with c2:
        st.metric("⏰ Peak Hour", f"{peak_hour}:00")
    with c3:
        st.metric("🚗 Peak Traffic", f"{peak_traffic:.0f}")
    with c4:
        st.metric("🛣️ Capacity Usage", f"{capacity_utilization:.1f}%")

    st.markdown("---")
    st.subheader("📌 System Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🚗 Total Records", f"{len(df):,}")
    with col2:
        st.metric("Average Traffic", f"{df['Vehicles'].mean():.0f}")
    with col3:
        st.metric("🎯 Model Accuracy", f"{accuracy:.2f}%")
    with col4:
        st.metric("📊 R² Score", f"{r2:.3f}")

    st.markdown("---")
    st.subheader("📊 Traffic Distribution")

    col1, col2 = st.columns(2)
    df["Traffic_Level"] = pd.cut(
        df["Vehicles"],
        bins=[0, 500, 900, np.inf],
        labels=["Low", "Medium", "High"]
    )
    level_counts = df["Traffic_Level"].value_counts()

    with col1:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(level_counts.index.astype(str), level_counts.values)
        ax.set_title("Traffic Level Distribution")
        ax.set_xlabel("Traffic Level")
        ax.set_ylabel("Number of Records")
        st.pyplot(fig)

    hourly = df.groupby("Hour")["Vehicles"].mean()
    with col2:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(hourly.index, hourly.values, marker="o")
        ax.set_title("Average Traffic by Hour")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Average Vehicles")
        ax.grid(True)
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("⚙️ System Features")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="feature-card">
        <h3>🤖 Machine Learning</h3>
        <p>Random Forest based traffic prediction.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="feature-card">
        <h3>📊 Data Analytics</h3>
        <p>Traffic trends, distributions and insights.</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="feature-card">
        <h3>🌦️ Smart Inputs</h3>
        <p>Weather, accidents, road capacity and time.</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# TRAFFIC PREDICTION
# ============================================================
elif page == "🔮 Traffic Prediction":
    st.title("🔮 Traffic Prediction")
    st.write("Enter road and environmental information to predict the expected traffic volume.")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        hour = st.slider("🕐 Hour of Day", 0, 23, 8)
        day = st.selectbox("📅 Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        day_number = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(day)

    with col2:
        weather_name = st.selectbox("🌦️ Weather Condition", ["Clear", "Cloudy", "Rainy", "Stormy"])
        weather_number = ["Clear", "Cloudy", "Rainy", "Stormy"].index(weather_name)
        accidents = st.slider("🚨 Number of Accidents", 0, 5, 0)
        road_capacity = st.slider("🛣️ Road Capacity", 500, 2000, 1200)

    st.markdown("---")

    if st.button("🚦 Predict Traffic", use_container_width=True):
        input_data = pd.DataFrame({
            "Hour": [hour],
            "Day_of_Week": [day_number],
            "Weather": [weather_number],
            "Accidents": [accidents],
            "Road_Capacity": [road_capacity]
        })

        prediction = model.predict(input_data)[0]

        if prediction < 500:
            level = "LOW TRAFFIC"
            message = "Traffic is expected to be normal."
        elif prediction < 900:
            level = "MEDIUM TRAFFIC"
            message = "Moderate traffic expected."
        else:
            level = "HIGH TRAFFIC"
            message = "Heavy traffic expected. Consider alternate routes."

        # Session State मध्ये डेटा सेव्ह करणे
        st.session_state.prediction_history.append({
            "Hour": hour,
            "Day": day,
            "Weather": weather_name,
            "Accidents": accidents,
            "Road Capacity": road_capacity,
            "Predicted Vehicles": round(prediction),
            "Traffic Level": level
        })

        # ============================================================
        # 💾 1. DATABASE INSERTION (SQLite मध्ये डेटा कायमचा सेव्ह करणे)
        # ============================================================
        insert_prediction(
            username=st.session_state.username,
            day=day,
            hour=hour,
            weather=weather_name,
            predicted_vehicles=round(prediction),
            traffic_level=level
        )

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Predicted Vehicles", f"{prediction:.0f}")
        with c2:
            st.metric("Traffic Level", level)
        with c3:
            st.metric("Road Capacity", f"{road_capacity}")

        if prediction < 500:
            st.success(f"🟢 {message}")
        elif prediction < 900:
            st.warning(f"🟡 {message}")
        else:
            st.error(f"🔴 {message}")

        # ============================================================
        # 🚨 2. EMERGENCY ALERT TRIGGER (HIGH TRAFFIC साठी)
        # ============================================================
        is_triggered, alert_msg = trigger_traffic_alert(
            location="Main Expressway / Central Zone",
            predicted_count=round(prediction),
            traffic_level=level
        )
        if is_triggered:
            st.toast(alert_msg, icon="🚨")

        # ============================================================
        # 📄 3. PDF REPORT GENERATION#
        # ============================================================
        pdf_bytes = generate_traffic_pdf(
            username=st.session_state.username,
            day=day,
            hour=hour,
            weather=weather_name,
            predicted_vehicles=round(prediction),
            traffic_level=level,
            recommendation=message
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            label="📄 Download Official PDF Report",
            data=pdf_bytes,
            file_name=f"Traffic_Report_{day}_{hour}h.pdf",
            mime="application/pdf",
            use_container_width=True
        )
# ============================================================
# PREDICTION HISTORY
# ============================================================

elif page == "📜 Prediction History":
    st.title("📜 Prediction History")
    st.write("View and download all traffic predictions made during this session.")
    st.markdown("---")

    history = st.session_state.prediction_history

    if len(history) == 0:
        st.info("No predictions available yet. Go to 🔮 Traffic Prediction and make a prediction.")
    else:
        history_df = pd.DataFrame(history)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Predictions", len(history_df))
        with col2:
            st.metric("Average Predicted Vehicles", f"{history_df['Predicted Vehicles'].mean():.0f}")

        st.markdown("---")
        st.dataframe(history_df, use_container_width=True)
        st.markdown("---")

        st.subheader("📊 Prediction Summary")
        summary = history_df["Traffic Level"].value_counts()
        st.bar_chart(summary)

        csv = history_df.to_csv(index=False)
        st.download_button(
            "⬇️ Download Prediction History",
            data=csv,
            file_name="prediction_history.csv",
            mime="text/csv",
            use_container_width=True
        )

        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.prediction_history = []
            st.success("Prediction history cleared.")
            st.rerun()

# ============================================================
# MAHARASHTRA TRAFFIC MAP + REAL PLACES
# ============================================================

elif page == "🗺️ Maharashtra Traffic Map":

    st.title("🗺️ Maharashtra Traffic Intelligence Map")

    st.write(
        "Maharashtra traffic monitoring with real "
        "OpenStreetMap schools, colleges and hospitals."
    )

    st.markdown("---")

    # ========================================================
    # CITY SELECTION
    # ========================================================

    city_locations = {
        "Pune": [18.5204, 73.8567],
        "Mumbai": [19.0760, 72.8777],
        "Nagpur": [21.1458, 79.0882],
        "Nashik": [19.9975, 73.7898],
        "Thane": [19.2183, 72.9781],
        "Kolhapur": [16.7050, 74.2433],
        "Solapur": [17.6599, 75.9064],
        "Satara": [17.6805, 74.0183],
        "Aurangabad": [19.8762, 75.3433],
        "Ahmednagar": [19.0948, 74.7480]
    }

    selected_city = st.selectbox(
        "🏙️ Select Maharashtra City",
        list(city_locations.keys())
    )

    # ========================================================
    # PLACE TYPE
    # ========================================================

    place_option = st.selectbox(
        "📍 Show Real Places",
        [
            "None",
            "🏫 Schools",
            "🎓 Colleges",
            "🏥 Hospitals",
            "🏫🎓🏥 All"
        ]
    )

    # ========================================================
    # MAP
    # ========================================================

    maharashtra_map = folium.Map(
        location=city_locations[selected_city],
        zoom_start=11,
        tiles="OpenStreetMap"
    )

    # ========================================================
    # EXISTING TRAFFIC DATA
    # ========================================================

    if "maharashtra_df" in globals():

        traffic_df = maharashtra_df.copy()

        for _, row in traffic_df.iterrows():

            if (
                "Latitude" not in traffic_df.columns
                or "Longitude" not in traffic_df.columns
            ):
                continue

            if pd.isna(row["Latitude"]) or pd.isna(row["Longitude"]):
                continue

            if "Vehicles" in traffic_df.columns:

                vehicles = row["Vehicles"]

                if vehicles >= 1100:
                    icon_color = "red"
                    level = "HIGH"

                elif vehicles >= 800:
                    icon_color = "orange"
                    level = "MEDIUM"

                else:
                    icon_color = "green"
                    level = "LOW"

            else:

                icon_color = "blue"
                level = "TRAFFIC"

            popup_text = f"""
            <b>{row.get('District', 'Unknown')}</b><br>
            🚗 Vehicles: {row.get('Vehicles', 'N/A')}<br>
            🛣️ Road Capacity: {row.get('Road_Capacity', 'N/A')}<br>
            🚨 Accidents: {row.get('Accidents', 'N/A')}<br>
            📊 Congestion: {row.get('Congestion_Percent', 'N/A')}%<br>
            🚦 Traffic Level: {row.get('Traffic_Level', level)}
            """

            folium.Marker(
                location=[
                    float(row["Latitude"]),
                    float(row["Longitude"])
                ],
                popup=folium.Popup(
                    popup_text,
                    max_width=300
                ),
                tooltip=(
                    f"{row.get('District', 'Location')} "
                    f"- {level}"
                ),
                icon=folium.Icon(
                    color=icon_color,
                    icon="car",
                    prefix="fa"
                )
            ).add_to(maharashtra_map)

    # ========================================================
    # REAL OPENSTREETMAP PLACES
    # ========================================================

    place_types = []

    if place_option == "🏫 Schools":
        place_types = ["school"]

    elif place_option == "🎓 Colleges":
        place_types = ["college"]

    elif place_option == "🏥 Hospitals":
        place_types = ["hospital"]

    elif place_option == "🏫🎓🏥 All":
        place_types = [
            "school",
            "college",
            "hospital"
        ]

    if place_types:

        if st.button(
            "🔍 Load Real Places",
            width="stretch"
        ):

            all_places = []

            with st.spinner(
                f"Loading real places in {selected_city}..."
            ):

                for place_type in place_types:

                    try:

                        places = get_real_places(
                            place_type,
                            selected_city
                        )

                        all_places.extend(
                            places
                        )

                    except Exception as e:

                        st.error(
                            f"Unable to load {place_type}: {e}"
                        )

            if len(all_places) == 0:

                st.warning(
                    "⚠️ No mapped places found for "
                    f"{selected_city}."
                )

            else:

                st.success(
                    f"✅ {len(all_places)} real mapped "
                    f"places found."
                )

                # --------------------------------------------
                # ADD PLACES TO MAP
                # --------------------------------------------

                for place in all_places:

                    tags = place.get(
                        "tags",
                        {}
                    )

                    if place["type"] == "node":

                        latitude = place.get("lat")
                        longitude = place.get("lon")

                    else:

                        center = place.get("center")

                        if not center:
                            continue

                        latitude = center.get("lat")
                        longitude = center.get("lon")

                    if latitude is None or longitude is None:
                        continue

                    name = tags.get(
                        "name",
                        "Unnamed Place"
                    )

                    amenity = tags.get(
                        "amenity",
                        ""
                    )

                    street = tags.get(
                        "addr:street",
                        "Address not available"
                    )

                    if amenity == "school":

                        marker_color = "blue"
                        marker_icon = "graduation-cap"
                        place_label = "School"

                    elif amenity == "college":

                        marker_color = "purple"
                        marker_icon = "graduation-cap"
                        place_label = "College"

                    else:

                        marker_color = "red"
                        marker_icon = "plus"
                        place_label = "Hospital"

                    popup_text = f"""
                    <b>{name}</b><br>
                    📍 {place_label}<br>
                    🏙️ {selected_city}<br>
                    🏠 {street}<br>
                    🌐 Source: OpenStreetMap
                    """

                    folium.Marker(
                        location=[
                            float(latitude),
                            float(longitude)
                        ],
                        popup=folium.Popup(
                            popup_text,
                            max_width=300
                        ),
                        tooltip=(
                            f"{name} - {place_label}"
                        ),
                        icon=folium.Icon(
                            color=marker_color,
                            icon=marker_icon,
                            prefix="fa"
                        )
                    ).add_to(
                        maharashtra_map
                    )

    # ========================================================
    # DISPLAY MAP
    # ========================================================

    st.subheader(
        f"🗺️ {selected_city} Traffic & Places"
    )

    st_folium(
        maharashtra_map,
        width="stretch",
        height=650
    )

    # ========================================================
    # LEGEND
    # ========================================================

    st.markdown("---")

    st.subheader("🗺️ Map Legend")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.write("🚗 **Traffic Location**")

    with col2:
        st.write("🏫 **School**")

    with col3:
        st.write("🎓 **College**")

    with col4:
        st.write("🏥 **Hospital**")

    # ========================================================
    # TRAFFIC SUMMARY
    # ========================================================

    if "maharashtra_df" in globals():

        st.markdown("---")

        st.subheader("📊 Maharashtra Traffic Summary")
     

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "🏙️ Districts",
           len(maharashtra_df)
        )

    with c2:
        if "Vehicles" in maharashtra_df.columns:
            st.metric(
                "🚗 Total Vehicles",
                f"{maharashtra_df['Vehicles'].sum():,.0f}"
            )

    with c3:
        if "Accidents" in maharashtra_df.columns:
            st.metric(
                "🚨 Total Accidents",
                f"{maharashtra_df['Accidents'].sum():,.0f}"
            )

    with c4:
        if "Congestion_Percent" in maharashtra_df.columns:
            st.metric(
                "📈 Avg Congestion",
                f"{maharashtra_df['Congestion_Percent'].mean():.1f}%"
            )

    st.markdown("---")

    st.subheader("📋 District-wise Traffic")

    st.dataframe(
        maharashtra_df,
        use_container_width=True,
        hide_index=True
    )

            

# ============================================================
# REAL WEATHER & MAHARASHTRA HOLIDAYS
# ============================================================

elif page == "🌦️ Weather & Holidays" or "Weather" in page:

    st.title("🌦️ Real Weather & Holiday Intelligence")

    st.write(
        "Real weather forecast and Maharashtra public holiday information."
    )

    st.markdown("---")

    # ========================================================
    # MAHARASHTRA LOCATIONS
    # ========================================================

    maharashtra_locations = {
        "Mumbai": (19.0760, 72.8777),
        "Pune": (18.5204, 73.8567),
        "Nagpur": (21.1458, 79.0882),
        "Nashik": (19.9975, 73.7898),
        "Chhatrapati Sambhajinagar": (19.8762, 75.3433),
        "Thane": (19.2183, 72.9781),
        "Navi Mumbai": (19.0330, 73.0297),
        "Kolhapur": (16.7050, 74.2433),
        "Solapur": (17.6599, 75.9064),
        "Satara": (17.6805, 74.0183)
    }

    # ========================================================
    # OFFICIAL MAHARASHTRA PUBLIC HOLIDAYS 2026
    # ========================================================

    maharashtra_holidays = {
        "2026-01-26": "Republic Day",
        "2026-02-15": "Mahashivratri",
        "2026-02-19": "Chhatrapati Shivaji Maharaj Jayanti",
        "2026-03-03": "Holi (Second Day)",
        "2026-03-19": "Gudhi Padwa",
        "2026-03-21": "Ramzan-Id (Id-Ul-Fitra)",
        "2026-03-26": "Ram Navami",
        "2026-03-31": "Mahavir Janmakalyanak",
        "2026-04-03": "Good Friday",
        "2026-04-14": "Dr. Babasaheb Ambedkar Jayanti",
        "2026-05-01": "Maharashtra Din / Buddha Pournima",
        "2026-05-28": "Bakri ID (Id-Uz-Zuha)",
        "2026-06-26": "Moharum",
        "2026-08-15": "Independence Day / Parsi New Year",
        "2026-08-26": "Id-E-Milad",
        "2026-09-14": "Ganesh Chaturthi",
        "2026-10-02": "Mahatma Gandhi Jayanti",
        "2026-10-20": "Dasara",
        "2026-11-08": "Diwali Amavasya (Lakshmi Pujan)",
        "2026-11-10": "Diwali (Balipratipada)",
        "2026-11-24": "Guru Nanak Jayanti",
        "2026-12-25": "Christmas"
    }

    # ========================================================
    # USER INPUT
    # ========================================================

    st.subheader("📍 Select Maharashtra Location")

    city = st.selectbox(
        "City / Location",
        list(maharashtra_locations.keys())
    )

    selected_date = st.date_input(
        "📅 Select Date",
        value=pd.Timestamp.today().date()
    )

    selected_hour = st.slider(
        "🕐 Select Hour",
        min_value=0,
        max_value=23,
        value=pd.Timestamp.now().hour
    )

    latitude, longitude = maharashtra_locations[city]
    day_name = selected_date.strftime("%A")
    date_string = selected_date.strftime("%Y-%m-%d")

    # ========================================================
    # HOLIDAY CHECK
    # ========================================================

    holiday_name = maharashtra_holidays.get(date_string, None)

    if holiday_name:
        day_type = "Public Holiday"
        st.success(f"🎉 Public Holiday: {holiday_name}")
    elif day_name == "Sunday":
        day_type = "Sunday"
        st.info("📅 Sunday — Weekly Holiday")
    elif day_name == "Saturday":
        day_type = "Saturday"
        st.info("📅 Saturday — Weekend")
    else:
        day_type = "Working Day"
        st.info("💼 Normal Working Day")

    # ========================================================
    # REAL WEATHER API
    # ========================================================

    st.markdown("---")
    st.subheader("🌦️ Real Weather Data")

    try:
        import requests

        weather_url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={latitude}"
            f"&longitude={longitude}"
            "&hourly=temperature_2m,"
            "relative_humidity_2m,"
            "precipitation,"
            "rain,"
            "weather_code,"
            "wind_speed_10m"
            f"&start_date={date_string}"
            f"&end_date={date_string}"
            "&timezone=Asia%2FKolkata"
        )

        response = requests.get(weather_url, timeout=10)
        response.raise_for_status()

        weather_data = response.json()
        hourly = weather_data["hourly"]

        temperature = hourly["temperature_2m"][selected_hour]
        humidity = hourly["relative_humidity_2m"][selected_hour]
        precipitation = hourly["precipitation"][selected_hour]
        rain = hourly["rain"][selected_hour]
        wind_speed = hourly["wind_speed_10m"][selected_hour]
        weather_code = hourly["weather_code"][selected_hour]

        weather_codes = {
            0: "☀️ Clear Sky",
            1: "🌤️ Mainly Clear",
            2: "⛅ Partly Cloudy",
            3: "☁️ Overcast",
            45: "🌫️ Fog",
            48: "🌫️ Depositing Rime Fog",
            51: "🌦️ Light Drizzle",
            53: "🌦️ Moderate Drizzle",
            55: "🌧️ Dense Drizzle",
            61: "🌧️ Slight Rain",
            63: "🌧️ Moderate Rain",
            65: "🌧️ Heavy Rain",
            71: "🌨️ Slight Snow",
            73: "🌨️ Moderate Snow",
            75: "🌨️ Heavy Snow",
            80: "🌦️ Rain Showers",
            81: "🌧️ Rain Showers",
            82: "⛈️ Heavy Rain Showers",
            95: "⛈️ Thunderstorm",
            96: "⛈️ Thunderstorm + Hail",
            99: "⛈️ Severe Thunderstorm"
        }

        weather_condition = weather_codes.get(weather_code, "🌦️ Unknown")

        # DISPLAY WEATHER
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric("🌡️ Temperature", f"{temperature:.1f} °C")
        with c2:
            st.metric("💧 Humidity", f"{humidity:.0f}%")
        with c3:
            st.metric("🌧️ Rain", f"{rain:.1f} mm")
        with c4:
            st.metric("💨 Wind", f"{wind_speed:.1f} km/h")

        st.info(f"🌦️ Weather: {weather_condition}")

        # TRAFFIC IMPACT INDICATOR
        traffic_impact = 0
        if rain > 0:
            traffic_impact += 15
        if rain >= 5:
            traffic_impact += 10
        if weather_code in [45, 48]:
            traffic_impact += 10
        if weather_code >= 95:
            traffic_impact += 20
        if holiday_name:
            traffic_impact += 10
        if day_name == "Sunday":
            traffic_impact -= 5

        traffic_impact = max(0, traffic_impact)

        st.markdown("---")
        st.subheader("🚗 Traffic Impact Indicator")

        if traffic_impact >= 30:
            st.error(f"🔴 HIGH IMPACT — {traffic_impact}%")
        elif traffic_impact >= 15:
            st.warning(f"🟡 MODERATE IMPACT — {traffic_impact}%")
        else:
            st.success(f"🟢 LOW IMPACT — {traffic_impact}%")

    except Exception as e:
        st.error("❌ Unable to fetch live weather data.")
        st.caption(f"Error: {e}")

    # ========================================================
    # INFORMATION TABLE
    # ========================================================

    st.markdown("---")
    st.subheader("📋 Location & Holiday Information")

    info_df = pd.DataFrame({
        "Parameter": [
            "Location",
            "Latitude",
            "Longitude",
            "Date",
            "Day",
            "Day Type",
            "Holiday"
        ],
        "Value": [
            city,
            latitude,
            longitude,
            selected_date.strftime("%d-%m-%Y"),
            day_name,
            day_type,
            holiday_name if holiday_name else "No Public Holiday"
        ]
    })

    st.dataframe(
        info_df,
        use_container_width=True,
        hide_index=True
    )

# ============================================================
# ANALYTICS
# ============================================================


# ============================================================
# MAHARASHTRA TRAFFIC ANALYTICS
# ============================================================

elif page == "📊 Analytics":

    st.title("📊 Maharashtra Traffic Analytics")
    st.write(
        "Analytics based only on the available Maharashtra traffic dataset."
    )

    st.markdown("---")

    # Load actual Maharashtra dataset
    maharashtra_file = "data/maharashtra_traffic.csv"
    # जुना लोड करण्याचा कोड काढून फक्त ही लाईन ठेवा:
    maharashtra_df = pd.read_csv("traffic_data.csv")
    # Convert actual numeric columns
    numeric_columns = [
        "Vehicles",
        "Road_Capacity",
        "Accidents",
        "Congestion_Percent"
    ]

    for col in numeric_columns:
        maharashtra_df[col] = pd.to_numeric(
            maharashtra_df[col],
            errors="coerce"
        )

    maharashtra_df = maharashtra_df.dropna(
        subset=numeric_columns
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    st.subheader("📌 Maharashtra Traffic Summary")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "🏙️ Districts",
            len(maharashtra_df)
        )

    with c2:
        st.metric(
            "🚗 Total Vehicles",
            f"{maharashtra_df['Vehicles'].sum():,.0f}"
        )

    with c3:
        st.metric(
            "🚨 Total Accidents",
            f"{maharashtra_df['Accidents'].sum():,.0f}"
        )

    with c4:
        st.metric(
            "📈 Avg Congestion",
            f"{maharashtra_df['Congestion_Percent'].mean():.1f}%"
        )

    st.markdown("---")

    # ========================================================
    # TOP TRAFFIC DISTRICTS
    # ========================================================

    st.subheader("🚗 Highest Traffic Districts")

    top_traffic = (
        maharashtra_df
        .sort_values(
            "Vehicles",
            ascending=False
        )
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        top_traffic["District"],
        top_traffic["Vehicles"]
    )

    ax.set_xlabel("District")
    ax.set_ylabel("Vehicles")
    ax.set_title("Top 10 Districts by Vehicle Traffic")

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    st.pyplot(fig)

    # ========================================================
    # ROAD CAPACITY VS VEHICLES
    # ========================================================

    st.subheader("🛣️ Vehicles vs Road Capacity")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.scatter(
        maharashtra_df["Road_Capacity"],
        maharashtra_df["Vehicles"],
        alpha=0.7
    )

    ax.set_xlabel("Road Capacity")
    ax.set_ylabel("Vehicles")
    ax.set_title(
        "Vehicle Traffic Compared with Road Capacity"
    )

    ax.grid(True)

    st.pyplot(fig)

    # ========================================================
    # ACCIDENT ANALYSIS
    # ========================================================

    st.subheader("🚨 Accidents by District")

    accident_data = (
        maharashtra_df
        .sort_values(
            "Accidents",
            ascending=False
        )
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        accident_data["District"],
        accident_data["Accidents"]
    )

    ax.set_xlabel("District")
    ax.set_ylabel("Accidents")
    ax.set_title("Top Districts by Accident Count")

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    st.pyplot(fig)

    # ========================================================
    # CONGESTION ANALYSIS
    # ========================================================

    st.subheader("📈 Congestion by District")

    congestion_data = (
        maharashtra_df
        .sort_values(
            "Congestion_Percent",
            ascending=False
        )
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        congestion_data["District"],
        congestion_data["Congestion_Percent"]
    )

    ax.set_xlabel("District")
    ax.set_ylabel("Congestion (%)")
    ax.set_title("Top Districts by Congestion")

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    st.pyplot(fig)

    # ========================================================
    # TRAFFIC LEVEL DISTRIBUTION
    # ========================================================

    st.subheader("🚦 Traffic Level Distribution")

    level_counts = (
        maharashtra_df["Traffic_Level"]
        .value_counts()
    )

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(
        level_counts.index.astype(str),
        level_counts.values
    )

    ax.set_xlabel("Traffic Level")
    ax.set_ylabel("Number of Districts")
    ax.set_title("Maharashtra Traffic Level Distribution")

    st.pyplot(fig)

    # ========================================================
    # DISTRICT DATA
    # ========================================================

    st.markdown("---")

    st.subheader("📋 District-wise Analytics")

    st.dataframe(
        maharashtra_df.sort_values(
            "Vehicles",
            ascending=False
        ),
        width="stretch",
        hide_index=True
    )



# ============================================================
# ML MODEL
# ============================================================
elif page == "🤖 ML Model":
    st.title("🤖 Machine Learning Models & Comparison")
    st.write("Comparing multiple ML algorithms to select the best model for traffic prediction.")
    st.markdown("---")

    # १. मॉडेल्सची डिक्शनरी
    models_dict = {
        "Random Forest": RandomForestRegressor(n_estimators=150, random_state=42, max_depth=15),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42, learning_rate=0.1)
    }

    results = []

    # २. सर्व मॉडेल्स ट्रेन करून अचूकता तपासणे
    for name, md in models_dict.items():
        md.fit(X_train, y_train)
        pred = md.predict(X_test)
        
        r2_val = r2_score(y_test, pred)
        mae_val = mean_absolute_error(y_test, pred)
        rmse_val = np.sqrt(mean_squared_error(y_test, pred))
        acc_val = max(0, min(100, r2_val * 100))

        results.append({
            "Model Name": name,
            "Accuracy (%)": f"{acc_val:.2f}%",
            "R² Score": f"{r2_val:.3f}",
            "MAE": f"{mae_val:.2f}",
            "RMSE": f"{rmse_val:.2f}"
        })

    # ३. तुलना करणारा टेबल दाखवणे
    st.subheader("⚖️ Model Performance Comparison")
    comp_df = pd.DataFrame(results)
    st.dataframe(comp_df, use_container_width=True)

    st.markdown("---")

    # ४. Feature Importance graph
    st.subheader("📊 Feature Importance (Random Forest)")
    importance = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(importance["Feature"], importance["Importance"])
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    st.pyplot(fig)

# ============================================================
# DATASET
# ============================================================

elif page == "📁 Dataset":
    st.title("📁 Traffic Dataset")
    st.write("Dataset used for training and evaluating the machine learning model.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", f"{len(df):,}")
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        st.metric("Missing Values", int(df.isnull().sum().sum()))

    st.markdown("---")
    st.dataframe(df.head(100), use_container_width=True)

    csv = df.to_csv(index=False)
    st.download_button(
        label="⬇️ Download Dataset CSV",
        data=csv,
        file_name="traffic_dataset.csv",
        mime="text/csv",
        use_container_width=True
    )

# ============================================================
# ABOUT PROJECT
# ============================================================

elif page == "ℹ️ About Project":

    st.title("ℹ️ About Traffic Intelligence System")

    st.write(
        "A smart machine-learning based traffic monitoring and "
        "prediction platform designed for Maharashtra."
    )

    st.markdown("---")

    # ========================================================
    # PROJECT OVERVIEW
    # ========================================================

    st.subheader("🚦 Project Overview")

    st.write(
        """
        This project is a Traffic Intelligence System that combines
        Machine Learning, traffic data, Maharashtra location data,
        weather information and holiday information to support
        smarter traffic monitoring and prediction.

        The system is designed to help users understand traffic
        conditions, identify congestion-prone locations and make
        better travel decisions.
        """
    )

    # ========================================================
    # OBJECTIVES
    # ========================================================

    st.subheader("🎯 Project Objectives")

    objectives = [
        "Predict expected traffic volume using Machine Learning.",
        "Monitor traffic conditions across Maharashtra.",
        "Visualize traffic locations using an interactive map.",
        "Identify Low, Medium and High traffic conditions.",
        "Display weather conditions and their possible traffic impact.",
        "Identify public holidays and weekends.",
        "Maintain prediction history for analysis.",
        "Provide traffic analytics and ML model comparison."
    ]

    for objective in objectives:
        st.write(f"✅ {objective}")

    st.markdown("---")

    # ========================================================
    # KEY FEATURES
    # ========================================================

    st.subheader("⭐ Key Features")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            ### 🚗 Traffic Prediction

            • Machine Learning based vehicle prediction  
            • Traffic level classification  
            • Road capacity consideration  
            • Accident information  
            • Hour and day based prediction
            """
        )

        st.markdown(
            """
            ### 🗺️ Maharashtra Traffic Map

            • Maharashtra traffic locations  
            • Interactive map  
            • Traffic severity markers  
            • Highway information  
            • District-wise traffic data
            """
        )

    with col2:

        st.markdown(
            """
            ### 🌦️ Weather Intelligence

            • Real weather information  
            • Temperature  
            • Rainfall  
            • Humidity  
            • Wind speed  
            • Weather condition
            """
        )

        st.markdown(
            """
            ### 🎉 Holiday Intelligence

            • Maharashtra public holidays  
            • Festival information  
            • Weekend identification  
            • Holiday traffic impact indicator
            """
        )

    st.markdown("---")

    # ========================================================
    # TECHNOLOGY STACK
    # ========================================================

    st.subheader("💻 Technology Stack")

    technology_df = pd.DataFrame({
        "Technology": [
            "Python",
            "Streamlit",
            "Pandas",
            "NumPy",
            "Scikit-learn",
            "Matplotlib",
            "Folium",
            "Open-Meteo API"
        ],

        "Purpose": [
            "Application development",
            "Web dashboard",
            "Data processing",
            "Numerical computation",
            "Machine Learning",
            "Data visualization",
            "Interactive maps",
            "Real weather data"
        ]
    })

    st.dataframe(
        technology_df,
        width="stretch",
        hide_index=True
    )

    st.markdown("---")

    # ========================================================
    # MACHINE LEARNING
    # ========================================================

    st.subheader("🤖 Machine Learning")

    st.write(
        """
        The system uses supervised Machine Learning to estimate
        traffic volume from traffic-related input features.
        Multiple regression models can be compared to identify
        a suitable model for the application.
        """
    )

    ml_models = [
        "🌲 Random Forest Regressor",
        "🌳 Decision Tree Regressor",
        "📈 Gradient Boosting Regressor"
    ]

    for model_name in ml_models:
        st.write(f"• {model_name}")

    st.markdown("---")

    # ========================================================
    # DATA USED
    # ========================================================

    st.subheader("📊 Data Used")

    data_sources = [
        "Traffic dataset used for Machine Learning",
        "Maharashtra traffic and location data",
        "Weather information from Open-Meteo",
        "Maharashtra public holiday information",
        "Road capacity and accident information"
    ]

    for source in data_sources:
        st.write(f"📌 {source}")

    st.markdown("---")

    # ========================================================
    # TRAFFIC LEVELS
    # ========================================================

    st.subheader("🚦 Traffic Classification")

    traffic_df = pd.DataFrame({
        "Traffic Level": [
            "🟢 LOW",
            "🟡 MEDIUM",
            "🔴 HIGH"
        ],

        "Meaning": [
            "Normal traffic conditions",
            "Moderate congestion expected",
            "Heavy congestion expected"
        ]
    })

    st.dataframe(
        traffic_df,
        width="stretch",
        hide_index=True
    )

    st.markdown("---")

    # ========================================================
    # BENEFITS
    # ========================================================

    st.subheader("🌍 How This Project Can Help People")

    benefits = [
        "Helps users understand expected traffic conditions.",
        "Supports better route and travel planning.",
        "Highlights high-traffic Maharashtra locations.",
        "Provides weather and holiday context for traffic.",
        "Helps analyze traffic patterns using historical data.",
        "Can be extended to real-time traffic monitoring."
    ]

    for benefit in benefits:
        st.write(f"💡 {benefit}")

    st.markdown("---")

    # ========================================================
    # FUTURE SCOPE
    # ========================================================

    st.subheader("🚀 Future Scope")

    future_scope = [
        "Real-time traffic data integration",
        "Live GPS and route-based traffic analysis",
        "Real-time accident alerts",
        "Automatic traffic congestion alerts",
        "Mobile application integration",
        "Advanced traffic forecasting",
        "Smart route recommendation",
        "Government/open-data integration"
    ]

    for item in future_scope:
        st.write(f"🔹 {item}")

    st.markdown("---")

    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.subheader("⚠️ Important Note")

    st.info(
        """
        Traffic predictions are estimates generated by the
        Machine Learning model and should not be considered
        as guaranteed real-world traffic conditions.

        Weather information depends on the availability of
        the external weather service.
        """
    )

    st.markdown("---")

    # ========================================================
    # PROJECT STATUS
    # ========================================================

    st.subheader("📌 Project Status")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "🧠 ML System",
            "Active"
        )

    with c2:
        st.metric(
            "🗺️ Maharashtra Map",
            "Active"
        )

    with c3:
        st.metric(
            "🌦️ Weather System",
            "Active"
        )

    st.markdown("---")

    st.success(
        "🚦 Maharashtra Traffic Intelligence System — "
        "Smart Mobility Through Data & Machine Learning"
    )