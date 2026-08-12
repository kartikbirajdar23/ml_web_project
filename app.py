import streamlit.components.v1 as components
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from database import init_db, insert_prediction, get_all_predictions
init_db()  # Database initialize होईल
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from export_pdf import generate_traffic_pdf
from alerts import trigger_traffic_alert

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
        "📊 Analytics",
        "🤖 ML Model",
        "📁 Dataset",
        "ℹ️ About Project"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **B.Tech Final Year Project**

    AI/ML Based Smart Traffic Prediction System

    Developed using:
    Python • Streamlit • Pandas • NumPy • Scikit-learn • Matplotlib
    """
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
# ANALYTICS
# ============================================================

elif page == "📊 Analytics":
    st.title("📊 Traffic Analytics")
    st.write("Explore traffic patterns and relationships in the dataset.")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Hourly Traffic")
        hourly = df.groupby("Hour")["Vehicles"].mean()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(hourly.index, hourly.values, marker="o")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Average Vehicles")
        ax.grid(True)
        st.pyplot(fig)

    with col2:
        st.subheader("Traffic vs Road Capacity")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(df["Road_Capacity"], df["Vehicles"], alpha=0.3)
        ax.set_xlabel("Road Capacity")
        ax.set_ylabel("Traffic Volume")
        ax.grid(True)
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("📅 Traffic by Day")
    daily = df.groupby("Day_of_Week")["Vehicles"].mean()
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(day_names, daily.values)
    ax.set_xlabel("Day")
    ax.set_ylabel("Average Traffic")
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("🌦️ Weather Impact")
    weather_avg = df.groupby("Weather")["Vehicles"].mean()
    weather_labels = ["Clear", "Cloudy", "Rainy", "Stormy"]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(weather_labels, weather_avg.values)
    ax.set_ylabel("Average Traffic")
    st.pyplot(fig)

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
    st.title("ℹ️ About Project")
    st.markdown("""
    ## 🚦 Smart Traffic Prediction System
    **B.Tech Final Year Project**

    ### 🎯 Objective
    Develop an intelligent traffic prediction system using AI & ML.

    ### 🧠 Technologies Used
    Python • Streamlit • Pandas • NumPy • Matplotlib • Scikit-learn
    """)

    st.markdown("---")
    st.subheader("👨‍💻 Project Information")
    st.info("Smart Traffic Prediction System | B.Tech Final Year Project | AI & Machine Learning")

# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">
🚦 Smart Traffic Prediction System | B.Tech Final Year Project | AI & Machine Learning
</div>
""", unsafe_allow_html=True)