

import math
import pickle
import requests
from typing import Any, Dict, List, Tuple

import numpy as np
import streamlit as st

# Page config - must run before other Streamlit calls that produce output
st.set_page_config(page_title="IPL Score Predictor 2024", layout="wide")

# List of teams (consistent ordering used for one-hot encoding)
TEAMS = [
    "Chennai Super Kings",
    "Delhi Capitals",
    "Punjab Kings",
    "Kolkata Knight Riders",
    "Mumbai Indians",
    "Rajasthan Royals",
    "Royal Challengers Bangalore",
    "Sunrisers Hyderabad",
    "Gujarat Titans",
    "Lucknow Super Giants",
]


def one_hot_team(team: str, teams: List[str]) -> List[int]:
    """Return a one-hot vector for `team` according to `teams` ordering."""
    return [1 if t == team else 0 for t in teams]


# Model loader (cached): use newer st.cache_resource if available, otherwise fallback to st.cache
if hasattr(st, "cache_resource"):
    @st.cache_resource
    def load_model(path: str):
        with open(path, "rb") as f:
            return pickle.load(f)
else:
    @st.cache(allow_output_mutation=True)
    def load_model(path: str):
        with open(path, "rb") as f:
            return pickle.load(f)


def fetch_weather_from_api(url_template: str, api_key: str, city: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Fetch weather using a user-provided URL template.
    The template must contain {city} and {key} placeholders.
    Example (OpenWeatherMap): "https://api.openweathermap.org/data/2.5/weather?q={city}&appid={key}&units=metric"
    Returns (success, data). On success data contains keys:
      - temp_c (float or None), humidity (int or None), wind_m_s (float or None), weather_main (str or None), raw (full JSON)
    On failure data contains {"error": "..."}
    """
    if not url_template or not api_key:
        return False, {"error": "No URL template or API key provided"}
    try:
        url = url_template.replace("{city}", city).replace("{key}", api_key)
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        j = resp.json()
        temp_c = None
        humidity = None
        wind_ms = None
        weather_main = None
        if isinstance(j, dict):
            if "main" in j:
                temp_c = j["main"].get("temp")
                humidity = j["main"].get("humidity")
            if "wind" in j:
                wind_ms = j["wind"].get("speed")
            if "weather" in j and isinstance(j["weather"], list) and j["weather"]:
                weather_main = j["weather"][0].get("main")
        return True, {
            "temp_c": temp_c,
            "humidity": humidity,
            "wind_m_s": wind_ms,
            "weather_main": weather_main,
            "raw": j,
        }
    except Exception as e:
        return False, {"error": str(e)}


def app():
    # Title and background
    st.markdown("<h1 style='text-align: center; color: white;'> IPL Score Predictor 2024 </h1>", unsafe_allow_html=True)
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://i.postimg.cc/W3D9LPMR/Designer.png");
            background-attachment: fixed;
            background-size: cover;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Description"):
        st.info(
            "A ML Model to predict IPL scores between teams during an ongoing match.\n\n"
            "To increase reliability the model expects at least 5 completed overs. "
            "If you'd like to include weather as model features, enable the weather option and provide an API template/key. "
            "NOTE: If your model wasn't trained with weather features, enable 'Model trained with weather' only AFTER retraining the model accordingly."
        )

    # Sidebar: model loading
    st.sidebar.header("Model / Files")
    model_path = st.sidebar.text_input("Model filename (in app folder)", value="ml_model.pkl")
    uploaded_model = st.sidebar.file_uploader("Or upload a model .pkl (optional)", type=["pkl"])
    model = None
    if uploaded_model is not None:
        try:
            model = pickle.load(uploaded_model)
            st.sidebar.success("Model uploaded successfully")
        except Exception as e:
            st.sidebar.error(f"Failed to load uploaded model: {e}")
    else:
        try:
            model = load_model(model_path)
            st.sidebar.success(f"Loaded model: {model_path}")
        except FileNotFoundError:
            st.sidebar.error(f"Model file not found: {model_path}. You can upload one on the sidebar.")
        except Exception as e:
            st.sidebar.error(f"Error loading model: {e}")

    # Match inputs
    st.header("Match Details")
    batting_team = st.selectbox("Select the Batting Team", TEAMS)
    # Bowling team choices exclude batting team
    bowling_options = [t for t in TEAMS if t != batting_team]
    bowling_team = st.selectbox("Select the Bowling Team", bowling_options)

    if bowling_team == batting_team:
        st.error("Bowling and Batting teams should be different")

    # Overs input split into completed overs and balls to avoid invalid fractions
    col1, col2 = st.columns(2)
    with col1:
        overs_completed = st.number_input("Completed overs (integer part)", min_value=5, max_value=19, value=5, step=1)
    with col2:
        balls_in_over = st.selectbox("Balls into current over (0-5)", [0, 1, 2, 3, 4, 5], index=0)
    overs = float(overs_completed) + (balls_in_over / 6.0)

    # Runs, wickets, last 5 overs stats
    runs = st.number_input("Enter current runs", min_value=0, max_value=500, value=0, step=1, format="%i")
    wickets = int(st.slider("Enter wickets fallen till now", 0, 10, 0))
    col3, col4 = st.columns(2)
    with col3:
        runs_in_prev_5 = st.number_input("Runs scored in the last 5 overs", min_value=0, max_value=int(runs), value=0, step=1, format="%i")
    with col4:
        wickets_in_prev_5 = st.number_input("Wickets taken in the last 5 overs", min_value=0, max_value=wickets, value=0, step=1, format="%i")

    # Weather integration (optional)
    st.subheader("Weather (optional)")
    use_weather = st.checkbox("Fetch weather now and include in features (requires API template & key)", value=False)
    st.markdown("If you enable weather, provide a URL template with placeholders `{city}` and `{key}`. Example (OpenWeatherMap):")
    st.code("https://api.openweathermap.org/data/2.5/weather?q={city}&appid={key}&units=metric")
    api_template = st.text_input("Weather API URL template (include {city} and {key})", value="")
    api_key = st.text_input("Weather API key", type="password")
    city = st.text_input("City for weather lookup", value="Mumbai")

    # Indicate if the model was trained with weather features (only check if true)
    trained_with_weather = st.checkbox("My model was trained with weather features (enable only if true)", value=False)

    # Compose base features
    features = []
    features += one_hot_team(batting_team, TEAMS)  # batting team one-hot
    features += one_hot_team(bowling_team, TEAMS)  # bowling team one-hot
    features += [int(runs), int(wickets), float(round(overs, 3)), int(runs_in_prev_5), int(wickets_in_prev_5)]

    # Weather fetching
    weather_info = {}
    if use_weather:
        success, data = fetch_weather_from_api(api_template, api_key, city)
        if not success:
            st.warning(f"Weather fetch failed: {data.get('error')}")
            weather_features = [0.0, 0.0, 0.0]
        else:
            weather_info = data
            temp = data.get("temp_c")
            hum = data.get("humidity")
            wind = data.get("wind_m_s")
            temp = 0.0 if temp is None else float(temp)
            hum = 0.0 if hum is None else float(hum)
            wind = 0.0 if wind is None else float(wind)
            weather_features = [temp, hum, wind]
            st.success(f"Weather fetched for {city}: temp={temp}°C, humidity={hum}%, wind={wind} m/s")
    else:
        weather_features = [0.0, 0.0, 0.0]

    # Append weather to features only if the model expects it
    if trained_with_weather:
        features += weather_features
    else:
        if use_weather:
            st.info("Weather fetched but NOT included in features because 'Model trained with weather features' is unchecked. Check it if your model expects weather inputs.")

    X = np.array([features], dtype=float)

    with st.expander("Feature summary (first row shown)"):
        st.write({"features_length": len(features)})
        st.write({"features": features})
        if weather_info:
            st.write({"weather_raw": weather_info.get("raw")})

    # Prediction action
    if st.button("Predict Score"):
        if model is None:
            st.error("No model loaded. Upload or specify a model file in the sidebar.")
            return
        try:
            pred = model.predict(X)
            my_prediction = int(round(float(pred[0])))
            lower = max(0, my_prediction - 5)
            upper = my_prediction + 5
            st.success(f"PREDICTED MATCH SCORE : {lower} to {upper}")
            st.info(f"Point estimate: {my_prediction}")
        except ValueError as ve:
            st.error(f"Model prediction failed due to feature shape mismatch or invalid input: {ve}")
            st.write(
                "Hint: Check model input dimendimensiures. If you enabled weather features, make sure "
                "your model was trained with those additional features and check 'My model was trained with weather features'."
            )
        except Exception as e:
            st.error(f"Prediction failed: {e}")


if __name__ == "__main__":
    app()
