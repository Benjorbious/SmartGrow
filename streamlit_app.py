import streamlit as st
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import tempfile
import pandas as pd
import plotly.express as px

# -------------------------
# Page Config and Dark Mode Toggle
# -------------------------
st.set_page_config(page_title="SmartGrow", layout="wide")
dark_mode = st.sidebar.checkbox("Dark Mode", value=True)

# -------------------------
# Dark / Light Mode CSS
# -------------------------
if dark_mode:
    st.markdown("""
    <style>
    .stApp {background-color: #1E1E1E; color: #F0F0F0;}
    [data-testid="stSidebar"] {background-color: #2B2B2B !important;}
    [data-testid="stSidebar"] * {color: #FFFFFF !important;}
    header, [data-testid="stToolbar"] {
        background-color: #2B2B2B !important;
        color: #FFFFFF !important;
    }
    button[title="Main menu"] {color: #FFFFFF !important;}
    .dataframe th, .dataframe td {color: #F0F0F0 !important;}
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {color: #F0F0F0 !important;}
    input, textarea, .stSlider > div > div {
        color: #FFFFFF !important;
        background-color: #000000 !important;
    }
    div[data-baseweb="select"] > div, input[type="number"] {
        background-color: #000000 !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
        border: 1px solid #FFFFFF !important;
    }
    .stSlider > div[data-baseweb="slider"] > div {background-color: #000000 !important;}
    .stSlider > div[data-baseweb="slider"] [role="slider"] {
        background-color: #FFFFFF !important;
        width: 18px !important;
        height: 18px !important;
    }
    button[kind="secondary"], button[kind="primary"], .stButton > button {
        background-color: #000000 !important;
        color: #FFFFFF !important;
        border: 1px solid #FFFFFF !important;
        border-radius: 8px !important;
    }
    button[kind="secondary"]:hover, button[kind="primary"]:hover, .stButton > button:hover {
        background-color: #333333 !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background-color: #000000 !important;
        color: #FFFFFF !important;
        border: 1px solid #FFFFFF !important;
        border-radius: 8px !important;
    }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    .stApp {background-color: #FFF8E7; color: #333333;}
    [data-testid="stSidebar"] {background-color: #F5EACB !important; color: #333333 !important;}
    header, [data-testid="stToolbar"] {background-color: #F5EACB !important; color: #333333 !important;}
    button[title="Main menu"] {color: #333333 !important;}
    .dataframe th, .dataframe td {color: #333333 !important;}
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {color: #333333 !important;}
    input, textarea, .stSlider > div > div {
        color: #000000 !important;
        background-color: #ffffff !important;
    }
    </style>
    """, unsafe_allow_html=True)

# -------------------------
# App Title and Description
# -------------------------
st.title("SmartGrow — AI vs Manual Plant Care")
st.markdown("""
Compare **AI recommendations** for plant care with **manual inputs**.  
- Plant Health: higher is better.  
- Green = Good ✅, Yellow = Moderate ⚡, Red = Poor ⚠️  
- Icons show what is applied: Water 💧, Sun 🌞, Nutrients 🌿
""")

# -------------------------
# Simple Greenhouse Environment (Gymnasium compatible)
# -------------------------
class SimpleGreenhouseEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, episode_length=12):
        super().__init__()
        self.episode_length = episode_length
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.water = float(0.5 + np.random.normal(scale=0.05))
        self.light = float(0.5 + np.random.normal(scale=0.05))
        self.nutrients = float(0.5 + np.random.normal(scale=0.05))
        self.t = 0
        self.history = []
        obs = self._get_obs()
        info = {}
        return obs, info  # (obs, info) for Gymnasium

    def _get_obs(self):
        return np.array([self.water, self.light, self.nutrients, self.t / (self.episode_length - 1)], dtype=np.float32)

    def step(self, action):
        if isinstance(action, np.ndarray) and action.ndim == 2:
            action = action[0]

        delta = np.clip(action, -1.0, 1.0) * 0.2
        self.water = np.clip(self.water + delta[0], 0.0, 1.0)
        self.light = np.clip(self.light + delta[1], 0.0, 1.0)
        self.nutrients = np.clip(self.nutrients + delta[2], 0.0, 1.0)
        self.t += 1

        mean_input = (self.water + self.light + self.nutrients) / 3.0
        balance_penalty = np.std([self.water, self.light, self.nutrients])
        mean_penalty = abs(mean_input - 0.6)
        variability = np.mean(np.abs(delta))

        plant_health = 1.0 - (balance_penalty * 1.2) - (mean_penalty * 1.0) + (variability * 0.8)
        plant_health = float(np.clip(plant_health, -1.0, 2.0))
        done = (self.t >= self.episode_length)

        info = {
            "plant_health": plant_health,
            "water": self.water,
            "light": self.light,
            "nutrients": self.nutrients,
        }
        return self._get_obs(), plant_health, done, False, info  # (obs, reward, terminated, truncated, info)

# -------------------------
# Helpers
# -------------------------
def input_label(value):
    if value < 0.33: return "Low 💧/🌞/🌿"
    elif value < 0.66: return "Medium 💧/🌞/🌿"
    else: return "High 💧/🌞/🌿"

# -------------------------
# Crop-Aware Health Labels
# -------------------------
def health_label(value):
    # Crop-specific visual flair (purely cosmetic)
    if "Maize" in crop_choice:
        good_icon, moderate_icon, poor_icon = "🌽✅", "🌽⚡", "🌽⚠️"
    elif "Wheat" in crop_choice:
        good_icon, moderate_icon, poor_icon = "🌾✅", "🌾⚡", "🌾⚠️"
    elif "Tomatoes" in crop_choice:
        good_icon, moderate_icon, poor_icon = "🍅✅", "🍅⚡", "🍅⚠️"
    elif "Rice" in crop_choice:
        good_icon, moderate_icon, poor_icon = "🍚✅", "🍚⚡", "🍚⚠️"
    else:  # Beans
        good_icon, moderate_icon, poor_icon = "🫘✅", "🫘⚡", "🫘⚠️"

    if value < 0.4:
        return f"Poor {poor_icon}", "red"
    elif value < 0.7:
        return f"Moderate {moderate_icon}", "yellow"
    else:
        return f"Good {good_icon}", "green"

def manual_to_numeric(choice):
    return {"Low": 0.2, "Medium": 0.5, "High": 0.8}[choice]
# -------------------------
# Sidebar Controls
# -------------------------
st.sidebar.header("Controls")

# Crop selection (visual only)
crop_choice = st.sidebar.selectbox(
    "Select Crop",
    ["Maize 🌽", "Wheat 🌾", "Tomatoes 🍅", "Rice 🍚", "Beans 🫘"],
    index=0
)
st.sidebar.caption(f"Current crop selected: **{crop_choice}**")

episode_length = st.sidebar.slider("Episode length (days)", 6, 30, 12)
train_timesteps = st.sidebar.slider("AI Training Timesteps", 1000, 20000, 4000, step=1000)
seed = st.sidebar.number_input("Random seed", value=0, min_value=0)
manual_water = st.sidebar.selectbox("Manual Water", ["Low", "Medium", "High"])
manual_light = st.sidebar.selectbox("Manual Light", ["Low", "Medium", "High"])
manual_nutrients = st.sidebar.selectbox("Manual Nutrients", ["Low", "Medium", "High"])
do_train = st.sidebar.button("Train / Retrain AI")
load_model_file = st.sidebar.file_uploader("Load model (.zip)", type=["zip"])

tmpdir = tempfile.gettempdir()
model_path = os.path.join(tmpdir, "ppo_greenhouse.zip")

# -------------------------
# Environment and Model Setup
# -------------------------
def make_env():
    return SimpleGreenhouseEnv(episode_length=episode_length)

env = DummyVecEnv([make_env])
model = None

if load_model_file:
    uploaded_path = os.path.join(tmpdir, "uploaded_model.zip")
    with open(uploaded_path, "wb") as f:
        f.write(load_model_file.getbuffer())
    model = PPO.load(uploaded_path, env=env)
    st.sidebar.success("Loaded AI model.")

# -------------------------
# Training
# -------------------------
if do_train:
    env = DummyVecEnv([make_env])
    model = PPO("MlpPolicy", env, verbose=0, seed=int(seed))
    model.learn(total_timesteps=int(train_timesteps))
    model.save(model_path)
    st.sidebar.success(f"Training complete ({train_timesteps} timesteps) and saved.")

# -------------------------
# Run Episodes
# -------------------------
def run_episode_ai(model, env_inst):
    obs, _ = env_inst.reset()
    done = False
    records = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env_inst.step(action)
        health_str, color = health_label(info["plant_health"])
        records.append({
            "Day": len(records) + 1,
            "Water": input_label(info["water"]),
            "Light": input_label(info["light"]),
            "Nutrients": input_label(info["nutrients"]),
            "Plant Health": health_str,
            "Color": color,
            "Numeric Health": info["plant_health"]
        })
    return pd.DataFrame(records)

def run_episode_manual(env_inst, water, light, nutrients):
    obs, _ = env_inst.reset()
    done = False
    records = []
    manual_numeric = np.array([
        manual_to_numeric(water) + np.random.normal(scale=0.05),
        manual_to_numeric(light) + np.random.normal(scale=0.05),
        manual_to_numeric(nutrients) + np.random.normal(scale=0.05)
    ])
    manual_numeric = np.clip(manual_numeric, 0, 1)
    while not done:
        obs, reward, done, _, info = env_inst.step(manual_numeric)
        health_str, color = health_label(info["plant_health"])
        records.append({
            "Day": len(records) + 1,
            "Water": water,
            "Light": light,
            "Nutrients": nutrients,
            "Plant Health": health_str,
            "Color": color,
            "Numeric Health": info["plant_health"]
        })
    return pd.DataFrame(records)

# -------------------------
# Layout and Visualization
# -------------------------
st.header("🌿 Day-by-Day Comparison")

if model is None:
    st.warning("Train or load an AI model first.")
else:
    env_ai = make_env()
    df_ai = run_episode_ai(model, env_ai)
    env_manual = make_env()
    df_manual = run_episode_manual(env_manual, manual_water, manual_light, manual_nutrients)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🤖 AI Recommendations")
        st.dataframe(
            df_ai.style.apply(lambda col: ['background-color:' + c for c in df_ai["Color"]], subset=["Plant Health"])
        )
    with col2:
        st.subheader("🧑 Manual Inputs")
        st.dataframe(
            df_manual.style.apply(lambda col: ['background-color:' + c for c in df_manual["Color"]], subset=["Plant Health"])
        )

    df_plot = pd.DataFrame({
        "Day": list(df_ai["Day"]) * 2,
        "Type": ["AI"] * len(df_ai) + ["Manual"] * len(df_manual),
        "Plant Health": list(df_ai["Numeric Health"]) + list(df_manual["Numeric Health"])
    })

    if dark_mode:
        bg_color = "#1E1E1E"
        font_color = "#F0F0F0"
        ai_color = "limegreen"
        manual_color = "orange"
    else:
        bg_color = "#FFF8E7"
        font_color = "#333333"
        ai_color = "green"
        manual_color = "darkorange"

    fig = px.bar(
        df_plot,
        x="Day",
        y="Plant Health",
        color="Type",
        barmode="group",
        color_discrete_map={"AI": ai_color, "Manual": manual_color},
        hover_data={"Plant Health": True, "Type": True, "Day": True},
        labels={"Plant Health": "Plant Health", "Type": "Input"}
    )

    fig.update_layout(
        title="🌿 AI vs Manual Plant Health (Hover for exact values)",
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font_color=font_color,
        title_font_color=font_color
    )
    st.plotly_chart(fig, use_container_width=True)

    avg_ai = df_ai["Numeric Health"].mean()
    avg_manual = df_manual["Numeric Health"].mean()
    st.markdown(f"**Average Plant Health:** 🤖 AI = {avg_ai:.2f}, 🧑 Manual = {avg_manual:.2f}")
