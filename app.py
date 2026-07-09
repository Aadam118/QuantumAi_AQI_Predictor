"""
Hybrid Quantum-Classical Neural Network — Air Quality Index (AQI) Predictor
============================================================================
Deploy target: Streamlit on Hugging Face Spaces

Architecture:
    Input (7 classical features)
        -> nn.Linear(7, 4) -> ReLU
        -> PennyLane TorchLayer (4-qubit AngleEmbedding + BasicEntanglerLayers)
        -> nn.Linear(4, 1) -> Sigmoid (raw output in [0, 1])
    AQI = raw_output * 500
"""

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
import streamlit as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

# ----------------------------------------------------------------------------
# 1. PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Quantum AQI Predictor",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# 2. QUANTUM CIRCUIT DEFINITION
# ----------------------------------------------------------------------------
N_QUBITS = 4
N_LAYERS = 3  # NOTE: must match the number of layers used during training.
              # If your saved .pth uses a different depth, change this value
              # so the weight tensor shapes match the checkpoint.

dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev, interface="torch")
def qnode(inputs, weights):
    """Main circuit used inside the TorchLayer during the forward pass."""
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="X")
    qml.BasicEntanglerLayers(weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(w)) for w in range(N_QUBITS)]


@qml.qnode(dev, interface="torch")
def vis_qnode(inputs, weights):
    """Separate circuit purely for Bloch-sphere visualization.
    Returns <X>, <Y>, <Z> expectation values for each of the 4 qubits."""
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="X")
    qml.BasicEntanglerLayers(weights, wires=range(N_QUBITS))
    x_exp = [qml.expval(qml.PauliX(w)) for w in range(N_QUBITS)]
    y_exp = [qml.expval(qml.PauliY(w)) for w in range(N_QUBITS)]
    z_exp = [qml.expval(qml.PauliZ(w)) for w in range(N_QUBITS)]
    return x_exp + y_exp + z_exp


weight_shapes = {"weights": (N_LAYERS, N_QUBITS)}


# ----------------------------------------------------------------------------
# 3. HYBRID MODEL DEFINITION
# ----------------------------------------------------------------------------
class climate_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.classical_in = nn.Linear(7, 4)
        self.quantum_layer = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.classical_out = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.relu(self.classical_in(x))
        x = self.quantum_layer(x)
        x = self.classical_out(x)
        return torch.sigmoid(x)  # raw output constrained to [0, 1]


# ----------------------------------------------------------------------------
# 4. MODEL LOADING (cached so it only happens once per session)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading pre-trained hybrid model...")
def load_model():
    model = climate_model()
    try:
        state_dict = torch.load("quantum_aqi_model.pth", map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        model.eval()
        return model, None
    except FileNotFoundError:
        return model, "Checkpoint 'quantum_aqi_model.pth' not found. Please add it next to app.py."
    except Exception as e:
        return model, f"Could not load checkpoint: {e}"


model, load_error = load_model()


# ----------------------------------------------------------------------------
# 5. AQI CATEGORY LOGIC
# ----------------------------------------------------------------------------
def get_aqi_category(aqi: float):
    """Returns (label, emoji, streamlit_display_fn) for a given AQI value."""
    if aqi <= 50:
        return "Good", "🟢", st.success
    elif aqi <= 100:
        return "Satisfactory", "🟡", st.success
    elif aqi <= 200:
        return "Moderate", "🟠", st.warning
    elif aqi <= 300:
        return "Poor", "🔴", st.warning
    elif aqi <= 400:
        return "Very Poor", "🟣", st.error
    else:
        return "Severe", "⚫", st.error


# ----------------------------------------------------------------------------
# 6. BLOCH SPHERE VISUALIZATION
# ----------------------------------------------------------------------------
def plot_bloch_sphere(bloch_vectors):
    """
    bloch_vectors: list of 4 tuples (x, y, z) — one Bloch vector per qubit.
    Draws a transparent sphere with a quiver arrow for each qubit state.
    """
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    # --- Transparent sphere surface ---
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 60)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(xs, ys, zs, alpha=0.2, cmap="coolwarm", linewidth=0, antialiased=True)

    # --- Reference axes through the sphere ---
    ax.plot([-1, 1], [0, 0], [0, 0], color="gray", lw=1)
    ax.plot([0, 0], [-1, 1], [0, 0], color="gray", lw=1)
    ax.plot([0, 0], [0, 0], [-1, 1], color="gray", lw=1)

    # --- Equator + meridian guide circles ---
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), np.zeros_like(theta), color="gray", lw=0.5, alpha=0.6)
    ax.plot(np.cos(theta), np.zeros_like(theta), np.sin(theta), color="gray", lw=0.5, alpha=0.6)

    # --- Quiver arrows for each qubit's Bloch vector ---
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
    for i, (bx, by, bz) in enumerate(bloch_vectors):
        ax.quiver(
            0, 0, 0, bx, by, bz,
            color=colors[i], linewidth=2.5, arrow_length_ratio=0.15,
            label=f"Qubit {i}"
        )

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Quantum State — Bloch Sphere Representation", fontsize=11)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_box_aspect([1, 1, 1])
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# 7. STREAMLIT UI
# ----------------------------------------------------------------------------
st.title("🌫️ Quantum-Enhanced Air Quality Index (AQI) Predictor")
st.caption("A Hybrid Quantum-Classical Neural Network built with PyTorch + PennyLane")

if load_error:
    st.warning(f"⚠️ {load_error} Predictions below use randomly initialized weights until the checkpoint is added.")

st.markdown("---")

feature_names = [
    "Temperature",
    "Humidity",
    "Wind Speed",
    "Pressure",
    "PM2.5 (normalized)",
    "PM10 (normalized)",
    "NO2 (normalized)",
]

left_col, right_col = st.columns([1, 1.3], gap="large")

with left_col:
    st.subheader("🔧 Input Features")
    st.write("Adjust the normalized sensor readings below:")

    feature_values = []
    for name in feature_names:
        val = st.slider(name, min_value=-1.0, max_value=1.0, value=0.0, step=0.01, key=name)
        feature_values.append(val)

    predict_clicked = st.button("🚀 Predict AQI", type="primary", use_container_width=True)

with right_col:
    st.subheader("📊 Prediction Results")

    if predict_clicked:
        input_tensor = torch.tensor([feature_values], dtype=torch.float32)

        with torch.no_grad():
            raw_output = model(input_tensor).item()

        aqi_value = raw_output * 500
        category, emoji, display_fn = get_aqi_category(aqi_value)

        # --- Prominent AQI display ---
        m1, m2 = st.columns(2)
        m1.metric("Predicted AQI", f"{aqi_value:.1f}")
        m2.metric("Category", f"{emoji} {category}")

        display_fn(f"**Air Quality Category: {category}** (AQI ≈ {aqi_value:.1f})")

        st.markdown("---")
        st.subheader("🧬 Quantum State Visualization")
        st.write(
            "The Bloch sphere below shows the 4-qubit state produced by the "
            "quantum layer for the current inputs, right after the classical "
            "`Linear(7,4) + ReLU` embedding stage."
        )

        # Recompute the intermediate classical representation that feeds
        # the quantum layer, then run the visualization QNode with the
        # trained quantum weights.
        with torch.no_grad():
            embed_input = torch.relu(model.classical_in(input_tensor))
            trained_weights = model.quantum_layer.weights.detach()
            vis_out = vis_qnode(embed_input[0], trained_weights)
            vis_out = torch.stack(vis_out).numpy()

        x_vals, y_vals, z_vals = vis_out[0:4], vis_out[4:8], vis_out[8:12]
        bloch_vectors = list(zip(x_vals, y_vals, z_vals))

        fig = plot_bloch_sphere(bloch_vectors)
        st.pyplot(fig)

        with st.expander("🔍 Raw expectation values"):
            for i, (bx, by, bz) in enumerate(bloch_vectors):
                st.write(f"**Qubit {i}** — ⟨X⟩ = {bx:.3f}, ⟨Y⟩ = {by:.3f}, ⟨Z⟩ = {bz:.3f}")
    else:
        st.info("Set the input features on the left and click **Predict AQI** to see results.")

st.markdown("---")
with st.expander("ℹ️ About the AQI Scale"):
    st.markdown(
        """
| AQI Range | Category      |
|-----------|---------------|
| 0–50      | 🟢 Good        |
| 51–100    | 🟡 Satisfactory|
| 101–200   | 🟠 Moderate    |
| 201–300   | 🔴 Poor        |
| 301–400   | 🟣 Very Poor   |
| 401–500+  | ⚫ Severe      |
        """
    )

st.caption("Built with PyTorch, PennyLane & Streamlit — deployed on Hugging Face Spaces.")
