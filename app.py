import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


st.set_page_config(
    page_title="Gemelo digital GLP",
    page_icon="🚨",
    layout="wide"
)

st.title("Gemelo digital experimental de GLP")
st.warning(
    "SIMULACIÓN EDUCATIVA. No es un detector de gas certificado ni debe "
    "controlar alarmas físicas. Utilice un detector y sistema de alarma "
    "certificados independientes."
)
st.caption(
    "Las señales son sintéticas e ilustrativas. No son mediciones reales "
    "ni representan un modelo físico validado."
)


# -----------------------------
# 1. Parámetros de simulación
# -----------------------------
with st.sidebar:
    st.header("Configuración de simulación")

    duration = st.slider("Duración (segundos)", 120, 1200, 480, 60)
    leak_start = st.slider("Inicio de fuga (s)", 10, duration - 40, 80)
    leak_end = st.slider(
        "Fin de fuga (s)",
        leak_start + 10,
        duration - 1,
        min(duration - 1, max(leak_start + 10, 260))
    )
    source = st.slider("Intensidad de fuga (ppm/s, ficticia)", 0.2, 4.0, 1.4, 0.1)
    ventilation = st.slider("Disipación por segundo", 0.001, 0.10, 0.025, 0.001)
    noise = st.slider("Desviación de ruido (ppm)", 0.0, 30.0, 9.0, 0.5)
    distance = st.select_slider("Distancia (cm)", options=[40, 50, 60], value=50)
    height = st.select_slider("Altura (cm)", options=[22, 42, 62], value=42)
    angle = st.select_slider("Ángulo (°)", options=[-30, 0, 45], value=0)
    horizon = st.slider("Horizonte de predicción (s)", 1, 20, 10)
    threshold = st.slider("Umbral didáctico de alarma (ppm)", 20, 300, 147)
    q = st.number_input("Kalman: varianza de proceso Q", min_value=0.001, value=1.5, step=0.5)
    r = st.number_input("Kalman: varianza de medición R", min_value=0.001, value=81.0, step=5.0)
    seed = st.number_input("Semilla aleatoria", value=42, step=1)


# -----------------------------
# 2. Generación de señales
# -----------------------------
@st.cache_data(show_spinner=False)
def simulate(n, start, end, intensity, dissipation, noise_sd,
             sensor_distance, sensor_height, sensor_angle, random_seed):
    rng = np.random.default_rng(int(random_seed))
    t = np.arange(n, dtype=int)

    # Factores ilustrativos; no calibrados experimentalmente.
    geo = (
        (50 / sensor_distance) ** 1.15
        * (1.0 if sensor_height == 42 else 0.72)
        * (1.0 if sensor_angle == 0 else 0.83)
    )

    true_signal = np.zeros(n)
    for i in range(1, n):
        influx = intensity * geo if start <= i < end else 0.0
        true_signal[i] = max(
            0.0,
            true_signal[i - 1] * (1.0 - dissipation) + influx
        )

    measured = np.maximum(
        0.0,
        true_signal + rng.normal(0.0, noise_sd, n)
    )

    return pd.DataFrame({
        "segundo": t,
        "concentracion_real_simulada": true_signal,
        "lectura_ruidosa": measured
    })


def kalman_filter(values, process_variance, measurement_variance):
    estimate = float(values[0])
    covariance = max(float(measurement_variance), 1.0)
    output = []

    for measurement in values:
        covariance += process_variance
        gain = covariance / (covariance + measurement_variance)
        estimate += gain * (float(measurement) - estimate)
        covariance *= (1.0 - gain)
        output.append(estimate)

    return np.asarray(output)


def make_features(df, signal_column, prediction_horizon):
    signal = pd.Series(df[signal_column].to_numpy())

    features = pd.DataFrame({
        "nivel": signal,
        "lag1": signal.shift(1),
        "lag3": signal.shift(3),
        "lag5": signal.shift(5),
        "media5": signal.rolling(5).mean(),
        "media12": signal.rolling(12).mean(),
        "pendiente3": signal - signal.shift(3),
        "pendiente8": signal - signal.shift(8),
        "desv5": signal.rolling(5).std().fillna(0)
    })

    target = df["concentracion_real_simulada"].shift(-prediction_horizon)
    data = features.copy()
    data["objetivo"] = target
    data["segundo"] = df["segundo"]

    return data.dropna().reset_index(drop=True)


@st.cache_data(show_spinner="Entrenando modelos predictivos...")
def evaluate_model(signal_frame, prediction_horizon):
    results = []
    predictions = {}

    for signal_name, filter_name in [
        ("lectura_ruidosa", "Sin Kalman"),
        ("lectura_kalman", "Con Kalman")
    ]:
        data = make_features(signal_frame, signal_name, prediction_horizon)
        split = int(len(data) * 0.75)
        train = data.iloc[:split]
        test = data.iloc[split:]

        feature_columns = [
            column for column in data.columns
            if column not in ("objetivo", "segundo")
        ]

        models = [
            (
                "Random Forest",
                RandomForestRegressor(
                    n_estimators=120,
                    min_samples_leaf=3,
                    random_state=42,
                    n_jobs=-1
                )
            ),
            (
                "Gradient Boosting",
                HistGradientBoostingRegressor(
                    max_iter=120,
                    max_leaf_nodes=15,
                    l2_regularization=1.0,
                    random_state=42
                )
            )
        ]

        for model_name, model in models:
            model.fit(train[feature_columns], train["objetivo"])
            predicted = np.maximum(0, model.predict(test[feature_columns]))
            actual = test["objetivo"].to_numpy()

            results.append({
                "Modelo": model_name,
                "Filtro": filter_name,
                "MAE (ppm)": mean_absolute_error(actual, predicted),
                "RMSE (ppm)": np.sqrt(mean_squared_error(actual, predicted)),
                "R²": r2_score(actual, predicted)
            })

            predictions[(model_name, filter_name)] = pd.DataFrame({
                "segundo": test["segundo"].to_numpy(),
                "real_futuro": actual,
                "prediccion": predicted
            })

    return pd.DataFrame(results), predictions


# -----------------------------
# 3. Recalcular al mover sliders
# -----------------------------
# Streamlit vuelve a ejecutar el archivo cuando cambia un parámetro.
# Por eso la simulación y la predicción se actualizan automáticamente.
df = simulate(
    duration, leak_start, leak_end, source, ventilation, noise,
    distance, height, angle, seed
)

df["lectura_kalman"] = kalman_filter(
    df["lectura_ruidosa"].to_numpy(), float(q), float(r)
)

results, predictions = evaluate_model(df, horizon)


# -----------------------------
# 4. Señales de concentración
# -----------------------------
st.subheader("1. Señales de concentración")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df["segundo"],
    y=df["concentracion_real_simulada"],
    name="Concentración real sintética",
    mode="lines"
))
fig.add_trace(go.Scatter(
    x=df["segundo"],
    y=df["lectura_ruidosa"],
    name="Lectura con ruido",
    mode="lines",
    opacity=0.55
))
fig.add_trace(go.Scatter(
    x=df["segundo"],
    y=df["lectura_kalman"],
    name="Estimación Kalman",
    mode="lines"
))
fig.add_hline(
    y=threshold,
    line_dash="dash",
    annotation_text="Umbral didáctico"
)
fig.update_layout(
    xaxis_title="Tiempo (s)",
    yaxis_title="Concentración (ppm)",
    height=390
)
st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# 5. Evaluación de modelos
# -----------------------------
st.subheader("2. Comparación predictiva (último 25 % de la serie)")
st.dataframe(
    results.style.format({
        "MAE (ppm)": "{:.2f}",
        "RMSE (ppm)": "{:.2f}",
        "R²": "{:.3f}"
    }),
    hide_index=True,
    use_container_width=True
)

best = results.sort_values("RMSE (ppm)").iloc[0]
st.info(
    f"Menor RMSE de prueba: {best['Modelo']} / {best['Filtro']} "
    f"({best['RMSE (ppm)']:.2f} ppm). Resultado sintético, "
    "no constituye una validación real."
)


# -----------------------------
# 6. Predicción y alarma virtual
# -----------------------------
st.subheader("3. Gemelo digital y luz de emergencia virtual")

prediction_options = list(predictions.keys())
selected_label = st.selectbox(
    "Modelo para visualizar y activar la alarma virtual",
    [f"{model} | {filter_name}" for model, filter_name in prediction_options]
)
selected_key = prediction_options[
    [f"{model} | {filter_name}" for model, filter_name in prediction_options]
    .index(selected_label)
]
prediction_df = predictions[selected_key].copy()

index = st.slider(
    "Instante de reproducción dentro de la prueba",
    min_value=0,
    max_value=len(prediction_df) - 1,
    value=len(prediction_df) - 1
)

selected_row = prediction_df.iloc[index]
predicted_value = float(selected_row["prediccion"])
alarm_on = predicted_value >= threshold

prediction_fig = go.Figure()
prediction_fig.add_trace(go.Scatter(
    x=prediction_df["segundo"],
    y=prediction_df["real_futuro"],
    name="Concentración real futura (sintética)"
))
prediction_fig.add_trace(go.Scatter(
    x=prediction_df["segundo"],
    y=prediction_df["prediccion"],
    name="Predicción para t+h"
))
prediction_fig.add_hline(
    y=threshold,
    line_dash="dash",
    annotation_text="Umbral didáctico"
)
prediction_fig.add_trace(go.Scatter(
    x=[int(selected_row["segundo"])],
    y=[predicted_value],
    mode="markers",
    marker=dict(size=13, color="red" if alarm_on else "green"),
    name="Instante seleccionado"
))
prediction_fig.update_layout(
    xaxis_title="Tiempo de emisión de predicción t (s)",
    yaxis_title="Concentración (ppm)",
    height=370
)
st.plotly_chart(prediction_fig, use_container_width=True)

metric1, metric2, metric3 = st.columns(3)
metric1.metric("Predicción t + horizonte", f"{predicted_value:.1f} ppm")
metric2.metric("Umbral de simulación", f"{threshold} ppm")
metric3.metric("Tiempo simulado", f"{int(selected_row['segundo'])} s")

if alarm_on:
    st.markdown(
        """
        <div style="
            background-color:#FDE8E7;
            border:2px solid #D32F2F;
            border-radius:12px;
            padding:18px;
            text-align:center;
            color:#B71C1C;
            font-size:20px;
            font-weight:700;">
            🔴 LUZ VIRTUAL ROJA ENCENDIDA — predicción igual o superior al umbral
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <div style="
            background-color:#E7F6EC;
            border:2px solid #2E9D59;
            border-radius:12px;
            padding:18px;
            text-align:center;
            color:#207A43;
            font-size:20px;
            font-weight:700;">
            🟢 LUZ VIRTUAL VERDE — predicción inferior al umbral
        </div>
        """,
        unsafe_allow_html=True
    )

state = {
    "simulacion": True,
    "segundo": int(selected_row["segundo"]),
    "horizonte_s": int(horizon),
    "modelo": selected_key[0],
    "filtro": selected_key[1],
    "prediccion_ppm": round(predicted_value, 3),
    "umbral_ppm": int(threshold),
    "luz_virtual_roja": bool(alarm_on)
}

with st.expander("Ver estado de la simulación (JSON)"):
    st.json(state)

st.download_button(
    "Descargar señales CSV",
    df.to_csv(index=False).encode("utf-8"),
    "senales_glp_sinteticas.csv",
    "text/csv"
)
st.download_button(
    "Descargar métricas CSV",
    results.to_csv(index=False).encode("utf-8"),
    "metricas_modelos.csv",
    "text/csv"
)
st.download_button(
    "Descargar predicciones CSV",
    prediction_df.to_csv(index=False).encode("utf-8"),
    "predicciones.csv",
    "text/csv"
)

st.caption(
    "Limitaciones: simulador de una zona, sin dinámica CFD; señales sintéticas "
    "no calibradas con un sensor real. Para uso real se requieren mediciones "
    "calibradas, ensayos independientes y una alarma certificada e independiente."
)
