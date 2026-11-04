import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

AUDIO_COLS = [
    "energy",
    "tempo",
    "danceability",
    "loudness",
    "liveness",
    "valence",
    "speechiness",
    "instrumentalness",
    "acousticness",
]

INFO_COLS = ["track_name", "track_artist", "playlist_genre"]


def cargar_datos(ruta_csv: str):
    df_clean = pd.read_csv(ruta_csv)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean[AUDIO_COLS])
    return df_clean, X_scaled


def buscar_cancion(texto: str, df_clean: pd.DataFrame) -> pd.DataFrame:
    mask = df_clean["track_name"].str.contains(texto, case=False, regex=False, na=False)
    return df_clean.loc[mask, INFO_COLS]


def recomendar_canciones(
    idx: int, df_clean: pd.DataFrame, X_scaled: np.ndarray, n: int = 5
):
    sims = cosine_similarity(X_scaled[idx].reshape(1, -1), X_scaled).ravel()
    sims[idx] = -np.inf

    top_idx = np.argsort(sims)[::-1][:n]
    recs = df_clean.loc[top_idx, INFO_COLS + AUDIO_COLS].copy()
    recs["similarity"] = sims[top_idx].round(3)
    return recs


def plot_recomendaciones(
    idx: int, recs: pd.DataFrame, df_clean: pd.DataFrame, X_scaled: np.ndarray
):
    angles = np.linspace(0, 2 * np.pi, len(AUDIO_COLS), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    def cerrar(v):
        return np.append(v, v[0])

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"polar": True})

    nombre_cancion = df_clean.loc[idx, "track_name"]
    base = cerrar(X_scaled[idx])
    ax.plot(angles, base, label=f"{nombre_cancion} (base)", linewidth=2)
    ax.fill(angles, base, alpha=0.1)

    for rec_idx, fila in recs.iterrows():
        ax.plot(
            angles,
            cerrar(X_scaled[rec_idx]),
            label=f"{fila['track_name']} ({fila['similarity']})",
        )

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(AUDIO_COLS)
    ax.set_title(f"Recomendaciones para '{nombre_cancion}'", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.5, 1.1))
    return fig
