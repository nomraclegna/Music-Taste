import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.recommender import (
    AUDIO_COLS,
    INFO_COLS,
    buscar_cancion,
    cargar_datos,
    plot_recomendaciones,
    recomendar_canciones,
)

BLUE_ROSE_DIR = ROOT_DIR / "assets" / "blue_rose.png"

st.set_page_config(page_title="Music Taste V1", page_icon=BLUE_ROSE_DIR, layout="wide")

st.title("Music Taste - Recomendador Musical Basado en Contenido")
st.markdown(
    "Selecciona o busca una canción del catálogo para encontrar las **5 pistas acústicamente más similares** "
    "usando `StandardScaler` y `cosine_similarity`."
)


@st.cache_data
def obtener_datos():
    ruta_csv = ROOT_DIR / "data" / "spotify_clean.csv"
    return cargar_datos(str(ruta_csv))


df_clean, X_scaled = obtener_datos()

busqueda = st.text_input(
    ":material/search: Busca por título de canción o artista: ", value=""
)

if busqueda.strip():
    coincidencias = buscar_cancion(busqueda, df_clean)
else:
    coincidencias = df_clean[INFO_COLS]

if coincidencias.empty:
    st.warning("No se encontraron canciones con ese texto. Intenta con otra búsqueda.")
else:
    idx_base = st.selectbox(
        "Selecciona la canción base:",
        options=coincidencias.index,
        format_func=lambda i: (
            f"{df_clean.loc[i, 'track_name']} - {df_clean.loc[i, 'track_artist']} ({df_clean.loc[i, 'playlist_genre']})"
        ),
    )

    n_recs = st.slider("Número de recomendaciones:", min_value=3, max_value=10, value=5)

    recs = recomendar_canciones(idx_base, df_clean, X_scaled, n=n_recs)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader(":material/keep: Canción Seleccionada")
        st.dataframe(
            df_clean.loc[[idx_base], INFO_COLS + AUDIO_COLS], use_container_width=True
        )

        st.subheader(f":material/music_note_2: Top {n_recs} Recomendaciones")
        st.dataframe(recs, use_container_width=True)

    with col2:
        st.subheader(":material/analytics: Comparación de Perfil Acústico (Z-score)")
        fig = plot_recomendaciones(idx_base, recs, df_clean, X_scaled)
        st.pyplot(fig)
