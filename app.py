import uuid
from bisect import bisect_right
import streamlit as st
import chromadb
import anthropic
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="Chatbot sobre tu PDF",
    page_icon="💬"
)

st.title("💬 Preguntale a tu PDF")
st.caption("RAG con ChromaDB, sentence-transformers y Claude")


@st.cache_resource
def cargar():
    modelo = SentenceTransformer("all-MiniLM-L6-v2")
    return modelo, chromadb.EphemeralClient(), anthropic.Anthropic()


modelo, db, cliente = cargar()


def leer_pdf(archivo):
    # Devuelve una lista con el texto de cada página (la posición 0 es la página 1)
    return [p.extract_text() or "" for p in PdfReader(archivo).pages]


def partir_con_paginas(paginas, tam=800, solape=100):
    """Parte el texto completo (como antes) y calcula en qué página cae cada fragmento."""
    texto = "\n".join(paginas)

    # Posición del texto donde empieza cada página
    inicios = []
    pos = 0
    for p in paginas:
        inicios.append(pos)
        pos += len(p) + 1  # +1 por el "\n" que une las páginas

    def pagina_de(posicion):
        return bisect_right(inicios, posicion)  # páginas numeradas desde 1

    resultado = []
    for i in range(0, len(texto), tam - solape):
        trozo = texto[i:i + tam]
        if trozo.strip():
            desde = pagina_de(i)
            hasta = pagina_de(min(i + len(trozo), len(texto)) - 1)
            resultado.append((trozo, desde, hasta))
    return resultado


def etiqueta_paginas(desde, hasta):
    return f"página {desde}" if desde == hasta else f"páginas {desde}-{hasta}"


def indexar(archivo):
    paginas = leer_pdf(archivo)

    if not "".join(paginas).strip():
        return None

    partes = partir_con_paginas(paginas)

    trozos = [t for t, _, _ in partes]
    metadatos = [{"desde": d, "hasta": h} for _, d, h in partes]

    nombre = f"docs_{uuid.uuid4().hex}"

    coleccion = db.create_collection(nombre)

    coleccion.add(
        ids=[str(i) for i in range(len(trozos))],
        documents=trozos,
        embeddings=modelo.encode(trozos).tolist(),
        metadatas=metadatos,
    )

    return nombre


def preguntar(coleccion, pregunta):
    res = coleccion.query(
        query_embeddings=modelo.encode([pregunta]).tolist(),
        n_results=5
    )

    trozos = res["documents"][0]
    metas = res["metadatas"][0]

    # A Claude solo le pasamos el texto; la página es para mostrarla en la interfaz
    contexto = "\n---\n".join(trozos)

    try:
        r = cliente.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            system=(
                "Respondé usando solo el contexto. "
                "Si la respuesta no está, decí que no lo sabés."
            ),
            messages=[
                {
                    "role": "user",
                    "content": f"Contexto:\n{contexto}\n\nPregunta: {pregunta}"
                }
            ],
        )

        return r.content[0].text, [
            (m["desde"], m["hasta"], t) for m, t in zip(metas, trozos)
        ]

    except anthropic.APIStatusError as e:
        return f"Error de la API (¿sin crédito?): {e.message}", []


# =========================
# CARGA DEL PDF
# =========================

archivo = st.file_uploader(
    "📄 Subí un PDF con texto",
    type="pdf"
)

if archivo is None:
    st.info("Subí un PDF para empezar a hacerle preguntas.")
    st.stop()


# Si es un archivo nuevo, lo procesamos
clave = (archivo.name, archivo.size)

if st.session_state.get("clave") != clave:

    if "coleccion" in st.session_state:
        db.delete_collection(st.session_state["coleccion"])

    with st.spinner("⚙️ Procesando el PDF y preparando el RAG..."):
        nombre = indexar(archivo)

    if nombre is None:
        st.error(
            "No pude sacar texto de ese PDF. "
            "¿Es un escaneo o una foto?"
        )
        st.stop()

    st.session_state["coleccion"] = nombre
    st.session_state["clave"] = clave
    st.session_state["mensajes"] = []


# =========================
# DOCUMENTO LISTO
# =========================

st.success("🟢 RAG listo para consultar")

st.markdown(
    f"""
### 📄 Documento cargado

**{archivo.name}**

Podés hacer preguntas sobre el contenido del documento.
"""
)

st.divider()


# =========================
# CHAT
# =========================

coleccion = db.get_collection(
    st.session_state["coleccion"]
)


for m in st.session_state["mensajes"]:
    with st.chat_message(m["rol"]):
        st.markdown(m["texto"])


if pregunta := st.chat_input(
    "💬 Escribí tu pregunta..."
):

    st.session_state["mensajes"].append(
        {
            "rol": "user",
            "texto": pregunta
        }
    )

    with st.chat_message("user"):
        st.markdown(pregunta)

    with st.chat_message("assistant"):

        with st.spinner("🔎 Buscando información relevante..."):
            respuesta, fuentes = preguntar(
                coleccion,
                pregunta
            )

        st.markdown(respuesta)

        if fuentes:
            paginas_usadas = sorted(
                {p for d, h, _ in fuentes for p in (d, h)}
            )
            st.caption(
                "📖 Páginas consultadas: "
                + ", ".join(str(p) for p in paginas_usadas)
            )

            with st.expander("📚 Fragmentos recuperados"):

                for i, (d, h, t) in enumerate(fuentes, 1):
                    st.markdown(f"**Fragmento {i} · {etiqueta_paginas(d, h)}**")
                    st.text(t)

    st.session_state["mensajes"].append(
        {
            "rol": "assistant",
            "texto": respuesta
        }
    )