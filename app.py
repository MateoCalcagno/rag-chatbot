import uuid
import streamlit as st
import chromadb
import anthropic
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Chatbot sobre tu PDF", page_icon="💬")
st.title("💬 Preguntale a tu PDF")
st.caption("RAG con ChromaDB, sentence-transformers y Claude")


@st.cache_resource
def cargar():
    modelo = SentenceTransformer("all-MiniLM-L6-v2")
    return modelo, chromadb.EphemeralClient(), anthropic.Anthropic()


modelo, db, cliente = cargar()


def leer_pdf(archivo):
    return "\n".join(p.extract_text() or "" for p in PdfReader(archivo).pages)


def partir(texto, tam=800, solape=100):
    return [texto[i:i + tam] for i in range(0, len(texto), tam - solape)]


def indexar(archivo):
    texto = leer_pdf(archivo)
    if not texto.strip():
        return None
    trozos = partir(texto)
    nombre = f"docs_{uuid.uuid4().hex}"
    coleccion = db.create_collection(nombre)
    coleccion.add(
        ids=[str(i) for i in range(len(trozos))],
        documents=trozos,
        embeddings=modelo.encode(trozos).tolist(),
    )
    return nombre


def preguntar(coleccion, pregunta):
    res = coleccion.query(
        query_embeddings=modelo.encode([pregunta]).tolist(), n_results=3
    )
    trozos = res["documents"][0]
    contexto = "\n---\n".join(trozos)
    try:
        r = cliente.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            system="Respondé usando solo el contexto. Si la respuesta no está, decí que no lo sabés.",
            messages=[{"role": "user", "content": f"Contexto:\n{contexto}\n\nPregunta: {pregunta}"}],
        )
        return r.content[0].text, trozos
    except anthropic.APIStatusError as e:
        return f"Error de la API (¿sin crédito?): {e.message}", []


archivo = st.file_uploader("Subí un PDF con texto", type="pdf")

if archivo is None:
    st.info("Subí un PDF para empezar a hacerle preguntas.")
    st.stop()

# Si es un archivo nuevo, lo procesamos y empezamos una conversación limpia
clave = (archivo.name, archivo.size)
if st.session_state.get("clave") != clave:
    if "coleccion" in st.session_state:
        db.delete_collection(st.session_state["coleccion"])
    with st.spinner("Procesando el PDF..."):
        nombre = indexar(archivo)
    if nombre is None:
        st.error("No pude sacar texto de ese PDF. ¿Es un escaneo o una foto?")
        st.stop()
    st.session_state["coleccion"] = nombre
    st.session_state["clave"] = clave
    st.session_state["mensajes"] = []

coleccion = db.get_collection(st.session_state["coleccion"])

for m in st.session_state["mensajes"]:
    with st.chat_message(m["rol"]):
        st.markdown(m["texto"])

if pregunta := st.chat_input("Escribí tu pregunta..."):
    st.session_state["mensajes"].append({"rol": "user", "texto": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)

    with st.chat_message("assistant"):
        with st.spinner("Buscando..."):
            respuesta, trozos = preguntar(coleccion, pregunta)
        st.markdown(respuesta)
        if trozos:
            with st.expander("Fragmentos usados"):
                for t in trozos:
                    st.text(t)

    st.session_state["mensajes"].append({"rol": "assistant", "texto": respuesta})
