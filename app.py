import streamlit as st
import chromadb
import anthropic
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Chatbot sobre mi CV", page_icon="💬")
st.title("💬 Preguntale a mi CV")
st.caption("RAG con ChromaDB, sentence-transformers y Claude")

@st.cache_resource
def cargar():
    modelo = SentenceTransformer("all-MiniLM-L6-v2")
    coleccion = chromadb.PersistentClient(path="db").get_collection("docs")
    return modelo, coleccion, anthropic.Anthropic()

modelo, coleccion, cliente = cargar()

def preguntar(pregunta):
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

if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

for m in st.session_state.mensajes:
    with st.chat_message(m["rol"]):
        st.markdown(m["texto"])

if pregunta := st.chat_input("Escribí tu pregunta..."):
    st.session_state.mensajes.append({"rol": "user", "texto": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)

    with st.chat_message("assistant"):
        with st.spinner("Buscando..."):
            respuesta, trozos = preguntar(pregunta)
        st.markdown(respuesta)
        if trozos:
            with st.expander("Fragmentos usados"):
                for t in trozos:
                    st.text(t)

    st.session_state.mensajes.append({"rol": "assistant", "texto": respuesta})
