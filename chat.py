import chromadb
import anthropic
from sentence_transformers import SentenceTransformer

modelo = SentenceTransformer("all-MiniLM-L6-v2")
coleccion = chromadb.PersistentClient(path="db").get_collection("docs")
cliente = anthropic.Anthropic()

def preguntar(pregunta):
    res = coleccion.query(
        query_embeddings=modelo.encode([pregunta]).tolist(), n_results=3
    )
    contexto = "\n---\n".join(res["documents"][0])
    try:
        r = cliente.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            system="Respondé usando solo el contexto. Si la respuesta no está, decí que no lo sabés.",
            messages=[{"role": "user", "content": f"Contexto:\n{contexto}\n\nPregunta: {pregunta}"}],
        )
        return r.content[0].text
    except anthropic.APIStatusError as e:
        return f"Error de la API (¿sin crédito?): {e.message}"

while True:
    p = input("\nPregunta (enter para salir): ")
    if not p:
        break
    print(preguntar(p))
