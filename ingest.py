from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer

modelo = SentenceTransformer("all-MiniLM-L6-v2")
db = chromadb.PersistentClient(path="db")
coleccion = db.get_or_create_collection("docs")

def leer_pdf(ruta):
    return "\n".join(p.extract_text() or "" for p in PdfReader(ruta).pages)

def partir(texto, tam=800, solape=100):
    return [texto[i:i + tam] for i in range(0, len(texto), tam - solape)]

trozos = partir(leer_pdf("documento.pdf"))
coleccion.add(
    ids=[str(i) for i in range(len(trozos))],
    documents=trozos,
    embeddings=modelo.encode(trozos).tolist(),
)
print(f"Guardados {len(trozos)} trozos")
