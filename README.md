# Chatbot RAG sobre documentos

Chatbot que responde preguntas sobre un PDF usando solo lo que dice el documento. Si la respuesta no está, lo dice en vez de inventar.

## Cómo funciona

1. **Ingesta** (`ingest.py`): lee el PDF, lo parte en fragmentos y los convierte en embeddings con sentence-transformers. Los guarda en ChromaDB.
2. **Consulta** (`app.py`): convierte la pregunta en un embedding, busca los 3 fragmentos más parecidos y se los pasa a Claude para que responda solo con ese contexto.
3. **Interfaz**: chat web hecho con Streamlit, con un desplegable que muestra los fragmentos usados.

## Tecnologías

Python, ChromaDB, sentence-transformers, Claude API (Haiku 4.5), Streamlit.

## Cómo correrlo

```bash
git clone https://github.com/MateoCalcagno/rag-chatbot
cd rag-chatbot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY="tu-clave"
```

Poné tu PDF en la carpeta con el nombre `documento.pdf` y corré:

```bash
python3 ingest.py
streamlit run app.py
```

## Posibles mejoras

- Subir varios PDFs desde la interfaz
- Citar la página de donde sale cada respuesta
- Comparar tu CV contra ofertas laborales
