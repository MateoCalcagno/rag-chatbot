# Preguntale a tu PDF

Chatbot que responde preguntas sobre un PDF que subís desde el navegador, usando solo lo que dice el documento. Si la respuesta no está, lo dice en vez de inventar.

## Cómo funciona (RAG)

1. **Subís un PDF** desde la barra lateral.
2. **Indexado**: se extrae el texto, se corta en fragmentos de 800 caracteres con solapamiento y cada fragmento se convierte en un embedding con sentence-transformers. Se guardan en ChromaDB, en memoria y por sesión.
3. **Consulta**: tu pregunta se convierte en embedding y se buscan los 3 fragmentos más cercanos.
4. **Respuesta**: esos fragmentos se envían a Claude (Haiku 4.5), que responde solo con ese contexto. Cada respuesta muestra los fragmentos usados.

El PDF no se guarda en disco: al cerrar la app, todo se borra.

## Tecnologías

Python, Streamlit, ChromaDB, sentence-transformers, Claude API, pypdf.

## Cómo correrlo

```bash
git clone https://github.com/MateoCalcagno/rag-chatbot
cd rag-chatbot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY="tu-clave"
streamlit run app.py
```

Después abrí http://localhost:8501 y subí un PDF con texto (no sirven los escaneos).

