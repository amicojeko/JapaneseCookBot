from flask import Flask, request, jsonify, Response
from flask_cors import CORS # Importa CORS
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.chat_engine import ContextChatEngine
import json
import time
import re
import logging

# ---------- Logging Setup ----------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------- Flask App Initialization ----------
app = Flask(__name__)
# !!! NOTA IMPORTANTE: Abilita CORS per permettere al tuo frontend
# di comunicare con questo server. Questo è fondamentale.
CORS(app)

# ---------- Load and Prepare Documents ----------
logging.info("Caricamento del ricettario da ricettario.json...")
try:
    with open("ricettario.json", encoding="utf-8") as f:
        raw_data = json.load(f)
except FileNotFoundError:
    logging.error("Errore: il file 'ricettario.json' non è stato trovato.")
    # Crea un file di esempio se non esiste
    sample_data = {
        "recipes": [{
            "title": "Ricetta di Esempio", "description": "Questa è una ricetta di prova.",
            "ingredients": ["ingrediente1", "ingrediente2"], "preparation": "Mescolare tutto.",
            "url": "http://example.com"
        }],
        "ingredients": [{"title": "Ingrediente di Esempio", "description": "Un ingrediente di prova."}]
    }
    with open("ricettario.json", "w", encoding="utf-8") as f:
        json.dump(sample_data, f)
    raw_data = sample_data
    logging.info("Creato un file 'ricettario.json' di esempio.")


documents = []

# Index recipes
for recipe in raw_data.get("recipes", []):
    content = f"""
### {recipe['title']}
**Descrizione:** {recipe.get('description', '')}
**Ingredienti:** {', '.join(recipe.get('ingredients', []))}
**Preparazione:** {recipe.get('preparation', '')}
**Contenuto:** {recipe.get('content', '')}
**Immagini:** {', '.join(recipe.get('images', []))}
**URL:** {recipe.get('url', '')}
"""
    documents.append(
        Document(
            text=content,
            metadata={
                "title": recipe['title'],
                "url": recipe.get('url'),
                "ingredients": recipe.get('ingredients', [])
            }
        )
    )

# Index ingredients
for ingredient in raw_data.get("ingredients", []):
    content = f"""
### {ingredient['title']}
**Descrizione:** {ingredient.get('description', '')}
**Contenuto:** {ingredient.get('content', '')}
**Immagini:** {', '.join(ingredient.get('images', []))}
**URL:** {ingredient.get('url', '')}
"""
    documents.append(
        Document(
            text=content,
            metadata={"title": ingredient['title'], "url": ingredient.get('url')}
        )
    )

logging.info(f"Caricati {len(documents)} documenti nella knowledge base.")

# ---------- Initialize LLM and Embeddings ----------
logging.info("Inizializzazione del modello LLM (Ollama) e degli embeddings...")
# Configurazione Ollama per usare il server esistente
llm = Ollama(
    model="gemma3:4b",
    base_url="http://localhost:11434",
    request_timeout=60.0,
    temperature=0.0 # NOTA: Impostato a 0.0 per ridurre le allucinazioni
)
embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-base")
Settings.llm = llm
Settings.embed_model = embed_model
logging.info("Modelli inizializzati con successo.")

# ---------- Build Index and Retriever ----------
logging.info("Creazione dell'indice vettoriale...")
index = VectorStoreIndex.from_documents(documents)
retriever = index.as_retriever()
logging.info("Indice creato con successo.")

# ---------- System Prompt ----------
system_prompt = """
Sei un assistente esperto di cucina giapponese. Rispondi sempre e solo in base alle ricette e agli ingredienti contenuti nella knowledge base.
Quando ti chiedono di consigliare ricette, cerca tra gli ingredienti e le descrizioni.
Quando ti chiedono "come si fa", rispondi con la preparazione completa.
Quando ti chiedono un ingrediente, spiega a cosa serve, come si usa e come si prepara.
Quando ti chiedono cosa cucinare con un ingrediente, cerca tra le ricette che lo contengono nel campo ingredienti.
Non inventare nulla: se non sai, di' che non è presente.
Usa uno stile cordiale e conciso.

IMPORTANTE: Formatta la tua risposta usando la sintassi Markdown standard.
- Per i link, usa: [testo del link](url)
- Per le immagini, usa: ![descrizione immagine](url)
- Per il grassetto, usa: **testo in grassetto**
- Per i titoli, usa: # titolo di primo livello, ## titolo di secondo livello, ### titolo di terzo livello
"""

# ---------- Chat Engine Setup ----------
# Questa istanza del chat_engine è globale e manterrà la memoria.
chat_engine = ContextChatEngine.from_defaults(
    retriever=retriever,
    llm=llm,
    system_prompt=system_prompt
)
logging.info("Chat engine globale pronto.")

def extract_openai_message(data):
    """Estrae l'ultimo messaggio utente dal formato OpenAI."""
    if 'messages' not in data or not isinstance(data['messages'], list):
        return ""
    for msg in reversed(data['messages']):
        if isinstance(msg, dict) and msg.get('role') == 'user':
            content = msg.get('content', '')
            if isinstance(content, str): return content
            elif isinstance(content, list):
                text_parts = [item.get('text', '') for item in content if isinstance(item, dict) and item.get('type') == 'text']
                return ' '.join(text_parts)
    return ""

def is_naming_request(message):
    """Verifica se il messaggio è una richiesta automatica di naming da parte del frontend."""
    naming_patterns = [
        "based on the chat history, give this conversation a name",
        "name this conversation in 10 characters or less"
    ]
    message_lower = message.lower()
    return any(pattern in message_lower for pattern in naming_patterns)

# ---------- API Endpoint ----------
@app.route('/v1/chat/completions', methods=['POST'])
def chat_endpoint():
    try:
        data = request.get_json()
        if not data:
            logging.warning("Richiesta ricevuta senza dati JSON.")
            return jsonify({"error": "No JSON data provided"}), 400

        logging.info(f"Dati ricevuti: {json.dumps(data, indent=2)}")
        user_message = extract_openai_message(data)
        if not user_message:
            logging.warning("Nessun messaggio utente trovato nella richiesta.")
            return jsonify({"error": "No user message found"}), 400

        logging.info(f"Messaggio utente estratto: '{user_message}'")

        if data.get('stream', False):
            logging.info("Richiesta di streaming rilevata. Avvio della generazione.")
            return Response(stream_generator(user_message), mimetype='text/event-stream')

        # Fallback per richieste non-streaming
        logging.info("Richiesta non-streaming. Generazione della risposta completa.")
        result = chat_engine.chat(user_message)
        response_content = str(result.response)
        response_data = {
            "id": f"chatcmpl-{int(time.time())}", "object": "chat.completion", "created": int(time.time()),
            "model": "gemma3:4b",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": response_content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": len(user_message.split()), "completion_tokens": len(response_content.split()), "total_tokens": len(user_message.split()) + len(response_content.split())}
        }
        return jsonify(response_data)

    except Exception as e:
        logging.error(f"Errore critico durante l'elaborazione della richiesta: {e}", exc_info=True)
        return jsonify({"error": {"message": str(e), "type": "internal_error"}}), 500

@app.route('/reset', methods=['POST'])
def reset_conversation():
    """Resetta la cronologia del chat engine."""
    global chat_engine
    logging.info("Richiesta di reset della conversazione ricevuta.")
    chat_engine.reset()
    logging.info("Conversazione resettata con successo.")
    return jsonify({"status": "success", "message": "Conversation has been reset."})


def stream_generator(user_message):
    """Generatore per la risposta in streaming, che gestisce sia le richieste normali sia quelle di naming."""
    chat_id = f"chatcmpl-stream-{int(time.time())}"
    model_name = "gemma3:4b"

    if is_naming_request(user_message):
        logging.info("--- Inizio Gestione Naming ---")
        match = re.search(r"```\n(.*?)\n---------", user_message, re.DOTALL)
        actual_user_text = match.group(1).strip() if match else ""
        name = "Chat"
        if actual_user_text:
            logging.info(f"Testo originale per il nome: '{actual_user_text}'")
            naming_prompt = f"Genera un titolo breve in italiano (massimo 3 parole) per una conversazione che inizia con: '{actual_user_text}'. Rispondi solo con il titolo, senza virgolette."
            logging.info(f"Prompt per la generazione del nome: '{naming_prompt}'")
            try:
                response = llm.complete(naming_prompt)
                name = response.text.strip().replace('"', '')
                logging.info(f"Nome generato dal LLM: '{name}'")
            except Exception as e:
                logging.error(f"Errore durante la generazione del nome: {e}")
                name = "Chat Iniziale"
        else:
            logging.warning("Non è stato possibile estrarre il testo per generare il nome.")

        chunk = {"id": chat_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": model_name, "choices": [{"index": 0, "delta": {"role": "assistant", "content": name}, "finish_reason": None}]}
        yield f"data: {json.dumps(chunk)}\n\n"
        final_chunk = {"id": chat_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": model_name, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"
        logging.info("--- Fine Gestione Naming ---")
        return

    try:
        logging.info("Avvio streaming per la query utente, mantenendo il contesto esistente.")
        streaming_response = chat_engine.stream_chat(user_message)

        first_token = True
        for token in streaming_response.response_gen:
            delta = {"content": token}
            if first_token:
                delta["role"] = "assistant"
                first_token = False

            chunk = {"id": chat_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": model_name, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
            yield f"data: {json.dumps(chunk)}\n\n"

        final_chunk = {"id": chat_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": model_name, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        yield f"data: {json.dumps(final_chunk)}\n\n"

    except Exception as e:
        logging.error(f"Errore durante lo streaming: {e}", exc_info=True)
        error_message = "\n\nMi dispiace, si è verificato un errore tecnico."
        error_chunk = {"id": chat_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": model_name, "choices": [{"index": 0, "delta": {"content": error_message}, "finish_reason": "stop"}]}
        yield f"data: {json.dumps(error_chunk)}\n\n"

    yield "data: [DONE]\n\n"

# ---------- Health Check Endpoint ----------
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

# ---------- Run Server ----------
if __name__ == '__main__':
    logging.info("Avvio del server Flask su [http://0.0.0.0:11435](http://0.0.0.0:11435)")
    app.run(host='0.0.0.0', port=11435, debug=False, threaded=True)
