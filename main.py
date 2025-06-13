from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.chat_engine import ContextChatEngine
import json

# Carica i dati dal file JSON
with open("ricettario.json", encoding="utf-8") as f:
    raw_data = json.load(f)

# Costruisci i documenti dalle ricette e dagli ingredienti
documents = []

# Ricette
for recipe in raw_data.get("recipes", []):
    text = f"""
### {recipe['title']}

**Descrizione:** {recipe.get('description', '')}

**Ingredienti:** {', '.join(recipe.get('ingredients', []))}

**Preparazione:** {recipe.get('preparation', '')}

**Contenuto:** {recipe.get('content', '')}

**Immagini:** {', '.join(recipe.get('images', []))}

**URL:** {recipe.get('url', '')}
"""
    documents.append(Document(
        text=text,
        metadata={
            "title": recipe["title"],
            "url": recipe.get("url"),
            "ingredients": recipe.get("ingredients", [])
        }
    ))

# Ingredienti
for ingredient in raw_data.get("ingredients", []):
    text = f"""
### {ingredient['title']}

**Descrizione:** {ingredient.get('description', '')}

**Contenuto:** {ingredient.get('content', '')}

**Immagini:** {', '.join(ingredient.get('images', []))}

**URL:** {ingredient.get('url', '')}
"""
    documents.append(Document(
        text=text,
        metadata={"title": ingredient["title"], "url": ingredient.get("url")}
    ))

# Inizializza Ollama + modello di embedding
llm = Ollama(model="gemma3:4b")  # Assicurati che "gemma3:4b" sia installato con `ollama pull gemma3:4b`
embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-base")

# Configura le impostazioni globali
Settings.llm = llm
Settings.embed_model = embed_model

# Costruisci l'indice vettoriale
index = VectorStoreIndex.from_documents(documents)
retriever = index.as_retriever()

# Prompt di sistema separato
system_prompt = """
Sei un assistente esperto di cucina giapponese. Rispondi sempre e solo in base alle ricette e agli ingredienti contenuti nella knowledge base.
Quando ti chiedono di consigliare ricette, cerca tra gli ingredienti e le descrizioni.
Quando ti chiedono \"come si fa\", rispondi con la preparazione completa.
Quando ti chiedono un ingrediente, spiega a cosa serve, come si usa e come si prepara.
Quando ti chiedono cosa cucinare con un ingrediente, cerca tra le ricette che lo contengono nel campo ingredienti.
Includi i link alle ricette o ingredienti se presenti.
Non inventare nulla: se non sai, di' che non è presente.
Usa uno stile cordiale e conciso.
"""

# Crea il motore di chat con contesto mantenuto
chat_engine = ContextChatEngine.from_defaults(
    retriever=retriever,
    llm=llm,
    system_prompt=system_prompt
)

# Interazione testuale con contesto
if __name__ == "__main__":
    print("Assistente di cucina pronto! Scrivi una domanda o 'esci' per terminare.")
    while True:
        q = input("> ")
        if q.lower() in {"esci", "exit", "quit"}:
            break
        response = chat_engine.chat(q)
        print(response.response)
