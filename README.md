# Japanese Cookbot Backend

Un assistente di cucina giapponese basato su IA che fornisce ricette e informazioni sugli ingredienti.

## Prerequisiti

- Python 3.8 o superiore
- pip (Python package manager)
- Ollama installato e in esecuzione (per il modello LLM)

## Installazione

1. Clona il repository (se non l'hai già fatto):
   ```bash
   git clone <repository-url>
   cd JapaneseCookbot
   ```

2. Crea e attiva un ambiente virtuale (consigliato):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Su Windows: venv\Scripts\activate
   ```

3. Installa le dipendenze:
   ```bash
   pip install -r requirements.txt
   ```

4. Assicurati che Ollama sia in esecuzione con il modello `gemma3:4b` scaricato:
   ```bash
   ollama pull gemma3:4b
   ```

## Avvio del server

Per avviare il server di backend:

```bash
python voice_backend.py
```

Il server sarà disponibile all'indirizzo: [http://localhost:11435](http://localhost:11435)

## Endpoint disponibili

- `POST /v1/chat/completions` - Endpoint principale per la chat
- `POST /reset` - Resetta la conversazione
- `GET /health` - Controllo dello stato del server

## Frontend

Il progetto include un'interfaccia utente web interattiva nel file `frontend.html` che offre:
- Riconoscimento vocale integrato
- Sintesi vocale per le risposte
- Interfaccia utente intuitiva

Per utilizzare il frontend:
1. Avvia il server backend come descritto sopra
2. Apri il file `frontend.html` direttamente nel tuo browser
3. Consenti l'accesso al microfono quando richiesto
4. Usa il pulsante del microfono per parlare con l'assistente

## Struttura del progetto

- `frontend.html` - Interfaccia utente web con riconoscimento vocale
- `voice_backend.py` - Il server Flask principale
- `ricettario.json` - Il file JSON contenente le ricette e gli ingredienti
- `requirements.txt` - Le dipendenze Python richieste

## Configurazione

Il server si aspetta di trovare un file `ricettario.json` nella directory principale. Se il file non esiste, ne verrà creato uno di esempio all'avvio.

## Note

- Assicurati che il server Ollama sia in esecuzione su `http://localhost:11434`
- Il server utilizza il modello `gemma3:4b` come LLM predefinito
- La porta predefinita del server è 11435
- Il frontend si aspetta che il backend sia in ascolto su `http://localhost:11435`
- Per il riconoscimento vocale, assicurati di utilizzare un browser moderno (Chrome, Firefox, Edge) con accesso al microfono

## Sviluppo

Per contribuire al progetto, segui questi passaggi:

1. Crea un nuovo branch per la tua funzionalità
2. Fai le tue modifiche
3. Assicurati che il codice sia formattato correttamente
4. Invia una pull request

## Licenza

MIT
