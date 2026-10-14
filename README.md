# Open-Enricher — Pipeline Distribuita di Video Understanding Multimodale per Recommender Systems

**Open-Enricher** è una libreria sperimentale e una piattaforma modulare client-server per la comprensione automatica dei contenuti video (*Video Understanding*) e la generazione di metadati strutturati per **Recommender Systems (RecSys)**[cite: 7, 20, 26].

Il sistema disaccoppia il controllo dell'esecuzione (gestito da un **client leggero sul PC locale**) dal calcolo intensivo su GPU (eseguito su un **worker remoto ospitato su Google Colab** ed esposto tramite **FastAPI**)[cite: 1, 7, 8, 14]. L'architettura coordina cinque stadi sequenziali (**Audio**, **Color**, **VLM**, **LLM**, **Judger**) e integra strategie di **quantizzazione a 4-bit (`bitsandbytes`)** e rilascio dinamico della VRAM per eseguire modelli multimodali da 8 miliardi di parametri anche su GPU con memoria limitata (come la NVIDIA T4 da 15 GB di Google Colab)[cite: 5, 13, 14, 16, 20, 22].

---

## Indice

1. [Parte 1: Guida Operativa, Setup e Comandi CLI](#parte-1-guida-operativa-setup-e-comandi-cli)
   - [1.1 Requisiti e Installazione Lato Client (PC Locale)](#11-requisiti-e-installazione-lato-client-pc-locale)
   - [1.2 Configurazione dei Segreti (.env) e del Data Lake su Google Drive](#12-configurazione-dei-segreti-env-e-del-data-lake-su-google-drive)
   - [1.3 Connessione al Server: Modalità Tailscale vs Modalità Colab CLI](#13-connessione-al-server-modalità-tailscale-vs-modalità-colab-cli)
   - [1.4 Comandi CLI: Deploy, Esecuzione degli Stage e Monitoraggio](#14-comandi-cli-deploy-esecuzione-degli-stage-e-monitoraggio)
2. [Parte 2: Architettura del Sistema e Mappa della Codebase](#parte-2-architettura-del-sistema-e-mappa-della-codebase)
   - [2.1 Architettura a Livelli e Gestione della VRAM](#21-architettura-a-livelli-e-gestione-della-vram)
   - [2.2 Struttura Completa del Repository](#22-struttura-completa-del-repository)
   - [2.3 Dettaglio dei Sottosistemi e Interazione tra i Moduli](#23-dettaglio-dei-sottosistemi-e-interazione-tra-i-moduli)

---

## Parte 1: Guida Operativa, Setup e Comandi CLI

### 1.1 Requisiti e Installazione Lato Client (PC Locale)

Il PC locale svolge il ruolo di **Control Plane**: impacchetta il codice sorgente, trasferisce i file di configurazione al server remoto e invia i comandi di avvio dei vari stage tramite richieste HTTP REST[cite: 3, 4, 6, 10]. Sul PC locale **non** serve una GPU né l'installazione di PyTorch o modelli di Machine Learning[cite: 8, 10].

#### Prerequisiti di sistema sul PC locale
* **Python >= 3.10**[cite: 7].
* **Per il collegamento tramite Tailscale (`--mode tailscale`):**
  * Client **Tailscale** installato e connesso alla stessa rete VPN (tailnet) dell'istanza Google Colab[cite: 1, 3].
  * Client **OpenSSH** (`ssh` e `scp`) disponibile nel terminale (incluso di default su Windows 10/11 PowerShell, Linux e macOS)[cite: 3].
* **Per il collegamento tramite Colab CLI (`--mode colab-cli`):**
  * Binario CLI `colab` installato e autenticato nel `PATH` di sistema per gestire upload file, esecuzione comandi e tunneling SSH[cite: 1, 3, 10].

#### Setup dell'ambiente virtuale locale
Dalla cartella radice del progetto (`open-enricher`), apri un terminale ed esegui:

```powershell
# 1. Crea un ambiente virtuale Python locale
python -m venv venv

# 2. Attiva l'ambiente virtuale
# Su Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Su Linux / macOS:
# source venv/bin/activate

# 3. Installa le sole dipendenze leggere richieste dal client locale
pip install requests python-dotenv pydantic
```

---

### 1.2 Configurazione dei Segreti (`.env`) e del Data Lake su Google Drive

#### 1. Creazione del file `.env` locale
Nella cartella radice del repository (allo stesso livello di `deploy.py`), crea un file `.env` con il tuo token Hugging Face[cite: 3, 6, 14]. Il token è necessario per scaricare i modelli di diarizzazione vocale (Pyannote) usati da WhisperX e per autenticare le richieste verso l'Hub di Hugging Face[cite: 14, 16]:

```env
HF_TOKEN=hf_tuoTokenSegretoHuggingFace
```

> **Gestione sicura dei segreti:** Il file `.env` non viene incluso nell'archivio `.tgz` del codice sorgente[cite: 4, 6]. Lo script di deploy lo rileva localmente e lo invia direttamente in `/content/.env` sulla macchina Colab, dove `server.py` lo carica all'avvio tramite `load_dotenv("/content/.env")` prima di inizializzare gli stage[cite: 3, 6, 14].

#### 2. Struttura del Data Lake su Google Drive
Il worker remoto su Colab utilizza **Google Drive** (montato in `/content/drive/MyDrive`) come livello di storage persistente (*Data Lake*)[cite: 15]. Al primo avvio del server FastAPI, il modulo `storage.py` genera e verifica automaticamente la seguente gerarchia di cartelle dentro `Open-Enricher-Database`[cite: 14, 15]:

```text
/content/drive/MyDrive/Open-Enricher-Database/
├── Models-Cache/                  # Cache persistente dei modelli (evita nuovi download ad ogni sessione)
│   ├── huggingface/               # Cache HF_HOME (pesi Qwen3-VL, Qwen3-8B, Pyannote)
│   └── torch/                     # Cache TORCH_HOME
├── Video-Inputs/                  # <-- CARICA QUI I VIDEO DA ANALIZZARE (es. test.mp4)
└── Outputs/                       # Risultati JSON e artefatti prodotti da ciascuno stage
    ├── Audio-Results/             # Output dello stage audio (<video>_audio.json)
    ├── Color-Results/             # Output dello stage color (<video>_color.json + <video>_frames/)
    ├── VLMs-Results/              # Output dello stage vlm (<video>_vlm.json, _raw.txt, _sidecar.json)
    ├── LLMs-Results/              # Output dello stage llm (<video>_llm.json)
    └── Judger-Results/            # Output dello stage judger (<video>_judger.json)
```

Prima di avviare la pipeline, carica i tuoi file video (es. `test.mp4`) all'interno della cartella **`Open-Enricher-Database/Video-Inputs/`** su Google Drive[cite: 15, 16, 17].

---

### 1.3 Connessione al Server: Modalità Tailscale vs Modalità Colab CLI

Il sistema supporta nativamente due strategie di rete intercambiabili tramite l'argomento `--mode`[cite: 1, 3, 9, 10]:

#### Strategia 1: `tailscale` (Connessione diretta su VPN Mesh)
1. Nel notebook Google Colab, monta Google Drive in `/content/drive`[cite: 15].
2. Installa e avvia Tailscale sulla macchina Colab abilitando il demone SSH (`tailscale up --ssh`).
3. Copia l'indirizzo IP Tailscale assegnato alla VM Colab (es. `100.92.180.6`).
4. In questa modalità:
   * `deploy.py` usa `scp` e `ssh` diretti verso `root@<IP_TAILSCALE>`[cite: 3].
   * Il client HTTP invia le richieste direttamente a `http://<IP_TAILSCALE>:8000`[cite: 10].

#### Strategia 2: `colab-cli` (Tunneling automatizzato tramite Colab CLI)
1. Avvia una sessione Colab gestita tramite il tool `colab` assegnandole un nome (es. `enricher`) e assicurati che Google Drive sia montato[cite: 1, 3, 15, 30].
2. In questa modalità:
   * `deploy.py` usa i comandi `colab upload -s <SESSION>` e `colab exec -s <SESSION>` per caricare l'archivio, il file `.env` ed eseguire `launch.sh`[cite: 3].
   * Quando lanci il client per eseguire gli stage, il context manager `setup_server_connection` in `client/network.py` apre **automaticamente** in background un tunnel SSH (`colab tunnel --port 8000 --session <SESSION>`), invia le richieste a `http://localhost:8000` e chiude il processo del tunnel al termine dell'esecuzione[cite: 10].

---

### 1.4 Comandi CLI: Deploy, Esecuzione degli Stage e Monitoraggio

#### 1. Comando di Deploy (`deploy.py`)
Il comando `deploy.py` va eseguito dalla radice del progetto ogni volta che avvii una nuova istanza Colab o modifichi il codice locale[cite: 6].

| Parametro | Obbligatorio | Valori ammessi | Descrizione |
| :--- | :---: | :--- | :--- |
| `--mode` | Sì | `tailscale`, `colab-cli` | Strategia di rete da utilizzare per il trasferimento e il bootstrap[cite: 1]. |
| `--target` | Sì | Stringa (IP o Sessione) | Indirizzo IP Tailscale della VM Colab oppure nome della sessione Colab CLI[cite: 1]. |

**Esempi di Deploy:**

```powershell
# Deploy tramite Tailscale
python deploy.py --mode tailscale --target 100.92.180.6

# Deploy tramite Colab CLI
python deploy.py --mode colab-cli --target enricher
```

**Operazioni eseguite automaticamente dal deploy:**
1. Crea un archivio compresso pulito `open-enricher.tgz` contenente `src/`, `pyproject.toml`, `requirements.txt` e `README.md`[cite: 4, 6].
2. Carica `open-enricher.tgz` in `/content/open-enricher.tgz` e `.env` in `/content/.env`[cite: 3].
3. Lancia `deployment/launch.sh` sulla macchina Colab, il quale[cite: 2, 3]:
   * Termina eventuali worker `uvicorn` già attivi (`pkill -9 -f "uvicorn"`) liberando la porta `8000` e la memoria GPU[cite: 2].
   * Estrae il codice in `/content/open-enricher`[cite: 2].
   * Crea (solo al primo avvio) un ambiente virtuale in `/content/open-enricher/venv` con `--system-site-packages` per ereditare PyTorch e CUDA nativi di Colab[cite: 2].
   * Verifica e installa le dipendenze di `requirements.txt` (scrivendo il log dettagliato in `/content/open-enricher/install.log`)[cite: 2, 5].
   * Esporta i percorsi di cache (`HF_HOME`, `TORCH_HOME`, `HF_HUB_DISABLE_SYMLINKS=1`) e i path delle librerie NVIDIA/CUDA[cite: 2].
   * Avvia il server FastAPI in background con `nohup` sulla porta `8000`, salvando i log in `/content/open-enricher/server.log`[cite: 2].

---

#### 2. Comando di Esecuzione Pipeline (`src.open_enricher.client`)
Una volta che il server FastAPI è attivo su Colab, usa il modulo `src.open_enricher.client` per lanciare l'analisi su un video[cite: 8, 9].

| Parametro | Obbligatorio | Default | Descrizione |
| :--- | :---: | :--- | :--- |
| `--mode` | Sì | - | Modalità di connessione (`tailscale` o `colab-cli`)[cite: 9]. |
| `--target` | Sì | - | IP Tailscale della macchina Colab o nome della sessione `colab-cli`[cite: 9]. |
| `--video` | Sì | - | Nome del file video con estensione presente in `Video-Inputs/` (es. `test.mp4`)[cite: 9, 15]. |
| `--stages` | No | `audio color vlm llm judger` | Lista di stage da eseguire in ordine, separati da spazio[cite: 9, 13]. |
| `--timeout` | No | `3600` | Timeout HTTP massimo (in secondi) per ogni singolo stage[cite: 9, 10]. |

**Esempi di Esecuzione:**

```powershell
# 1. Eseguire tutti gli stage di default in sequenza (via Tailscale)
python -m src.open_enricher.client --mode tailscale --target 100.92.180.6 --video test.mp4

# 2. Eseguire i 4 stage principali (Audio -> Color -> VLM -> LLM)
python -m src.open_enricher.client --mode tailscale --target 100.92.180.6 --video test.mp4 --stages audio color vlm llm

# 3. Eseguire solamente gli stage VLM e LLM (riutilizzando i JSON Audio e Color già salvati su Drive)
python -m src.open_enricher.client --mode tailscale --target 100.92.180.6 --video test.mp4 --stages vlm llm

# 4. Eseguire un singolo stage (es. solo VLM) con timeout personalizzato
python -m src.open_enricher.client --mode tailscale --target 100.92.180.6 --video test.mp4 --stages vlm --timeout 7200

# 5. Eseguire la pipeline tramite Colab CLI (tunnel SSH gestito automaticamente)
python -m src.open_enricher.client --mode colab-cli --target enricher --video test.mp4 --stages audio color vlm llm
```

Al termine di ciascuno stage, il client stampa nel terminale il payload JSON di stato restituito dal server e verifica che non vi siano errori prima di procedere allo stage successivo[cite: 8, 11].

---

#### 3. Comandi di Monitoraggio e Diagnostica in Tempo Reale
Durante l'esecuzione degli stage più intensivi (`audio`, `vlm`, `llm`), puoi aprire un **secondo terminale** sul tuo PC per monitorare in tempo reale i log del worker su Colab[cite: 2, 3]:

```powershell
# Visualizza in tempo reale i log del server FastAPI e l'avanzamento dei modelli
ssh root@100.92.180.6 "tail -n 50 -f /content/open-enricher/server.log"

# Controlla l'occupazione della VRAM e il carico della GPU su Colab
ssh root@100.92.180.6 "nvidia-smi"

# Ispeziona il log di installazione di pip in caso di problemi durante il deploy
ssh root@100.92.180.6 "tail -n 40 /content/open-enricher/install.log"
```

---

## Parte 2: Architettura del Sistema e Mappa della Codebase

### 2.1 Architettura a Livelli e Gestione della VRAM

Il sistema è progettato seguendo una separazione rigorosa in tre livelli architetturali[cite: 10, 11, 12, 14, 15]:

1. **Infrastructure Layer (`deployment/*`, `client/network.py`, `remote/server.py`):** Gestisce il trasporto di rete (`SSH`/`SCP`/`Colab CLI`), i tunnel SSH, il client HTTP e il server web FastAPI[cite: 3, 10, 14].
2. **Domain Layer (`client/orchestrator.py`, `remote/stages/*`):** Racchiude la logica applicativa di orchestrazione e i singoli stage di analisi multimodale, tutti conformi al protocollo `PipelineStage`[cite: 11, 16, 17, 18, 20, 21].
3. **Data Layer (`client/pipeline_cmd.py`, `remote/storage.py`, `vlm_core/config.py`):** Definisce le strutture dati immutabili di configurazione (`@dataclass(frozen=True)`) e gestisce i percorsi di I/O e caching sul Data Lake di Google Drive[cite: 12, 15, 22].

#### Gestione Dinamica della VRAM (Prevenzione Out-Of-Memory)
Per far coesistere modelli pesanti (`WhisperX large-v2`, `Pyannote`, `Qwen3-VL-8B-Instruct` e `Qwen3-8B`) su una singola GPU Colab T4 da 15 GB di VRAM[cite: 16, 20, 22]:
* **Lazy Loading on Demand:** All'avvio di `server.py`, gli stage vengono registrati nel dizionario `STAGES` ma **nessun peso neurale viene caricato in GPU**[cite: 14, 16, 20].
* **Quantizzazione NF4 a 4-bit (`BitsAndBytesConfig`):** Sia `Qwen3-VL-8B-Instruct` sia `Qwen3-8B` vengono istanziati in 4-bit con calcolo in `float16`, riducendo la memoria richiesta dai pesi a ~5.5 GB[cite: 20, 22, 27].
* **Budget Controllato di Token Visivi:** Il modulo `vlm_core` impone limiti stringenti su `min_frames`, `max_frames`, `shortest_edge` e `longest_edge` del `video_processor` di Qwen3-VL per evitare esplosioni quadratiche nella matrice di attenzione del vision encoder[cite: 22, 27].
* **Pulizia Garantita (`try...finally`):** Ogni stage rilascia esplicitamente i riferimenti ai modelli (`del model`, `del processor`/`tokenizer`), invoca il Garbage Collector di Python (`gc.collect()`) e svuota l'allocatore CUDA (`torch.cuda.empty_cache()`) prima di restituire la risposta HTTP[cite: 16, 20, 27].

---

### 2.2 Struttura Completa del Repository

```text
open-enricher/
├── .env                                 # Segreti locali (HF_TOKEN), copiati in /content/.env durante il deploy
├── pyproject.toml                       # Configurazione di build del pacchetto e dipendenze opzionali
├── requirements.txt                     # Lista dipendenze installate nell'ambiente virtuale di Colab
├── deploy.py                            # Entrypoint locale per il packaging e il deploy remoto
├── deployment/                          # Sottosistema di Deployment (Infrastructure)
│   ├── cli_parser.py                    # Parser CLI per deploy.py (--mode, --target)
│   ├── packager.py                      # Creazione tarball open-enricher.tgz dallo stato locale
│   ├── network.py                       # Strategie di deploy: TailscaleStrategy e ColabCliStrategy
│   └── launch.sh                        # Script Bash di provisioning ed esecuzione Uvicorn su Colab
└── src/
    └── open_enricher/
        ├── client/                      # Sottosistema Client Locale
        │   ├── __main__.py              # Entrypoint CLI (python -m src.open_enricher.client)
        │   ├── cli_parser.py            # Parser e validatore degli argomenti del client
        │   ├── pipeline_cmd.py          # Dataclass immutabile PipelineConfig (Data Layer)
        │   ├── pipeline_order.py        # Ordine canonico STAGE_ORDER = ("audio", "color", "vlm", "llm", "judger")
        │   ├── orchestrator.py          # Coordinatore sequenziale dei job (PipelineOrchestrator)
        │   └── network.py               # Client HTTP (PipelineClient) e context manager tunnel SSH
        └── remote/                      # Sottosistema Server Remoto (Google Colab Worker)
            ├── server.py                # App FastAPI, startup lifespan e endpoint POST /stage/{stage_name}
            ├── storage.py               # Singleton DatabaseManager (db) per il Data Lake su Google Drive
            └── stages/                  # Pipeline Stages di Machine Learning (Domain Layer)
                ├── interface.py         # Contratto formale PipelineStage (Protocol)
                ├── audio.py             # Stage 1: Estrazione audio, trascrizione WhisperX e diarizzazione
                ├── color.py             # Stage 2: Rilevamento cambi scena, campionamento ibrido e K-Means
                ├── vlm.py               # Stage 3: Orchestratore dello stage Vision-Language
                ├── vlm_core/            # Sottopacchetto modulare per l'inferenza video Qwen3-VL
                │   ├── __init__.py      # Esportazione delle interfacce pubbliche di vlm_core
                │   ├── config.py        # Validatori di tipo/range, ModelConfig e MetadataConfig
                │   ├── video.py         # Ispezione durata (ffprobe) e taglio clip (ffmpeg)
                │   ├── model.py         # Caricamento Qwen3-VL 4-bit, limiti frame/pixel e generazione
                │   ├── vlm_params.json  # Iperparametri di inferenza VLM (fps, max_new_tokens, ecc.)
                │   └── vlm_prompts.md   # Prompt strutturato per l'estrazione di metadati visivi
                ├── llm.py               # Stage 4: Fusione metadati con Qwen3-8B 4-bit e schema Pydantic
                └── judger.py            # Stage 5: Validazione automatica qualità e controllo allucinazioni
```

---

### 2.3 Dettaglio dei Sottosistemi e Interazione tra i Moduli

#### 1. Sottosistema di Deployment (`deploy.py` & `deployment/*`)
Gestisce l'aggiornamento atomico del codice e dei segreti dal PC locale alla VM Colab[cite: 3, 4, 6]:
* **`deploy.py`** importa `parse_args` da `deployment/cli_parser.py`[cite: 1, 6], `ProjectPackager` da `deployment/packager.py`[cite: 4, 6] e le classi di rete da `deployment/network.py`[cite: 3, 6]. Legge `deployment/launch.sh` convertendo eventuali fine-riga Windows CRLF (`\r\n`) in LF (`\n`) per evitare errori di sintassi Bash su Linux[cite: 2, 6].
* **`deployment/packager.py` (`ProjectPackager`)** utilizza una directory temporanea (`tempfile.TemporaryDirectory`) per copiare `src/` (escludendo `__pycache__` e `*.pyc`) e i file di progetto (`pyproject.toml`, `requirements.txt`, `README.md`), generando l'archivio `open-enricher.tgz`[cite: 4].
* **`deployment/network.py`** definisce il protocollo `DeployStrategy` e due implementazioni concrete[cite: 3]:
  * `TailscaleStrategy`: trasferisce `open-enricher.tgz` e `.env` via `scp` ed esegue `launch.sh` passando lo script allo standard input di `ssh root@<target> bash -s`[cite: 3].
  * `ColabCliStrategy`: trasferisce gli artefatti con `colab upload` e avvia `/content/launch.sh` tramite `colab exec`[cite: 3].

---

#### 2. Sottosistema Client Locale (`src/open_enricher/client/*`)
Permette di invocare da terminale l'esecuzione remota di uno o più stage[cite: 8, 9]:
* **`client/__main__.py`** è il punto di ingresso[cite: 8]: chiama `parse_args()` da `client/cli_parser.py`[cite: 8, 9], che valida gli input rispetto a `STAGE_ORDER` definito in `client/pipeline_order.py`[cite: 9, 13] e costruisce l'oggetto immutabile `PipelineConfig` definito in `client/pipeline_cmd.py`[cite: 9, 12].
* **`client/network.py`** fornisce:
  * `setup_server_connection(mode, target, port=8000)`: un context manager (`@contextmanager`) che restituisce l'URL base del server (`http://<target>:8000` per Tailscale oppure `http://localhost:8000` dopo aver aperto il tunnel SSH in background per Colab CLI)[cite: 8, 10].
  * `PipelineClient` e `post_stage()`: eseguono la chiamata `requests.post(f"{base_url}/stage/{stage_name}", json={"video_name": video_name}, timeout=timeout)`[cite: 10].
* **`client/orchestrator.py` (`PipelineOrchestrator`)** itera sugli stage richiesti in `config.stages`, invoca `client.execute_stage()`, formatta e stampa a video la risposta JSON ricevuta dal server e interrompe immediatamente l'esecuzione se uno stage restituisce `"status": "failed"`[cite: 11].

---

#### 3. Sottosistema Server Remoto Core (`server.py`, `storage.py`, `interface.py`)
Costituisce la spina dorsale del worker in ascolto su Google Colab[cite: 14, 15, 18]:
* **`remote/storage.py` (`DatabaseManager`)**: espone l'istanza singleton globale `db`[cite: 15]. Centralizza tutti i path su Google Drive (`db.video_inputs`, `db.models_cache`, `db.get_output_dir(stage_name)`), crea le cartelle mancanti con `db.setup_database()` e reindirizza le variabili d'ambiente `HF_HOME` e `TORCH_HOME` su Drive tramite `db.configure_environment()`[cite: 15].
* **`remote/stages/interface.py` (`PipelineStage`)**: definisce tramite `typing.Protocol` l'interfaccia che ogni stage deve rispettare (attributo `name: str` e metodo `run(self, video_filename: str) -> Dict[str, Any]`)[cite: 18].
* **`remote/server.py`**:
  1. Carica `/content/.env` e chiama `db.configure_environment()` prima di importare le librerie di Machine Learning[cite: 14, 15].
  2. Istanzia e registra nel dizionario `STAGES` i 5 stage: `AudioStage`, `ColorStage`, `VLMStage`, `LLMStage` e `JudgerStage`[cite: 14].
  3. All'avvio (`lifespan`), inizializza le cartelle su Google Drive (`db.setup_database()`) ed effettua il login automatico a Hugging Face se `HF_TOKEN` è presente[cite: 14].
  4. Espone l'endpoint `POST /stage/{stage_name}`, che riceve `{"video_name": "..."}`, invoca `STAGES[stage_name].run(video_filename)` e traduce eventuali `FileNotFoundError` in errori HTTP `400` ed eccezioni interne in errori HTTP `500`[cite: 14].

---

#### 4. Sottosistema Pipeline Stages (`src/open_enricher/remote/stages/*`)

Tutti gli stage leggono l'input da `db.video_inputs` (o dai risultati degli stage precedenti in `db.get_output_dir(...)`) e salvano il proprio output su Google Drive[cite: 15, 16, 17, 20, 21]:

##### Stage 1: `audio.py` (`AudioStage`)
* **Tecnologie:** `whisperx` (`large-v2` in `float16`), modello di allineamento temporale fonetico e `DiarizationPipeline` (Pyannote)[cite: 16].
* **Funzionamento:**
  1. Carica il modello WhisperX `large-v2`, estrae la traccia audio dal video e genera la trascrizione grezza, poi scarica il modello dalla VRAM[cite: 16].
  2. Carica il modello di allineamento per la lingua rilevata, allinea i timestamp parola per parola e libera nuovamente la VRAM[cite: 16].
  3. Esegue la diarizzazione degli speaker (`DiarizationPipeline`) usando `HF_TOKEN`, associa ogni battuta al relativo parlante (`SPEAKER_00`, `SPEAKER_01`, ecc.) e scarica il modello dalla GPU[cite: 16].
* **Output:** Salva una lista di oggetti `DialogueLine` (`start_timestamp`, `end_timestamp`, `speaker`, `text`) in `Outputs/Audio-Results/<video>_audio.json`[cite: 15, 16].

##### Stage 2: `color.py` (`ColorStage`)
* **Tecnologie:** `PySceneDetect` (`ContentDetector`), `OpenCV` (`cv2`), `Pillow` e `scikit-learn` (`KMeans`)[cite: 5, 17].
* **Funzionamento:**
  1. Esegue un **Hybrid Sampling**: rileva i cambi di scena con `ContentDetector(threshold=27.0)` selezionando il frame centrale delle scene più lunghe (fino a `max_long_scenes=120`) e vi unisce una griglia temporale regolare ogni `grid_interval_sec=180` secondi (con un filtro di distanza minima di 10 secondi tra i frame e fallback sul frame centrale per video brevi)[cite: 14, 17].
  2. Per ciascun keyframe selezionato, ridimensiona l'immagine a `150x150` pixel ed esegue il clustering **K-Means** ($k=3$) nello spazio RGB per estrarre i 3 colori dominanti in formato esadecimale (`#rrggbb`)[cite: 17].
* **Output:** Salva le immagini `.jpg` dei keyframe nella cartella `Outputs/Color-Results/<video>_frames/` e il riepilogo strutturato (`scene_id`, `timestamp`, `frame_file`, `color_palette`) in `Outputs/Color-Results/<video>_color.json`[cite: 15, 17].

##### Stage 3: `vlm.py` (`VLMStage`) e il pacchetto `vlm_core/`
* **Tecnologie:** `Qwen/Qwen3-VL-8B-Instruct` quantizzato a 4-bit (`nf4`), `AutoProcessor`, `ffmpeg` e `ffprobe`[cite: 22, 27, 28].
* **Organizzazione modulare (`vlm_core/`):**
  * `vlm_core/config.py`: legge e valida rigorosamente gli iperparametri da `vlm_params.json` (`fps`, `max_new_tokens`, `do_sample`, `temperature`, `top_p`, `top_k`, `repetition_penalty`) e il prompt da `vlm_prompts.md`, costruendo le dataclass immutabili `ModelConfig` e `MetadataConfig`[cite: 22, 25, 26].
  * `vlm_core/video.py`: tramite il context manager `video_input_path`, ispeziona la durata del video con `ffprobe` (`probe_duration_seconds`) e, se `METADATA_VIDEO_SECONDS` è impostato, ricodifica al volo i primi $N$ secondi senza audio con `ffmpeg` (`clip_video_prefix`) in una cartella temporanea[cite: 22, 28].
  * `vlm_core/model.py`: carica `Qwen3-VL-8B-Instruct` in 4-bit (`load_model`), configura i limiti di memoria sul processore video (`apply_long_video_limits`), prepara i tensori multimodali (`prepare_video_inputs`), esegue `model.generate()` (`extract_metadata`) e rilascia la VRAM (`unload_model`)[cite: 24, 27].
  * `vlm.py` (`VLMStage`): coordina l'esecuzione di `vlm_core`, effettua il parsing sicuro dell'output JSON generato dal modello e persiste i risultati su Drive[cite: 21, 31].
* **Output:** Salva in `Outputs/VLMs-Results/` tre file[cite: 15, 31]:
  1. `<video>_vlm.json`: il JSON strutturato (contenente `title_guess`, `short_summary`, `tags`, `scene_timeline`, `visual_style`, `recommendation_signals`, ecc.) pronto per essere consumato dallo stage `llm`[cite: 20, 26].
  2. `<video>_vlm_raw.txt`: il testo grezzo generato dal modello[cite: 30, 31].
  3. `<video>_vlm_sidecar.json`: il log completo di tutti i parametri di configurazione utilizzati durante l'inferenza[cite: 29, 31].

##### Stage 4: `llm.py` (`LLMStage`)
* **Tecnologie:** `Qwen/Qwen3-8B` quantizzato a 4-bit (`BitsAndBytesConfig`) e `Pydantic` (`BaseModel`)[cite: 20].
* **Funzionamento:**
  1. **Data Fusion (`_gather_metadata`):** raccoglie automaticamente dal Data Lake tutti i JSON prodotti dagli stage precedenti (`<video>_audio.json`, `<video>_color.json`, `<video>_vlm.json`) e li compatta in un unico contesto multimodale[cite: 20].
  2. **Assistant Prefill Forcing (`_generate_text`):** disabilita il thinking mode (`enable_thinking=False`) e concatena il carattere `{` alla fine del prompt dell'assistente per obbligare il modello a iniziare immediatamente con un oggetto JSON valido[cite: 20].
  3. **Validazione Pydantic & Auto-Repair Loop:** valida la risposta tramite lo schema rigoroso `VideoProfile` (composto da `RecSysMetadata`, `SearchAndDiscovery`, `AudienceProfiling`, `ContentSummary`) con `VideoProfile.model_validate_json()`[cite: 20]. Se il primo tentativo fallisce la validazione, inietta l'errore esatto di Pydantic nella cronologia della chat ed esegue un secondo tentativo di auto-riparazione direttamente in VRAM prima di scaricare il modello[cite: 20].
* **Output:** Salva il profilo finale validato per il Recommender System in `Outputs/LLMs-Results/<video>_llm.json`[cite: 15, 20].

##### Stage 5: `judger.py` (`JudgerStage`)
* **Ruolo:** Modulo finale di Quality Assurance (*LLM-as-a-Judge*) predisposto per verificare la coerenza dei metadati estratti rispetto alle evidenze raccolte, rilevare eventuali allucinazioni o troncamenti e salvare il verdetto in `Outputs/Judger-Results/`[cite: 15, 19, 29, 30].