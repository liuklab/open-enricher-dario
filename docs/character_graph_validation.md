# Validazione del ramo 12-combined su Slurm

La base è `debug` con i 16 commit intermedi di `cursor/12-combined-himym-aa84`. La clip storica di HIMYM parte a 00:00:20 e dura 30 secondi. Il job 122717 ha scritto un profilo per la clip. Il job 122809 è stato cancellato dopo circa 6 ore e 25 minuti. Su HIMYM ha prodotto 6.750 nodi e nessun arco `speaks`, poi il generatore del profilo ha esaurito la memoria GPU. Su Silicon Valley si è fermato durante il grafo. The Kingdom era ancora in attesa. I ritocchi successivi riducono la frequenza di campionamento, raggruppano i volti, limitano il riepilogo e leggono i timestamp BaSSL in formato orario. Il job BaSSL 122853 ha lasciato un artefatto separato da verificare.

## Sequenza degli esperimenti

| Esperimento | Input | Criteri di riuscita |
| --- | --- | --- |
| InsightFace | Clip HIMYM, pochi frame | CUDA realmente attivo, embedding non vuoti, tempi e numero di campioni registrati |
| Audio e TalkNet | Clip HIMYM con dialogo | Trascrizione con timestamp e speaker; forward reale con finestre valide e almeno due finestre affidabili distinte per uno speaker visibile annotato |
| Grafo personaggi | Stessa clip, output nuovi | Volti simultanei con ID distinti, ricomparse coerenti, archi `speaks` collegati a nodi e timestamp verificabili, riepilogo limitato. Confrontare con una verifica manuale della scena |
| BaSSL e timeline | Prima clip, poi artefatto 122853 o un episodio intero | Intervalli validi entro la durata, segmentazione reale su episodio, copertura e ordine temporale |
| EVA/Q-Former e memoria visiva | Clip con più scene | Token reali 32×768 per frame, memoria popolata e retrieval che usa solo timestamp precedenti |
| Composizione e profilo | Clip HIMYM, poi brevi estratti degli altri due titoli | Profilo conforme allo schema con metadati, evidenze e timestamp coerenti. Controllare audio, dialogo, colore, allocazione dei frame, VLM, LLM e metriche. Verificare gli stati degli stage anche con exit code Slurm 0 |
| Pipeline completa | Tutti i video in input con 12-combined e tutti gli stage attivi | Un profilo valido per ogni titolo, manifest coerente con il file sorgente, nessun fallback inatteso, tempi e picchi di memoria registrati |

## Fix da validare e diagnosi successive

- Il grafo conserva la mappa dagli ID originali delle rilevazioni agli ID raggruppati. I test di regressione coprono anche uno speaker con tre identità concorrenti.
- Controllare la copertura di TalkNet. Il percorso attuale limita il forward a quattro finestre per titolo. Misurare quante finestre hanno dialogo diarizzato e verificare gli archi nella clip. Se la copertura è insufficiente, distribuire un budget esplicito di finestre sui tratti con dialogo e contare una sola volta ciascuna finestra per coppia speaker–personaggio.
- Misurare separatamente rilevazione, raggruppamento, abbinamento, TalkNet e generazione del profilo. Se il grafo cresce troppo, misurare il costo delle comparazioni con gli esemplari. Se il profilo esaurisce ancora la memoria, misurare i token in ingresso e il picco GPU prima di modificare il budget.
- Verificare cache e ripresa con un output isolato per clip e un dataset separato per i titoli interi. Controllare che un archivio con lo stesso nome e una durata diversa non faccia saltare un run.

## Operatività

Spegnere e verificare sul nodo di login la sessione tmux `oe-slurm-watcher` prima della prima sottomissione. Per ogni esperimento registrare commit, input, job ID, percorso output, stati degli stage e risultati. Controllare la coda ogni 10 minuti. Dopo l'uscita dalla coda verificare stato ed exit code con `sacct` e ispezionare gli artefatti. Fermare i tentativi su ogni problema dopo tre prove fallite e lasciare il run completo sospeso finché i controlli sulle clip restano aperti.
