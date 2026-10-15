# Esperimento 05 — finestre da 5 minuti

Branch: `exp/05-segments-5min`  
Video: `Stranger_Things.S01E03.mkv`  
`Qwen3-VL-8B` di nuovo, un passaggio per ciascuna di 11 finestre da 5 minuti (8 frame, 450 token). L’LLM e il judger sono quelli dell’esperimento 03. Audio e colori riusati.

## Esito

Tutte e 11 le finestre hanno prodotto JSON. Overall **0,774**, **usable=false**. Soglia di grounding 0,40: qui il grounding è **0,205**.

| Dimensione | Esp. 03 (file intero) | Esp. 04 (Qwen2.5) | Esp. 05 (finestre) |
| :--- | ---: | ---: | ---: |
| conformance | 1.000 | 1.000 | 1.000 |
| grounding | 0.864 | 0.788 | 0.205 |
| overall | 0.932 | 0.914 | 0.774 |
| usable | sì | sì | no |

## I metadati visivi sono i migliori finora

La timeline distingue le finestre: camera da letto, ragazzi sul divano, stazione di polizia, ragazza con cuffia EEG, corridoio di scuola, bosco. In `on_screen_text` compaiono `CHAPTER THREE` e `HOLLY. JOLLY`, il titolo vero dell’episodio, che il passaggio unico sul file intero non leggeva. La luce è descritta (luci fatate, lampada verde, ombre) e non è più una copia degli hex.

Il profilo LLM butta via quei dettagli. Il hook dice che una donna cerca il fratello scomparso. I tag includono «Supervillain mystery», che nella timeline non c’è. Genere `Mystery`, età `Young Adults`, stile di nuovo «deep blues and blacks».

## Il judger

Il no su `usable` è coerente con la sua regola: il profilo non riusa le parole dell’evidenza (`Mystery` e `Psychological` hanno grounding 0). Non sta misurando se la timeline è buona. Sta misurando se il riassunto finale le somiglia. Qui il riassunto è più povero e in un punto inventato, quindi il punteggio basso è sensato sul profilo e cieco sulla qualità delle finestre. La specificity sulle etichette di una sola parola resta il pezzo ottuso, ma non è ciò che rende il profilo inutilizzabile.

## Prossimo giro

Non rifare le 11 finestre. Sul branch `exp/06-cite-timeline` si rilancia solo l’LLM sullo stesso JSON segmentato, obbligandolo a riusare oggetti, luoghi e il titolo letto a schermo, e a non introdurre un tipo di antagonista assente dalla timeline.
