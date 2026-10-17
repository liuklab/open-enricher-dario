# Esperimenti del round exp2

La pipeline produce un profilo video (`VideoProfile`) per un recommender: trascrizione, colori di scena, descrizioni visive a finestre, poi un profilo testuale. Questo round confronta **come** si taglia l’episodio e **dove** finiscono dialoghi, colori e contesto, per capire quale combinazione rende il profilo più fedele all’episodio e più utile in ricerca.

Ogni variante gira in parallelo su due macchine, con lo stesso commit:


| Macchina | Video                        |
| -------- | ---------------------------- |
| Colab    | `Stranger_Things.S01E03.mkv` |
| Kaggle   | `Suits.S01E02.mkv`           |


I risultati finiscono in `runs/exp2/`. Un agente di benchmark, che vede solo il profilo e una sinossi umana, assegna un punteggio da 1 a 5 su completezza, specificità, fedeltà, coerenza e utilità per il recommender. La fedeltà pesa di più. Lo stage `judger` deterministico non entra in questo round.

Le varianti 01–07 sono fisse. Le 08–10 sono ipotesi: l’orchestratore le rivede dopo aver letto i punteggi di entrambi gli episodi.

## Cosa si varia

- **Segmentazione.** `interval` taglia il video a finestre fisse di 5 minuti. `scenes` unisce i cambi di scena in blocchi tra 45 e 150 secondi, al massimo 24.
- **Instradamento di dialoghi e colori.** `llm` li passa al modello che scrive il profilo. `vlm` li mette nel prompt visivo, così la descrizione della finestra è già ancorata a ciò che si dice e ai colori.
- **Memoria del VLM.** Ogni finestra riceve un riassunto compresso delle finestre precedenti.
- **Riassunto dei dialoghi verso il VLM.** Un modello locale condensa la trascrizione e quel testo entra nel prompt visivo.
- **Trama progressiva.** Una trama corta si aggiorna finestra per finestra e viene ripassata al modello.
- **Controllo delle affermazioni.** Dopo il profilo, un secondo passaggio toglie o ammorbidisce le frasi che timeline e riassunto dei dialoghi non sostengono.



## Le dieci prove


| Id                    | Ipotesi                                                                                        |
| --------------------- | ---------------------------------------------------------------------------------------------- |
| `01-interval-llm`     | Baseline: finestre da 5 minuti; dialoghi e colori arrivano solo all’LLM.                       |
| `02-interval-vlm`     | Stesse finestre, ma dialoghi e colori entrano nel VLM.                                         |
| `03-vlm-memory`       | Come 01, con la memoria delle finestre precedenti nel VLM.                                     |
| `04-scenes-llm`       | Taglio per scene, evidenze all’LLM.                                                            |
| `05-scenes-vlm`       | Taglio per scene, evidenze nel VLM.                                                            |
| `06-scenes-memory`    | Taglio per scene, evidenze all’LLM, più memoria del VLM.                                       |
| `07-dialogue-summary` | Finestre da 5 minuti ed evidenze all’LLM, con il riassunto dei dialoghi nel prompt del VLM.    |
| `08-rolling-plot`     | Finestre da 5 minuti, evidenze nel VLM, più trama progressiva. Da rivedere dopo 01–07.         |
| `09-best-combo`       | Combinazione dei knobs che hanno dato la fedeltà più alta sui due episodi. Segnaposto.         |
| `10-claim-check`      | Parte dal profilo migliore e aggiunge il passaggio che elimina le affermazioni non supportate. |


01–03 isolano l’instradamento e la memoria a parità di taglio temporale. 04–06 ripetono il confronto sul taglio per scene. 07 verifica se un riassunto dei dialoghi aiuta il VLM senza spostare le evidenze grezze. 08–10 servono a combinare ciò che ha funzionato e a misurare se un controllo finale delle affermazioni alza la fedeltà.