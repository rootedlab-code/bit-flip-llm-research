# E3 — La superficie critica di un modello quantizzato

Soggetto: `Qwen/Qwen2.5-0.5B-Instruct-GGUF` @`9217f5d`, file `q4_k_m` (491.400.032 byte).
Confronto: lo stesso modello in bf16 (E1).
Dati grezzi: `results/e3-gguf-bit-census.csv`, `results/e3-gguf-scale-fragility.csv`.

## Metodo

Stesso metodo esatto di E1, applicato a una popolazione diversa: non i pesi, ma le
**scale**. Il parser si convalida chiudendo l'aritmetica sul file — intestazione
5.947.744 byte + dati = 491.400.032 byte, esatto. Le scale fp16 si estraggono
rimodellando l'intervallo di ogni tensore in (blocchi × byte per blocco) e prendendone
la colonna; il loro istogramma passa poi per lo stesso calcolo per posizione di bit.

## Primo risultato, inatteso: il file non e quasi mai K-quantizzato

| tipo | tensori | ne[0] divisibile per 256 |
|---|---|---|
| Q5_0 | 133 | no |
| Q8_0 | 13 | no |
| Q6_K | 12 | si |
| Q4_K | 12 | si |
| F32 | 121 | — |

La separazione e **perfetta, senza eccezioni**: i K-quant richiedono righe multiple di
256, il lato nascosto di Qwen2.5-0.5B e 896 = 3,5 × 256, e solo `ffn_down`
(4864 = 19 × 256) soddisfa il requisito. Tutto il resto ripiega su quantizzazioni
legacy a blocchi di 32.

Conseguenza pratica: un file etichettato `q4_k_m` **non descrive il formato dei suoi
pesi**. Chi progetta una difesa assumendo super-blocchi da 256 la sbaglia su 146
tensori su 170.

## Censo dei bit per funzione

| ruolo | bit | quota |
|---|---|---|
| quanti | 3.563.012.096 | 91,745% |
| **scale fp16** | **272.556.032** | **7,018%** |
| scale intere | 45.760.512 | 1,178% |
| float (norme, bias) | 2.289.664 | 0,059% |

## Le scale hanno la stessa debolezza universale dei pesi

Nelle scale di blocchi da 32 **e** in quelle da 256, il bit 14 vale zero nel
**100,00%** dei casi — stessa proprieta trovata in E1 sui pesi bf16, stessa
conseguenza: chi colpisce quel bit non ha bisogno di sapere che valore stia colpendo.

In fp16 il bit alto dell'esponente moltiplica per 2¹⁶ = 65.536 (contro 2¹²⁸ in bf16,
vedi `tests/test_codec.py`). Esattamente **un bit su sedici** di ogni scala e
catastrofico: 6,2500%, cioe uno per scala, sempre lo stesso.

## Il confronto che risponde alla domanda

| formato | bit totali | bit catastrofici | quota | raggio | pesi persi per flip casuale |
|---|---|---|---|---|---|
| bf16 safetensors | 7.904.524.288 | 494.787.536 | 6,2595% | 1 peso | 0,062595 |
| gguf q4_k_m | 3.883.618.304 | 17.034.752 | 0,4386% | 40,1 pesi | **0,175711** |

La quantizzazione riduce di **14 volte** la quota di bit catastrofici — e aumenta di
**40 volte** il raggio di ciascuno, perche una scala governa il proprio blocco intero.
Il saldo:

> a parita di guasto casuale, il file quantizzato perde **2,807 volte piu pesi** di
> quello in bf16.

La quantizzazione non protegge: **concentra**. Sposta il rischio da una popolazione
grande di bit poco pericolosi a una popolazione piccola di bit molto pericolosi, e la
concentrazione peggiora il valore atteso invece di migliorarlo.

## Limiti di questa conclusione — dichiarati

«Pesi persi» non e «danno al modello», e la differenza conta in tre modi:

1. **Il moltiplicatore non e lo stesso.** Un peso bf16 colpito viene moltiplicato per
   2¹²⁸, uno quantizzato per 2¹⁶: dodici ordini di grandezza di differenza nella
   gravita per singolo peso. Il conteggio qui sopra li tratta come equivalenti perche
   entrambi superano la soglia — una scelta discutibile, dichiarata.
2. **Il danno quantizzato e correlato.** I 32 pesi di un blocco sono contigui nella
   stessa riga; 40 pesi sparsi in tensori diversi sono un'altra cosa. Quale delle due
   distruzioni pesi di piu sull'uscita non lo dice questo esperimento.
3. **Nessuno dei due numeri e ancora una misura di degrado.** Serve E2: perplexity e
   accuratezza su un modello davvero colpito.

Fino a quel punto, il risultato di E3 va letto per quello che e — una misura di
**superficie**, non di conseguenza.
