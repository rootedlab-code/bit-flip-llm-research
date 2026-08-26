# E1 — Gerarchia di fragilita dei bit

Soggetti: `Qwen/Qwen2.5-0.5B-Instruct` @`7ae5576` e
`huihui-ai/Qwen2.5-0.5B-Instruct-abliterated-v3` @`3dee99d`.
494.032.768 pesi bf16 in 290 tensori, tutti in BF16.
Dati grezzi: `results/e1-bit-hierarchy-{base,abliterated}.csv`.

## Metodo

Le statistiche sono **esatte, non campionate**. L'esito di un flip dipende solo dal
pattern di 16 bit, non da quale peso lo porti: l'istogramma dei 65.536 pattern
riassume quindi senza perdita l'intero modello, e ogni frazione qui sotto e un
conteggio, non una stima. La copertura si verifica da se — la somma dell'istogramma
deve coincidere con il numero di parametri dichiarato dall'intestazione, e coincide.

I file di modello sono stati letti sotto il contesto `immutable`: il loro SHA-256 e
stato riverificato a fine esperimento ed e invariato.

## Il risultato

| bit | campo | vale 0 in | \|Δw\| mediano | \|Δw\| max | catastrofici |
|---|---|---|---|---|---|
| 0-6 | mantissa | ~50-58% | 6,1e-05 … 3,9e-03 | ≤ 64 | **0%** |
| 7-10 | esponente basso | 36-66% | 9,3e-03 … 1,7e-01 | ≤ 5,5e+04 | 0% |
| 11 | esponente | 0,15% | 1,1e-02 | 1,4e+07 | 0,1508% |
| 12-13 | esponente | 0,00% | 1,1e-02 | 3,9e+21 | 0,0020% |
| **14** | **esponente alto** | **100,00%** | **3,9e+36** | 3,4e+38 | **99,998%** |
| 15 | segno | 50,02% | 2,3e-02 | 428 | 0% |

## Perche il bit 14 e la superficie d'attacco universale

Non conta solo che il suo moltiplicatore sia il piu grande (2¹²⁸, dimostrato in
`tests/test_codec.py`). Conta che il suo **valore sia prevedibile**: il 99,9926% dei
pesi ha |w| < 1, l'esponente mediano e 120 contro un bias di 127, e il bit 14 vale
zero nel **100,00%** dei pesi di entrambi i modelli.

Ne segue il fatto che rende praticabili gli attacchi della letteratura:

> per amplificare un peso, l'attaccante non ha bisogno di sapere **quale** peso sta
> colpendo. Deve solo colpire il bit 14 di un peso qualsiasi, e il flip sara 0→1.

Il contraltare e altrettanto netto: i bit 11-13 valgono 1 quasi ovunque, quindi
ribaltarli **divide** invece di moltiplicare — sono innocui per costruzione, non per
fortuna. Ogni peso ha esattamente un bit universalmente amplificante.

## La cifra per E4

**6,2595%** dei bit del file sono catastrofici (494.787.536 su 7.904.524.288), cioe
**uno ogni 15,98**. Un guasto che cada a caso in questo file ha dunque circa una
probabilita su sedici di essere distruttivo — ed e questa la frazione che, incrociata
con un FIT rate, dara il tempo medio prima di un danno naturale.

## L'ablazione non cambia la fragilita

Il modello abliterato ha lo stesso profilo, cifra per cifra fino alla quarta decimale:
gli scarti sulla frazione di bit a zero restano sotto lo 0,003% su ogni posizione, e i
bit catastrofici passano da 494.787.536 a 494.787.360 — **176 bit di differenza su
7,9 miliardi**. Rimuovere l'allineamento non irrobustisce ne indebolisce il modello
davanti a un guasto: sposta il comportamento, non la geometria dei pesi.

## Confini

Questa e la fragilita *aritmetica* di un peso isolato. Non dice ancora nulla su
quanto il **modello** degradi: un peso portato a 6,8e+36 in un tensore poco usato puo
non cambiare una virgola dell'output. Quella e la domanda di E2.
