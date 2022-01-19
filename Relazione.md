# Relazione Progetto Calcolo Numerico 2021/2022


## Fase 1

#### Riportiamo i risultati ottenuti su un set di 10 immagini differenti, di cui 8 semplici figure geometriche e 2 fotografie in bianco e nero, applicando 4 diversi algoritmi di correzione.
In particolar modo, per ogni immagine, visualizziamo la stessa in 6 versioni:
- Dopo la degradazione mediante l'operatore di **BLUR** ( con sigma = 1.3, dimensione 9x9 e rumore guassiano con deviazione standard pari a 0.05 ) 
- Dopo l'applicazione di un algoritmo di correzione **NAIVE** che utilizza il metodo del gradiente coniugato implementato dalla funzione di libreria *minimize*
- Dopo l'introduzione di un **termine di regolarizzazione di Tikhonov** volto a ridurre gli effetti del rumore nella ricostruzione. 
L'algoritmo utilizzato è ancora una volta quello della funzione di libreria *minimize*. ( lambda utilizzato = 0.08) 
- Dopo l'applicazione di un algoritmo differente a quello dei due punti precedenti. In particolar modo qui utilizziamo l'**algoritmo del metodo del gradiente implementato a lezione**. ( lmbda utilizzato = 0.08 ) 
- Dopo l'applicazione del **metodo del gradiente implementato a lezione** ma utilizzando una funzione di *Variazione Totale* come termine di regolarizzazione.

Seguono i risultati ottenuti ed infine alcune osservazioni.

![K3_0 05_0 08-sample1](/tests/plots/K3_0.05_0.08-sample1.png)
![K3_0 05_0 08-sample1](/tests/plots/K3_0.05_0.08-sample2.png)
![K3_0 05_0 08-sample1](/tests/plots/K3_0.05_0.08-sample3.png)
![K3_0 05_0 08-sample1](/tests/plots/K3_0.05_0.08-sample4.png)
![K3_0 05_0 08-sample1](/tests/plots/K3_0.05_0.08-sample5.png)
![K3_0 05_0 08-sample1](/tests/plots/K3_0.05_0.08-sample6.png)
![K3_0 05_0 08-sample1](/tests/plots/K3_0.05_0.08-sample7.png)
![K3_0 05_0 08-sample1](/tests/plots/K3_0.05_0.08-sample8.png)
![K3_0 05_0 08-sample1](/tests/plots/K3_0.05_0.08-sample9.png)
![K3_0 05_0 08-sample1](/tests/plots/K3_0.05_0.08-sample10.png)


In primo luogo possiamo notare come, per tutte e 10 le immagini, la correzione naive porti ad un peggioramento sostanziale dell'immagine degradata (Blurred and Noised); 
quest'ultima, infatti, risulta più chiara e nitida di quella che dovrebbe essere una sua versione migliorata.
Va sottolineato come, aumentando il numero di iterazioni eseguite dalla funzione di correzione **NAIVE**, questa migliori se pur non sensibilmente.

Ora, analizzando esclusivamente le 8 immagini formate di figure geometriche, notiamo come ognuna delle correzioni
porti ad un graduale miglioramento dell'immagine che, nella sua ultima versione, risulta estremamente simile alla figura di partenza.

La stessa osservazione non può però essere fatta per le ultime 2 figure (quelle formate da fotografie) le quali nell'ultima versione ( TV correction )
risultano, a prima vista, peggiori di quelle regolarizzate col metodo del gradiente implementato a lezione.

## FASE 2 

