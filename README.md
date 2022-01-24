# Coin Prediction

### Predicție de preț pentru un altcoin la alegere.

##### Authors: Ciopei Dragoș, Fărcal Andrei-Ioan, Marin Mălina, Popescu Adana, Vătui Adrian

### Cerinta

Folosind datele istorice (exemplu [aici](https://www.cryptodatadownload.com/data/)) să se creeze un sistem de forecast
pentru a estima prețul în funcție de setul de date utilizat (zi, oră, minut).

- Procesarea seturilor de date și completarea lor cu Relative strength index, Moving average, si MACD.
- Realizarea pentru fiecare set de date a unui model(de ex: SVM, RNN, LTSM, etc...) și testarea parametrilor de control
  ai modelelor în vederea îmbunătățirii performanței acestora
- Includerea celor mai bune modele obținute într-o interfață grafică cu opțiuni de testare pe date reale (introduse
  manual)
  și posibilitatea alegerii manuale a intervalelor de timp pentru forecast.
- (Optional) Integrarea unui API de culegere a datelor reale (ca de ex Bitcoin Price Index API) și testarea automată a
  aplicației pe acele date

In realizarea proiectului am ales sa determinam preturile pentru 2 monede foarte populare, Ethereum (ETH) si Bitcoin (
BTC)
deoarece modelele de predictie sunt bazate pe retele neuronale (LSTM) sau masini cu vector suport (SVM) iar datele de
antrenamet erau in numar mare si respectau o distributie relativ constatnta (fara fluctuatii anormale).

### Interfata cu utilizatorul (Farcal Andrei-Ioan)

Pentru realizarea interfetei am utilizat libraria grafica [tkinter](https://docs.python.org/3/library/tkinter.html).
Utilizatorul poate interactiona cu:

- un calendar pentru a selecta ziua pana cand doreste sa vada evolutia pretului
- o sectiune unde poate selecta coin-ul pentru care se va face prezicerea
- un dropdown pentru selectarea algoritmului de predictie
- (pentru partea de bonus) un dropdown cu modul de afisare al sentimentelor

### Realizarea modelului bazat pe Support Vector Machines - SVM (Marin Malina Alexandra)

Aceasta este o tehnica moderna de clasificare supervizata, mai precis Support Vector Regression (SVR). SVM separa
clasele calculand o suprafata de decizie aflata la distanta maxima de punctele clasificate. SVR utilizeaza acelasi
principiu ca si SVM, dar pentru probleme de regresie - gasirea unei functii care aproximeaza maparea de la un input la
numere reale pe baza unui exemplu de antrenare.

- testarea parametrilor de control cu scopul imbunatatirii performantei modelului
- antrenare pentru doua monede: bitcoin si ethereum
- integrare cu interfata grafica
- grafice pentru predictii, in functie de data aleasa de utilizator
- parsare date (input de la utilizator, csv) in vederea extragerii datelor necesare in formatul potrivit cu scopul
  afisarii graficelor in interfata

### Analiza sentimentelor si testare automata (Ciopei Dragos)

Am realizat o reprezentare grafica a comentariilor de pe Twitter (cele mai recente 3000) relativ la sentimentul pe care
acestea le exprima astfel incat utilizatorul sa aiba o reprezentare grafica a intregii piete de criptomonede in functie
de subiectivitate si sentiment sau doar niste proportii numerice estimative. Rezultatele transformate in map-uri au fost
furnizate front-end-ului pentru afisare. De asemenea, m-am ocupat de planificarea validarii in viitor a datelor estimate
in trecut astfel incat sa putem capata o imagine clara aspura corectitudinii prezicerii dar si de utilitarul de
descarcare a csv-urilor.

### Preprocesarea si completarea datelor (Popescu Adana)

Am fost responsabila de preprocesarea CSV-urilor care contineau date legate de cele doua monede selectate si completarea
lor cu Relative Strength Index, Moving Average si MACD. Atat datele de input cat si valorile cerute au fost procesate cu
ajutorul librariilor [pandas](https://pandas.pydata.org/) si [pandas_ta](https://github.com/twopirllc/pandas-ta). In
plus, am oferit posibilitatea introducerii datelor prezise in dataset-ul generat si completarea lor cu RSI, pentru a
sfatui utilizatorul legat de achizitionarea monedei cerute.

### LSTM (Vatui Adrian)

M-am ocupat de modelul LSTM. LSTM, sau Long Short-Term Memory, este un tip de retea neuronala recurenta care exceleaza
in clasificarea, procesarea si prezicerea datelor dintr-o serie. Am elaborat arhitectura retelei si am testat diversi
hiperparametri pentru a imbunatati acuratetea si performanta modelului, apoi l-am integrat in interfata grafica. De
asemenea, deoarece antrenarea retelei este destul de costisitoare, am salvat modelele cele mai des folosite (<50 zile
diferenta), dar am luat in calcul si posibilitatea de a antrena 'online' unul, in caz ca nu a fost salvat.