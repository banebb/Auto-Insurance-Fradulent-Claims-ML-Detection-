# Detekcija lažnih zahteva za osiguranje automobila primenom metoda mašinskog učenja

**Bane [Prezime], [Indeks]**
**[Ime kolege], [Indeks]**

---

## I. UVOD

Osiguranje automobila predstavlja jednu od najzastupljenijih vrsta osiguranja, a lažni zahtevi za odštetu (engl. *fraudulent claims*) nanose značajnu finansijsku štetu osiguravajućim kompanijama i indirektno povećavaju premije svim korisnicima. Prema procenama industrije, lažni zahtevi čine 5–10% svih prijavljenih šteta, što rezultira godišnjim gubicima od više milijardi dolara na globalnom nivou [1]. Pravovremeno prepoznavanje potencijalno lažnih zahteva ima ključnu ulogu u smanjenju finansijskih gubitaka i efikasnijem procesuiranju legitimnih zahteva.

Kreiranje klasifikacionog modela omogućava automatsko predviđanje da li je prijavljeni zahtev za osiguranje lažan ili legitiman, na osnovu karakteristika polise, incidenta i korisnika osiguranja. U ovom radu primenjen je sistematičan pristup koji obuhvata pripremu podataka, selekciju obeležja metodama teorije informacija, te obuku i evaluaciju više klasifikatora.

---

## II. BAZA PODATAKA

### A. Struktura baze podataka

Korišćena baza podataka sadrži informacije o zahtevima za osiguranje automobila i obuhvata 1000 uzoraka sa 40 obeležja [2]. Obeležja se mogu grupisati u četiri kategorije:

- **Karakteristike polise:** trajanje korisničkog odnosa, starost korisnika, država polise, nivo pokrića, odbitna franšiza, godišnja premija, limit kišobran polise
- **Karakteristike korisnika:** pol, nivo obrazovanja, zanimanje, hobiji, bračni status
- **Karakteristike incidenta:** datum, tip incidenta, tip sudara, ozbiljnost, kontaktirane službe, država i grad incidenta, sat, broj vozila, telesne povrede, svedoci, policijski izveštaj
- **Karakteristike zahteva:** ukupan iznos zahteva, zahtev za povrede, zahtev za imovinu, zahtev za vozilo

Ciljna promenljiva `fraud_reported` ima dve klase:
- **N** — legitiman zahtev: 753 uzorka (75,3%)
- **Y** — lažan zahtev: 247 uzoraka (24,7%)

Odnos klasa je približno 3:1, što ukazuje na nebalansiran skup podataka (Sl. 1.).

*Sl. 1. Raspodela uzoraka po klasama.*

### B. Obrada podataka

Analizom baze podataka utvrđeno je prisustvo nedostajućih vrednosti u četiri obeležja, kao i postojanje obeležja bez prediktivne vrednosti.

**Korak 1: Uklanjanje obeležja.** Uklonjeno je 6 obeležja koja ne nose korisnu informaciju za klasifikaciju:
- `_c39` — 100% nedostajućih vrednosti (prazan stupac)
- `policy_number` — jedinstveni identifikator polise
- `policy_bind_date`, `incident_date` — datumski stringovi koji nisu transformisani
- `insured_zip` — prevelik broj jedinstvenih vrednosti, ponaša se kao identifikator
- `incident_location` — slobodan tekst, previsoka kardinalnost

**Korak 2: Dopuna nedostajućih vrednosti.** Preostala četiri obeležja sa nedostajućim vrednostima obrađena su na sledeći način (Tabela 1):

*Tabela 1: Obrada nedostajućih vrednosti.*

| Obeležje | Broj nedostajućih | % | Strategija dopune |
|----------|:-:|:-:|---|
| `authorities_contacted` | 91 | 9,1% | Zamena sa "None" (nijedna služba kontaktirana) |
| `collision_type` | 178 | 17,8% | Zamena '?' sa "No collision" (krađa, oštećenje u mestu) |
| `police_report_available` | 343 | 34,3% | Zamena '?' sa "NO" (pretpostavka: izveštaj nije dostupan) |
| `property_damage` | 360 | 36,0% | Zamena '?' sa "NO" (pretpostavka: nema oštećenja imovine) |

Nakon obrade, baza sadrži 1000 uzoraka i 34 stupca (33 obeležja i ciljna promenljiva), od čega je 16 numeričkih i 17 kategoričkih obeležja.

### C. Analiza obeležja

Analiza individualnih obeležja otkrila je da obeležje `incident_severity` (ozbiljnost incidenta) ima najjaču pojedinačnu vezu sa ciljnom promenljivom. Lažni zahtevi su disproporcionalno zastupljeniji kod zahteva sa oznakom "Major Damage" (velika šteta). Takođe, određeni hobiji korisnika (`insured_hobbies`) pokazuju iznenađujuće jaku korelaciju sa lažnim zahtevima, što može ukazivati na specifične obrasce ponašanja.

Ukupan iznos zahteva (`total_claim_amount`) je pozitivno koreliran sa lažnim zahtevima, ali je redundantan sa svojim komponentama (`injury_claim`, `property_claim`, `vehicle_claim`), što je značajno za izbor metode selekcije obeležja.

---

## III. SELEKCIJA OBELEŽJA

Za smanjenje dimenzionalnosti i identifikaciju najrelevantnijih obeležja primenjene su dve metode zasnovane na teoriji informacija: uzajamna informacija (engl. *Mutual Information*, MI) i MRMR metoda (engl. *Minimum Redundancy Maximum Relevance*).

### A. Uzajamna informacija (MI)

Uzajamna informacija kvantifikuje statističku zavisnost između obeležja i ciljne promenljive [3]. Za dve slučajne promenljive X i Y definisana je kao:

```
MI(X; Y) = H(Y) − H(Y|X)
```

gde je H(Y) entropija ciljne promenljive, a H(Y|X) uslovna entropija nakon opservacije obeležja X. Za razliku od korelacije, MI detektuje bilo koji oblik zavisnosti — linearni, nelinearni i kategorički.

Za procenu MI korišćen je KNN estimator (k=5) za numerička obeležja [4], dok je za kategorička obeležja korišćeno direktno brojanje koopkurencija. Top 5 obeležja prema MI rangiranju prikazano je u Tabeli 2.

*Tabela 2: Prvih 5 obeležja po MI skoru.*

| Rang | Obeležje | MI skor |
|:----:|----------|:-------:|
| 1 | `incident_severity` | 0,1237 |
| 2 | `insured_hobbies` | 0,0724 |
| 3 | `total_claim_amount` | 0,0296 |
| 4 | `auto_model` | 0,0245 |
| 5 | `collision_type` | 0,0182 |

**Ograničenje MI:** MI evaluira svako obeležje nezavisno, ne uzimajući u obzir redundantnost između obeležja. Dva visoko korelirana obeležja mogu oba dobiti visok MI skor, iako zajedno ne nose značajno više informacija nego jedno od njih.

### B. MRMR metoda

MRMR metoda otklanja ograničenje MI-ja tako što u svakom koraku bira obeležje koje istovremeno maksimizira relevantnost prema ciljnoj promenljivoj i minimizuje redundantnost sa već odabranim obeležjima [5]:

```
max_f [ MI(f; target) − (1/|S|) · Σ MI(f; s) za s ∈ S ]
```

gde je S skup već odabranih obeležja. Na ovaj način, MRMR je u stanju da promoviše obeležja sa niskim individualnim MI skorom ali visokim doprinosom u kombinaciji sa ostalim obeležjima.

Primenom MRMR metode odabrano je **20 obeležja** od polaznih 33. Poređenje MI i MRMR rangiranja pokazuje da su 13 obeležja zajednička u oba top-20 rang lista. Značajan primer je obeležje `witnesses` (svedoci), koje ima MI skor blizu nule (individualno nekorisno), ali ga MRMR odabira jer nosi jedinstvenu informaciju koja nije sadržana u ostalim obeležjima.

Selektovana obeležja sačuvana su u fajlu `data/processed/selected_features.csv` (1000 uzoraka, 20 obeležja + ciljna promenljiva).

---

## IV. PRIPREMA PODATAKA ZA KLASIFIKACIJU

Priprema podataka za obuku klasifikatora obuhvata sledeće korake:

1. **Kodiranje ciljne promenljive:** Vrednost 'Y' (lažan zahtev) mapirana je na 1, a 'N' (legitiman) na 0.

2. **Kodiranje kategoričkih obeležja:** Za pretvaranje kategoričkih obeležja u numerički format korišćen je LabelEncoder. Iako LabelEncoder uvodi veštačko uređenje, odabran je umesto one-hot kodiranja jer bi one-hot značajno proširio prostor obeležja (neki atributi poput `incident_city` imaju veliki broj kategorija), što je nepoželjno pri malom broju uzoraka (1000).

3. **Podela na skupove:** Podaci su podeljeni na trening (70%, 700 uzoraka), validacioni (15%, 150 uzoraka) i test skup (15%, 150 uzoraka). Korišćena je stratifikovana podela kako bi se očuvao odnos klasa (~75%/25%) u svakom skupu.

4. **Standardizacija:** Primenjena je StandardScaler normalizacija (srednja vrednost 0, standardna devijacija 1). Skaler je prilagođen isključivo na trening skupu, a zatim primenjen na validacioni i test skup, čime je sprečeno curenje podataka (engl. *data leakage*).

**Zašto tri skupa umesto dva?** Ako se hiperparametri podešavaju na test skupu, indirektno se model prilagođava tom skupu i prijavljene metrike postaju optimistički pristrasne. Validacioni skup apsorbuje ovu pristrasnost, čuvajući test skup kao istinski neviđene podatke za finalnu evaluaciju.

---

## V. KLASIFIKATORI

Zadatak je klasifikacija uzoraka u dve klase: lažan i legitiman zahtev. S obzirom na nebalansiranost klasa (75%/25%), metrike tačnosti (accuracy) nisu dovoljne za evaluaciju. Kao primarna metrika korišćen je F1 skor, koji balansira preciznost i odziv. Dodatno su praćeni preciznost, odziv (recall) i ROC AUC.

Za sve klasifikatore korišćena je unakrsna validacija sa 5 segmenata (engl. *5-fold stratified cross-validation*) na trening skupu za procenu stabilnosti, dok je validacioni skup korišćen za izbor hiperparametara. Test skup je evaluiran isključivo jednom, za finalno izabranu konfiguraciju.

### A. Logistička regresija

Logistička regresija (LOG) je generalizovani linearni model koji predviđa verovatnoću pripadnosti klasi pomoću sigmoidne funkcije [6]:

```
P(lažan | X) = sigmoid(w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ)
```

gde je sigmoid(z) = 1 / (1 + e^(−z)), a w₀...wₙ su koeficijenti koji se uče minimizacijom unakrsne entropije sa L2 regularizacijom:

```
Gubitak = unakrsna_entropija + (1/C) · Σ wᵢ²
```

Parametar C kontroliše jačinu regularizacije — manje vrednosti C znače jaču regularizaciju i jednostavniji model.

**Zašto logistička regresija?** Odabrana je kao prvi klasifikator iz više razloga: (1) koeficijenti direktno pokazuju koja obeležja utiču na predikciju prevare, (2) daje kalibrisane verovatnoće koje omogućavaju podešavanje praga klasifikacije, i (3) brzo se obučava, što omogućava iterativno eksperimentisanje.

#### Eksperiment 1: Bazni model (C=1,0, balansirane težine)

Konfiguracija: C=1,0, `class_weight='balanced'`, solver L-BFGS, max_iter=1000.

Parametar `class_weight='balanced'` automatski dodeljuje težine klasama inverzno proporcionalno njihovoj zastupljenosti: težina za klasu lažnih zahteva je ~2,02, a za legitimne ~0,66. Bez balansiranih težina, model bi mogao postići 75% tačnosti predviđajući sve zahteve kao legitimne — potpuno beskorisno za detekciju prevare.

*Tabela 3: Rezultati baznog modela na trening i validacionom skupu.*

| Skup | Tačnost | Preciznost | Odziv | F1 | ROC AUC |
|------|:-------:|:----------:|:-----:|:--:|:-------:|
| Trening | 0,7200 | 0,4591 | 0,7457 | 0,5683 | 0,7830 |
| Validacioni | 0,7133 | 0,4500 | 0,7297 | 0,5567 | 0,7828 |

Metrike na trening i validacionom skupu su bliske, što ukazuje da model ne pokazuje preobučenost. Odziv od 73% znači da model detektuje većinu lažnih zahteva, ali niska preciznost od 45% znači da je više od polovine prijavljenih slučajeva lažni alarm.

#### Eksperiment 2: Stabilnost unakrsnom validacijom

Primenjena je 5-fold stratifikovana unakrsna validacija na trening skupu (Tabela 4).

*Tabela 4: Rezultati unakrsne validacije baznog modela.*

| Metrika | Srednja vrednost | Std. devijacija |
|---------|:----------------:|:---------------:|
| F1 | 0,528 | 0,051 |
| Preciznost | 0,428 | 0,058 |
| Odziv | 0,699 | 0,063 |
| ROC AUC | 0,760 | 0,038 |

Umerene standardne devijacije (0,04–0,06) ukazuju na razumnu stabilnost modela.

#### Eksperiment 3: Podešavanje parametra C

Testirane su vrednosti C ∈ {0,001; 0,01; 0,1; 0,5; 1,0; 5,0; 10,0; 50,0; 100,0}. Najbolji rezultat na validacionom skupu postignut je za **C=0,01** sa F1=0,5714 (u poređenju sa F1=0,5567 za C=1,0). Jača regularizacija pomaže jer sa samo 700 uzoraka za obuku ograničava prilagođavanje šumu u podacima.

#### Eksperiment 4: Potvrda unakrsnom validacijom

Za C=0,01, unakrsna validacija daje F1 = 0,536 (±0,021) — niža standardna devijacija u poređenju sa C=1,0 (±0,051) ukazuje na stabilniji model.

#### Eksperiment 5: Podešavanje praga klasifikacije

Podrazumevani prag klasifikacije je 0,5, ali za nebalansirane podatke to ne mora biti optimalno. Testiran je opseg pragova od 0,1 do 0,9. Rezultat: optimalan prag po F1 metrici je **0,50** — podrazumevana vrednost je već bila optimalna.

#### Eksperiment 6: Poređenje sa i bez balansiranih težina

Bez `class_weight='balanced'`, model postiže višu tačnost (76,7%) ali detektuje samo 5,4% lažnih zahteva (odziv=0,0541, F1=0,1026). Ovo potvrđuje da su balansirane težine neophodne za ovaj nebalansirani skup podataka.

#### Eksperiment 7: L1 i Elastic Net regularizacija (sva obeležja)

Dodatno je istražen uticaj tipa regularizacije na modelu sa svih 33 obeležja:

- **L1 (Lasso):** Potiskuje koeficijente nerelevantnih obeležja na tačno nulu, efektivno vršeći selekciju obeležja tokom obuke. Korišćen je SAGA solver.
- **Elastic Net (L1+L2):** Kombinuje L1 sposobnost eliminacije obeležja sa L2 stabilnošću pri koreliranim obeležjima. Korišćen je l1_ratio=0,5.

#### Eksperiment 8: Poređenje modela sa 20 i 33 obeležja

Poređeni su najbolji modeli sa MRMR selektovanih 20 obeležja i sa svih 33 obeležja. Poređenje je izvršeno isključivo na validacionom skupu kako bi se izbegla pristrasnost test skupa. Oba modela koriste identičnu podelu podataka (isti `random_state=42` i proporcije 70/15/15), čime je obezbeđeno da se isti uzorci nalaze u istim skupovima.

*Tabela 5: Poređenje sumarnih rezultata logističke regresije.*

| Konfiguracija | Val F1 | Val Preciznost | Val Odziv | Val ROC AUC |
|---------------|:------:|:--------------:|:---------:|:-----------:|
| Bazni (C=1,0, 20 obeležja) | 0,5567 | 0,4500 | 0,7297 | 0,7828 |
| Optimalan (C=0,01, 20 obeležja) | 0,5714 | 0,4590 | 0,7568 | 0,7797 |
| Bez balansiranih težina | 0,1026 | 1,0000 | 0,0541 | 0,7778 |

Konačan izbor modela logističke regresije: **C=0,01, class_weight='balanced', prag=0,50** sa 20 MRMR obeležja.

### B. [Naziv drugog klasifikatora]

*[Ovaj deo popunjava kolega.]*

### C. [Naziv trećeg klasifikatora]

*[Ovaj deo popunjava kolega.]*

---

## VI. POREĐENJE KLASIFIKATORA

*Tabela 6: Prikaz mera uspešnosti klasifikatora.*

| Klasifikator | Preciznost | Tačnost | Odziv | F1 |
|---|:---:|:---:|:---:|:---:|
| **LOG** | — | — | — | — |
| [Klasifikator 2] | — | — | — | — |
| [Klasifikator 3] | — | — | — | — |

*[Popuniti nakon finalne evaluacije svih klasifikatora na test skupu. Svaki klasifikator se evaluira na test skupu tačno jednom, sa konfiguracijom izabranom na osnovu validacionih metrika.]*

*Sl. X. Matrica konfuzije optimalnog modela.*

---

## VII. ZAKLJUČAK

*[Popuniti nakon završetka svih eksperimenata.]*

---

## VIII. REFERENCE

[1] Coalition Against Insurance Fraud, "By the Numbers: Fraud Statistics," https://insurancefraud.org/fraud-stats/ (2023).

[2] Kaggle, "Auto Insurance Claims Data," https://www.kaggle.com/datasets/buntyshah/auto-insurance-claims-data (2019).

[3] C. E. Shannon, "A Mathematical Theory of Communication," *Bell System Technical Journal*, vol. 27, pp. 379–423, 1948.

[4] A. Kraskov, H. Stögbauer, and P. Grassberger, "Estimating mutual information," *Physical Review E*, vol. 69, no. 6, 2004.

[5] H. Peng, F. Long, and C. Ding, "Feature selection based on mutual information: criteria of max-dependency, max-relevance, and min-redundancy," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 27, no. 8, pp. 1226–1238, 2005.

[6] Praktikum iz mašinskog učenja, Tijana Nosek, Branko Brkljač, Danica Despotović, Milan Sečujski, Tatjana Lončar-Turukalo; Univerzitet u Novom Sadu, 2020.
