# ML Tribology Pipeline

Ez a projekt egy végpontok közötti (end-to-end) Machine Learning folyamat, amely tribométeres mérési adatok elemzésére, prediktív modellezésére és a kenési állapotok (súrlódási tényező - COF és FAI) optimalizálására szolgál.

## Főbb funkciók
* **Adatfeldolgozás:** Zajszűrés (rolling mean), downsampling, és anomáliák (outlierek) automatikus eltávolítása.
* **Feature Engineering:** Fizikai paraméterek (pl. Hertz-féle feszültség) és interakciós jellemzők generálása, VIF (Variance Inflation Factor) alapú multikollinearitás szűrés.
* **Modell tanítás és optimalizálás:** 7 különböző algoritmus (XGBoost, LightGBM, CatBoost, Random Forest, MLP, KNN, Ridge) hiperparaméter-hangolása Optuna (Bayesian Optimization) segítségével.
* **Ensemble:** A legjobb 3 modell automatikus kiválasztása és súlyozott kombinálása.
* **Magyarázhatóság (XAI):** SHAP (SHapley Additive exPlanations) alapú ábrák a funkciók fontosságának és hatásának megértéséhez.
* **Optimum keresés és DoE:** Kompromisszumos (Pareto) optimumok meghatározása az észteresített és alapolajokra, valamint új mérési pontok javaslata (Active Learning / Design of Experiments).
* **Automatikus Riportálás:** Átfogó HTML és Excel riport generálása a futás legvégén a legfontosabb metrikákkal és diagramokkal.

## Könyvtárstruktúra

A projekt futtatásához az alábbi mappa-struktúrának kell rendelkezésre állnia a gyökérkönyvtárban:

```text
Thesis/
├── Test_Data/        # Ide kell helyezni a nyers mérési adatokat (.xlsx formátumban)
├── Results/          # A generált ábrák, riportok (.html, .xlsx) és modellek (.pkl) helye
├── Cache/            # Ideiglenes fájlok a gyorsabb újrafutás érdekében
├── main.py           # A fő futtatható szkript
├── config.py         # Globális beállítások (könyvtárak, paraméterek, stílusok)
├── utils.py          # Segédfüggvények (ábrázolás, adatkezelés, riportálás)
├── transformers.py   # Egyedi scikit-learn transzformátorok (VIF, Interakciók, Scaler)
├── requirements.txt  # A projekt Python függőségei
└── README.md         # Ez a dokumentum
```
*(Megjegyzés: A `Results` és a `Cache` mappákat a program automatikusan létrehozza, ha nem léteznek).*

## Telepítés

1. Győződj meg róla, hogy Python 3.9 vagy újabb verzió van telepítve a gépeden.
2. Nyiss egy parancssort a projekt mappájában, és telepítsd a szükséges csomagokat:

```bash
pip install -r requirements.txt
```

## Használat

1. Helyezd a tribométeres méréseket tartalmazó Excel fájlokat (`.xlsx`) a `Test_Data` mappába. Az Excel fájloknak tartalmazniuk kell egy `Sheet Numeric SRA` (vagy elsődleges) munkalapot a megfelelő oszlopokkal (Time, Load, Temperature, COF stb.).
2. Futtasd a fő szkriptet:

```bash
python main.py
```

3. A futás befejeztével az eredmények megtalálhatók a `Results` mappában, és az összefoglaló `Eredmenyek_Riport.html` automatikusan megnyílik az alapértelmezett böngésződben.

### Gyorsítótár (Cache) használata
A program az adatok betöltését és a modellek betanítását alapértelmezetten gyorsítótárazza (cache) a gyorsabb fejlesztés és ábragenerálás érdekében. Ha a nyers adatok megváltoztak, vagy nulláról szeretnéd újratanítani a modelleket, állítsd a `config.py`-ban a `USE_CACHE` változót `False`-ra, vagy töröld a `Cache` mappa tartalmát.

## Kiegészítő szkriptek
A projekt tartalmaz néhány kiegészítő szkriptet is, amelyek a diplomamunkához szükséges extra ábrákat és folyamatábrákat generálják:
* `Hertzian.py`: Legenerálja a Hertz-féle feszültségeloszlást bemutató 3D-s ábrát.
* `Stribeck Curve.py`: Egy idealizált Stribeck-görbét rajzol ki.

Ezeket a `python fájlnév.py` paranccsal külön lehet futtatni.
