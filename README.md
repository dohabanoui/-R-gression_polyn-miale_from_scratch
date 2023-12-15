# Multi-Polynomial Regression

## Table of Contents
1. [Le modèle](#le-modèle)
2. [Les erreurs du modèle](#les-erreurs-du-modèle)
3. [Gradient](#gradient)
4. [Évaluation du modèle](#évaluation-du-modèle)
5. [Analyse du dataset](#analyse-du-dataset)
6. [Diviser notre dataset en x et y](#diviser-notre-dataset-en-x-et-y)
7. [Split des données](#split-des-données)
8. [La normalisation du dataset](#la-normalisation-du-dataset)
9. [Pour le cas de la régression linéaire](#pour-le-cas-de-la-régression-linéaire)
10. [Pour la régression polynomiale multivariée](#pour-la-régression-polynomiale-multivariée)
11. [Conclusion](#conclusion)

## Le modèle
- **Création de la matrice X**
  - [x_train](#) et [x_test](#)
- **Création d'un vecteur paramètre θ**
  - Initialisation avec des coefficients aléatoires
- **Fonction coût : Erreur Quadratique Moyenne**
  - [Cost Function](#)
- **Phase d'entraînement**
  - [Descente de gradient](#) avec et sans régularisation
- **Vecteur de prédiction**
  - [Predictions](#)

## Les erreurs du modèle
- Erreurs quadratiques moyennes pour différentes régularisations

## Gradient
- [Courbes d'apprentissage](#)

## Évaluation du modèle
- Coefficient de détermination
  - [Performances](#)

## Analyse du dataset
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from main import*
%matplotlib inline
dataset=pd.read_csv("advertising.csv")
dataset.head()
