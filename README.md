1/ Fraud Detection — Chargeback Classification

Projet de détection de fraudes sur des transactions financières.

2/ Objectif

Prédire si un client est frauduleux (`target_is_fraud = 1`) à partir de données comportementales et financières.
Minimiser les faux positifs et mieux détecter les faux négatifs.


3/ Preprocessing (`preprop_final.ipynb`)
- Suppression des colonnes à risque de leakage
- Encodage des variables catégorielles
- Standardisation (StandardScaler)
- Feature engineering (interactions, flags)
- SMOTE appliqué **uniquement sur le train set** après split

4/ Modèles entraînés
 Modèle | Description 

 Régression Logistique | Baseline 
 Decision Tree | Interprétable, optimisé en profondeur 
 XGBoost (Kaggle) | Optimisé via Optuna sur ROC-AUC 
 XGBoost (Coût) | Optimisé sur matrice de coût métier 


5/ Ordre d'exécution

**1. `EDA/EDA_Final.ipynb`**
Analyse exploratoire : distributions, corrélations, détection d'outliers, visualisation du déséquilibre de classes.

**2. `Preprocess/preprop_final.ipynb`**
Nettoyage et transformation des données brutes. Génère les fichiers `train_preprocessed.csv` et `test_preprocessed.csv` dans `data/`.

**3. `Modelisation/regression_logistique.ipynb`**
Baseline avec régression logistique sur les données preprocessées.

**4. `Models/DecisionTree_Final.ipynb`**
Entraînement et optimisation d'un Decision Tree.

**5. `Models/XGBOOST_Optimisation_Kaggle.ipynb`**
XGBoost avec tuning Optuna optimisé sur le score Kaggle (ROC-AUC).

**6. `Models/XGBOOST_Optimisation_coût.ipynb`**
Variante XGBoost optimisée sur une matrice de coût métier.

---

## 📊 Données

- **Train** : 233 956 lignes, 21 features, ~3% de fraudes (avant SMOTE)
- **Test** : 40 000 lignes
- **Features** : age, tenure, revenus, score crédit, comportement transactionnel, risque IP, devices, etc.

---

## ⚙️ Installation

```bash
pip install numpy pandas scikit-learn xgboost imbalanced-learn optuna torch
```

---

## 📤 Format de soumission

```csv
customer_id,target_is_fraud
CUST_XXXXXXXX,0
CUST_YYYYYYYY,1
```
