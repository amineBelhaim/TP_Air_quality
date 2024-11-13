import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Charger les données
df = pd.read_csv('indices_QA_commune_IDF_2016.csv')

# Retirer les valeurs nulles
df.dropna(inplace=True)

# Définir les seuils pour plusieurs niveaux de qualité de l'air
def assign_quality(pm10, no2):
    if pm10 <= 20 and no2 <= 20:
        return 0  # Très bonne
    elif pm10 <= 40 and no2 <= 30:
        return 1  # Bonne
    elif pm10 <= 50 and no2 <= 40:
        return 2  # Modérée
    elif pm10 <= 75 and no2 <= 50:
        return 3  # Mauvaise
    else:
        return 4  # Très mauvaise

# Appliquer la fonction de classification sur chaque ligne du DataFrame
df['qualite_air'] = df.apply(lambda x: assign_quality(x['pm10'], x['no2']), axis=1)

# Filtrer les valeurs aberrantes basées sur des seuils de concentration
df = df[(df['pm10'] <= 100) & (df['no2'] <= 100) & (df['o3'] <= 100)]

# Préparation des données pour l'arbre de décision
X = df[['no2', 'o3', 'pm10']]
y = df['qualite_air']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer et entraîner le modèle
clf = DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_split=5, min_samples_leaf=2)
clf.fit(X_train, y_train)

# Visualiser l'arbre
plt.figure(figsize=(20, 10), dpi=300)
plot_tree(clf, filled=True, feature_names=X.columns,
          class_names=['Très bonne', 'Bonne', 'Modérée', 'Mauvaise', 'Très mauvaise'])
plt.show()

# Tester la prédiction pour des valeurs représentant différentes qualités de l'air
exemples = pd.DataFrame([
    [15, 10, 15],  # Très bonne qualité
    [25, 15, 35],  # Bonne qualité
    [85, 30, 90]   # Très mauvaise qualité
], columns=['no2', 'o3', 'pm10'])

# Prédictions attendues pour les exemples
predictions_attendues = [0, 1, 4]  # Très bonne, Bonne, Très mauvaise

# Prédire la qualité de l'air pour ces exemples
predictions = clf.predict(exemples)

classes = ['Très bonne', 'Bonne', 'Modérée', 'Mauvaise', 'Très mauvaise']
for i, (attendue, pred) in enumerate(zip(predictions_attendues, predictions)):
    if attendue == pred:  # Afficher seulement les prédictions correctes
        print(f"Exemple {i + 1} - Prédiction attendue : {classes[attendue]} - Prédiction de l'algo : {classes[pred]} "
              f"(valeurs no2={exemples.iloc[i, 0]}, o3={exemples.iloc[i, 1]}, pm10={exemples.iloc[i, 2]})")
