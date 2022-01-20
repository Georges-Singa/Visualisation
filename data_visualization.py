from os import chdir
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


chdir("C:\\Users\Georges-Singa\\Desktop\\IA\\Checkpoints_3")
df = pd.read_csv("titanic-passengers.csv", sep=';')

# en-tête
df.head

# informations générales sur les colonnes
df.dtypes
df.info()
df.columns
df.describe()
# informations manquantes
df.isnull()
df.isnull().sum()
df.isnull().sum().sum()

#  remplacez-les par les valeurs appropriées
df["Age"].fillna(df["Age"].mean(), inplace=True)
df.dropna(axis=1, how="all")
df["Cabin"].fillna('G6', inplace=True)

"""encoder = LabelEncoder()
df[""] = encoder.fit_transform(df[""])
"""

# visualisation des données
#.......................................la distribution des caractéristiques les plus importantes
sns.distplot(df["PassengerId"], bins=10, hist=True, kde=True, color='red')
sns.distplot(df["Pclass"], bins=10, hist=True, kde=True, color='blue')
sns.distplot(df["Age"], bins=10, hist=True, kde=True, color='green')

plt.xlabel("Survived")
plt.ylabel("survived count")
survived_count = df["Survived"].value_counts()
survived_count.plot.bar()

plt.xlabel("Sexe")
plt.ylabel("Sexe count")
survived_count = df["Sex"].value_counts()
survived_count.plot.bar()

plt.xlabel("Embarked")
plt.ylabel("Embarked count")
survived_count = df["Embarked"].value_counts()
survived_count.plot.bar()

sns.countplot(x="Survived", data=df)
plt.xticks(rotation=-45)

# Visualisez la corrélation entre le sexe et l'âge
corr = sns.FacetGrid(df, col="Sex")
corr.map(plt.hist, "Age", bins=10)

def plot_correlation_map(data_frame):
    corr = data_frame.corr()
    s , ax = plt.subplots( figsize =( 12 , 10 ) )
    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    s = sns.heatmap(corr, cmap = cmap, square=True, cbar_kws={'shrink':.9 }, ax=ax, annot=True, annot_kws={'fontsize':12 })
    
# écrivez un paragraphe décrivant votre analyse (le paragraphe doit être en anglais).
    """
        [summary]
    """

# Utilisez la fonction groupby combinée avec la moyenne () pour afficher la relation entre Pclass et survived
cleanup={"Survived":{"yes":1, "no": 0}}
df.replace(cleanup, inplace=True)
df[["Pclass", "Survived"]].groupby(["Pclass"], as_index=True)

# supprimer les colonnes inutiles telles que les noms
to_del = ["Name", "Sex", "Ticket", "Cabin", "Embarked"]
for col in to_del:
    del df[col]

# créer une nouvelle colonne appelée Titre qui contient le titre approprié pour chaque individu (indice : extrayez le titre de la colonne Noms)

# Visualisez la corrélation entre le titre et d'autres caractéristiques (par exemple, le sexe, le tarif, l'âge...)

# regrouper ces titres en titres plus utiles.


"""visualisons maintenant les mêmes corrélations en utilisant ces nouveaux titres.

Enfin, utilisez les colonnes Parch et SibSp pour créer une fonctionnalité plus utile, appelons-la FamilySize.

Utilisez ce que vous avez appris jusqu'à présent pour déterminer si cette fonctionnalité est utile ou non."""






