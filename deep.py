import numpy as np
import pandas as pd
import streamlit as st
import dill

# Titre de l'application
st.title("Clustering - Déploiement")

# Mise en cache du chargement du modèle
@st.cache_resource()
def chargement_modele():
    """Charge le modèle enregistré avec dill."""
    try:
        with open("cluster.sav", "rb") as f:
            modele = dill.load(f)
    except FileNotFoundError:
        st.error("Le fichier 'wine.sav' est introuvable. Assurez-vous qu'il est dans le même répertoire que ce script.")
        modele = None
    return modele

# Chargement du modèle ou du pipeline
wkf = chargement_modele()
if wkf is None:
    st.stop()

# Organisation en colonnes
col1, col2, col3 = st.columns([1, 2, 1])

# Saisie utilisateur
SepLen = col1.text_input("Sepal length: (4.3 - 7.9)", "5")
SepWidth = col1.text_input("Sepal width: (2.0 - 4.4)", "3")
PetLen = col2.text_input("Petal length: (1.0 - 6.9)", "2")
PetWidth = col2.text_input("Petal width: (0.1 - 2.5)", "1")

# Bouton pour calculer
if col3.button("Calculer"):
    # Fonction pour convertir les saisies utilisateur
    def try_parse(str_value):
        try:
            value = float(str_value)
        except ValueError:
            value = np.nan
        return value

    # Conversion des saisies utilisateur en DataFrame
    the_value = {
        'Sepal_Length': try_parse(SepLen),
        'Sepal_Width': try_parse(SepWidth),
        'Petal_Length': try_parse(PetLen),
        'Petal_Width': try_parse(PetWidth)
    }

    df = pd.DataFrame([the_value])

    # Vérification si des valeurs sont NaN
    if df.isnull().values.any():
        st.error("Certains champs contiennent des valeurs invalides. Veuillez vérifier vos entrées.")
    else:
        # Si wkf est un pipeline, il faut le passer correctement à travers le pipeline
        try:
            # Assurez-vous que les données sont dans le bon format (par exemple, numpy array)
            data_to_predict = df.values  # Convertir DataFrame en numpy array
            
            # Vérification si wkf est un pipeline et s'il contient un pré-traitement
            if hasattr(wkf, 'predict'):
                cluster = wkf.predict(data_to_predict)[0]  # Prédiction du cluster
                distances = wkf.transform(data_to_predict)[0]  # Calcul des distances aux clusters
            else:
                st.error("Le modèle chargé n'est pas un pipeline valide contenant un KMeans.")
                st.stop()
            
            # Affichage des résultats
            st.write(f"Cluster attribué : **{cluster}**")
            st.write("Distances aux clusters :")
            st.dataframe(pd.DataFrame(distances, index=["Distance"], columns=['Cluster 0', 'Cluster 1', 'Cluster 2']))
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")

        # Affichage d'une image supplémentaire
        st.image("figure_1.png", caption="Caractérisation des groupes")
