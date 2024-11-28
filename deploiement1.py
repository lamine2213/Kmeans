import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, classification_report, confusion_matrix

# Charger les données
@st.cache_data
def load_data():
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    return train_df, test_df

train_df, test_df = load_data()

st.title("Analyse et Modélisation des Données de Satisfaction des Passagers Aériens")

# Prétraitement des données
st.header("Prétraitement des données")
st.write("Informations initiales sur les données d'entraînement :")
st.write(train_df.info())

# Séparer les colonnes numériques et catégorielles
num_cols = train_df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = train_df.select_dtypes(include=['object']).columns

# Imputation des valeurs manquantes
num_imputer = SimpleImputer(strategy='mean')
train_df[num_cols] = num_imputer.fit_transform(train_df[num_cols])
test_df[num_cols] = num_imputer.transform(test_df[num_cols])

cat_imputer = SimpleImputer(strategy='most_frequent')
train_df[cat_cols] = cat_imputer.fit_transform(train_df[cat_cols])
test_df[cat_cols] = cat_imputer.transform(test_df[cat_cols])

# Vérification post-imputation
st.write("Nombre de valeurs manquantes après imputation :")
st.write(train_df.isna().sum())

# Suppression des colonnes inutiles
columns_to_drop = ['Unnamed: 0', 'id']
train_df.drop(columns=[col for col in columns_to_drop if col in train_df.columns], axis=1, inplace=True)
test_df.drop(columns=[col for col in columns_to_drop if col in test_df.columns], axis=1, inplace=True)

# Encodage des variables catégorielles
for col in ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']:
    lb = LabelEncoder()
    train_df[col] = lb.fit_transform(train_df[col])
    test_df[col] = lb.transform(test_df[col])

st.write("Données après le nettoyage et le codage :")
st.write(train_df.describe())

# Analyse exploratoire des données (EDA)
st.header("Analyse Exploratoire des Données (EDA)")

# Graphiques interactifs
st.subheader("Distribution des variables catégorielles")
columns_to_plot = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']
column = st.selectbox("Choisir une colonne à explorer :", columns_to_plot)
fig, ax = plt.subplots()
sns.countplot(data=train_df, x=column, ax=ax)
st.pyplot(fig)

# Matrice de corrélation
st.subheader("Matrice de Corrélation")
cor = train_df.corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cor, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Préparation des données pour la modélisation
st.header("Modélisation des Données")
train_x = train_df.drop('satisfaction', axis=1)
train_y = train_df['satisfaction']
test_x = test_df.drop('satisfaction', axis=1)
test_y = test_df['satisfaction']

scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

# Choix du modèle
st.subheader("Choix et Evaluation des Modèles")
models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=10),
    "SVM": SVC(probability=True),
    "Naive Bayes": MultinomialNB()
}

selected_model = st.selectbox("Choisissez un modèle à entraîner :", list(models.keys()))
model = models[selected_model]
model.fit(train_x, train_y)
predictions = model.predict(test_x)

# Évaluation du modèle
accuracy = accuracy_score(test_y, predictions)
precision = precision_score(test_y, predictions, average='weighted')
recall = recall_score(test_y, predictions, average='weighted')
st.write(f"**Accuracy** : {accuracy:.2f}")
st.write(f"**Precision** : {precision:.2f}")
st.write(f"**Recall** : {recall:.2f}")
st.text("Rapport de classification :")
st.text(classification_report(test_y, predictions))

# Courbe ROC
st.subheader("Courbe ROC")
if hasattr(model, "predict_proba"):
    y_prob = model.predict_proba(test_x)[:, 1]
else:  # For models like SVM
    y_prob = model.decision_function(test_x)

fpr, tpr, thresholds = roc_curve(test_y, y_prob)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots()
ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic')
ax.legend(loc="lower right")
st.pyplot(fig)
