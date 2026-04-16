import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# Dados de exemplo: intenções de um Chatbot
data = {
    'texto': ["cancelar pedido", "estorno", "ajuda login", "senha errada"],
    'classe': ["financeiro", "financeiro", "suporte", "suporte"]
}
df = pd.DataFrame(data)

# Criando um Pipeline que une Vetorização + Modelo
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])

# Treinando
pipeline.fit(df['texto'], df['classe'])

# SALVANDO O MODELO NO DISCO
joblib.dump(pipeline, 'modelo_chatbot.pkl')
print(" Modelo salvo com sucesso como 'modelo_chatbot.pkl'!")
