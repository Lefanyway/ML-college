from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import nest_asyncio
from pyngrok import ngrok
import uvicorn
import threading # Import threading

# CONFIGURAÇÃO DO NGROK (Substitui pelo teu token real)
# Sem isto, o ngrok dará erro de autenticação!
NGROK_TOKEN = "COLOQUE AQUI O SEU TOKEN"
ngrok.set_auth_token(NGROK_TOKEN)

app = FastAPI()

# Definimos o formato do JSON que a API vai receber
class Mensagem(BaseModel):
    texto: str

# Carregue o modelo
model = joblib.load('modelo_chatbot.pkl')

@app.post("/predict")
async def predict(item: Mensagem):
    # O modelo faz a predição baseada no texto recebido
    predicao = model.predict([item.texto])[0]
    probabilidade = model.predict_proba([item.texto]).max()

    return {
        "intencao": predicao,
        "confianca": float(probabilidade),
        "status": "sucesso"
    }

# Configuração para rodar o servidor dentro do Colab
nest_asyncio.apply()

# Criamos o túnel público
try:
    public_url = ngrok.connect(8000).public_url
    print(f" TUA API ESTÁ ONLINE EM: {public_url}")
    print("Para testar, adiciona '/docs' ao final da URL acima.")
except Exception as e:
    print(f" Erro ao conectar o ngrok: {e}")

# FIX: Run Uvicorn in a separate thread to avoid event loop conflicts in Colab
config = uvicorn.Config(app, host="0.0.0.0", port=8000, loop="asyncio")
server = uvicorn.Server(config)

def run_server():
    server.run()

thread = threading.Thread(target=run_server)
thread.start()

print("Uvicorn server started in a background thread.")
print("The Colab cell has finished executing. The server should be accessible via ngrok.")

