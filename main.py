from fastapi import FastAPI
from routes import clientes, produtos
import os

app = FastAPI(title="API Unificada: Chance de Compra + Recomendação")

app.include_router(clientes.router, prefix="/clientes", tags=["Clientes"])
app.include_router(produtos.router, prefix="/produtos", tags=["Produtos"])

#if __name__ == "__main__":
#    import uvicorn
#    port = int(os.environ.get("PORT", 8000))  # pega a porta da variável de ambiente ou usa 8000 por padrão
#    uvicorn.run(app, host="0.0.0.0", port=port)
