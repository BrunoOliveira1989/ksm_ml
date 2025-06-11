from fastapi import FastAPI
from routes import clientes, produtos

app = FastAPI(title="API Unificada: Chance de Compra + Recomendação")

app.include_router(clientes.router, prefix="/clientes", tags=["Clientes"])
app.include_router(produtos.router, prefix="/produtos", tags=["Produtos"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
