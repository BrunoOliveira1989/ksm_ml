# main.py
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import create_engine, text
import joblib
import numpy as np

# Conexão
DATABASE_URL = (
    "postgresql+psycopg2://kodiak_pocket_owner:"
    "k0rm1fPEwAyU@ep-long-surf-a5z0iq90-pooler."
    "us-east-2.aws.neon.tech/ksm?sslmode=require"
)
engine = create_engine(DATABASE_URL)

# Carregar modelos treinados
clf = joblib.load("modelo_treinado.pkl")
mlb = joblib.load("mlb.pkl")
le = joblib.load("le.pkl")

# Carregar produtos
def carregar_produtos():
    query = text("SELECT id, description FROM products")
    with engine.connect() as conn:
        rows = conn.execute(query).mappings().all()
    return {str(r["id"]): r["description"] for r in rows}

produtos_dict = carregar_produtos()

# Carregar histórico
def carregar_historico():
    query = text("SELECT customer_id, product_id FROM sales")
    with engine.connect() as conn:
        rows = conn.execute(query).mappings().all()

    from collections import defaultdict
    agrupado = defaultdict(set)
    for r in rows:
        agrupado[r["customer_id"]].add(str(r["product_id"]))
    historico = [
        {"cliente_id": cid, "produtos": list(produtos)}
        for cid, produtos in agrupado.items()
        if len(produtos) > 1
    ]
    return historico

historico_compras = carregar_historico()

# FastAPI
app = FastAPI(title="Sugestão de Produtos")

class Cliente(BaseModel):
    cliente_id: int

@app.post("/sugerir/")
def sugerir(cliente: Cliente, n: int = 3, listar_comprados: bool = Query(False)):
    produtos_cliente = next(
        (h["produtos"] for h in historico_compras if h["cliente_id"] == cliente.cliente_id),
        None,
    )
    if not produtos_cliente:
        raise HTTPException(404, "Cliente não encontrado ou sem histórico suficiente")

    x_input = mlb.transform([produtos_cliente])
    probs = clf.predict_proba(x_input)[0]
    top_n_idx = np.argsort(probs)[::-1][:n]
    sugestoes = le.inverse_transform(top_n_idx)

    sugestoes_com_nome = [
        {"produto_id": pid, "nome": produtos_dict.get(pid, "Nome desconhecido")}
        for pid in sugestoes
    ]

    resposta = {
        "cliente_id": cliente.cliente_id,
        "sugestoes": sugestoes_com_nome,
    }

    if listar_comprados:
        resposta["produtos_comprados_nome"] = [
            {"produto_id": pid, "nome": produtos_dict.get(pid, "Nome desconhecido")}
            for pid in produtos_cliente
        ]

    return resposta
