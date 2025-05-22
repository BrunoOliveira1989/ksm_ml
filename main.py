# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from collections import defaultdict
from typing import List
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from fastapi import Query
import numpy as np

# ---------------------------------------------------
# 1. Conexão com o banco
# ---------------------------------------------------
DATABASE_URL = (
    "postgresql+psycopg2://kodiak_pocket_owner:"
    "k0rm1fPEwAyU@ep-long-surf-a5z0iq90-pooler."
    "us-east-2.aws.neon.tech/ksm?sslmode=require"
)

engine = create_engine(DATABASE_URL)

# ---------------------------------------------------
# 2. Carregar histórico de compras direto do banco
# ---------------------------------------------------
def carregar_historico() -> List[dict]:
    """
    Lê customer_id e product_id da tabela sales
    e devolve uma lista no formato:
    [{"cliente_id": 123, "produtos": [prodA, prodB, ...]}, ...]
    """
    query = text(
        """
        SELECT customer_id, product_id
        FROM sales
        """
    )

    with engine.connect() as conn:
        rows = conn.execute(query).mappings().all()

    # Agrupar produtos por cliente
    agrupado = defaultdict(set)        # usa set para evitar duplicados
    for r in rows:
        agrupado[r["customer_id"]].add(str(r["product_id"]))  # garante string

    # Converte para a estrutura desejada
    historico = [
        {"cliente_id": cid, "produtos": list(produtos)}
        for cid, produtos in agrupado.items()
        if len(produtos) > 1            # exclui clientes com só 1 produto
    ]
    return historico


historico_compras = carregar_historico()

# ---------------------------------------------------
# 3. Preparar dados e treinar o modelo
# ---------------------------------------------------
X, y = [], []
for compra in historico_compras:
    produtos = compra["produtos"]
    for produto_alvo in produtos:
        features = [p for p in produtos if p != produto_alvo]
        if features:                    # garante que haja features
            X.append(features)
            y.append(produto_alvo)

mlb = MultiLabelBinarizer()
X_encoded = mlb.fit_transform(X)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_encoded, y_encoded)

# ---------------------------------------------------
# 3.1 Carregar nomes dos produtos
# ---------------------------------------------------
def carregar_produtos() -> dict:
    """
    Lê product_id e description da tabela products
    e devolve um dicionário {id: descrição}
    """
    query = text(
        """
        SELECT id, description
        FROM products
        """
    )

    with engine.connect() as conn:
        rows = conn.execute(query).mappings().all()

    return {str(r["id"]): r["description"] for r in rows}

produtos_dict = carregar_produtos()

# ---------------------------------------------------
# 4. API FastAPI
# ---------------------------------------------------
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

    # Monta sugestões normalmente
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
    
# ---------------------------------------------------
# 5. Para executar:
# uvicorn app:app --reload
# ---------------------------------------------------
