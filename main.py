from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from collections import defaultdict
from typing import List
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
import numpy as np

# ---------------------------------------------------
# 1. Conexão com o banco
# ---------------------------------------------------
print("[INFO] Conectando ao banco de dados...")

DATABASE_URL = (
    "postgresql+psycopg2://kodiak_pocket_owner:"
    "k0rm1fPEwAyU@ep-long-surf-a5z0iq90-pooler."
    "us-east-2.aws.neon.tech/ksm?sslmode=require"
)

engine = create_engine(DATABASE_URL)
print("[INFO] Conexão criada com sucesso.")

# ---------------------------------------------------
# 2. Carregar histórico de compras direto do banco
# ---------------------------------------------------
def carregar_historico() -> List[dict]:
    print("[INFO] Carregando histórico de compras da tabela 'sales'...")
    query = text("SELECT customer_id, product_id FROM sales")

    with engine.connect() as conn:
        rows = conn.execute(query).mappings().all()

    agrupado = defaultdict(set)
    for r in rows:
        agrupado[r["customer_id"]].add(str(r["product_id"]))

    historico = [
        {"cliente_id": cid, "produtos": list(produtos)}
        for cid, produtos in agrupado.items()
        if len(produtos) > 1
    ]
    print(f"[INFO] Histórico carregado com sucesso: {len(historico)} clientes encontrados.")
    return historico

historico_compras = carregar_historico()

# ---------------------------------------------------
# 3. Preparar dados e treinar o modelo
# ---------------------------------------------------
print("[INFO] Preparando dados e treinando o modelo...")

X, y = [], []
for compra in historico_compras:
    produtos = compra["produtos"]
    for produto_alvo in produtos:
        features = [p for p in produtos if p != produto_alvo]
        if features:
            X.append(features)
            y.append(produto_alvo)

mlb = MultiLabelBinarizer()
X_encoded = mlb.fit_transform(X)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_encoded, y_encoded)

print(f"[INFO] Modelo treinado com sucesso. Total de amostras: {len(X)}")

# ---------------------------------------------------
# 3.1 Carregar nomes dos produtos
# ---------------------------------------------------
def carregar_produtos() -> dict:
    print("[INFO] Carregando nomes dos produtos da tabela 'products'...")
    query = text("SELECT id, description FROM products")

    with engine.connect() as conn:
        rows = conn.execute(query).mappings().all()

    produtos = {str(r["id"]): r["description"] for r in rows}
    print(f"[INFO] {len(produtos)} produtos carregados.")
    return produtos

produtos_dict = carregar_produtos()

# ---------------------------------------------------
# 4. API FastAPI
# ---------------------------------------------------
print("[INFO] Inicializando API FastAPI...")

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

print("[INFO] API pronta para receber requisições.")

# ---------------------------------------------------
# Para executar local:
# uvicorn main:app --reload
# No Docker:
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
# ---------------------------------------------------
