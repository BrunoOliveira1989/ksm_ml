from pydantic import BaseModel
from sqlalchemy import text
from pathlib import Path
import pandas as pd
from collections import defaultdict

# Importa engine do banco a partir da config (recomendado)
from config.database import engine  # Ajuste esse import conforme sua estrutura

# Paths para carregar modelos de machine learning (ajustando para o diretório correto)
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR.parent / "data"

import joblib

# Carregar modelos ML para chance e recomendação
clf_chance = joblib.load(DATA_DIR / "modelo_chance.pkl")
scaler = joblib.load(DATA_DIR / "scaler.pkl")

clf_recom = joblib.load(DATA_DIR / "modelo_treinado.pkl")
mlb = joblib.load(DATA_DIR / "mlb.pkl")
le = joblib.load(DATA_DIR / "le.pkl")

def carregar_produtos():
    query = text("SELECT id, description FROM products")
    with engine.connect() as conn:
        rows = conn.execute(query).mappings().all()
    return {str(r["id"]): r["description"] for r in rows}

produtos_dict = carregar_produtos()

def carregar_historico():
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
    return historico

historico_compras = carregar_historico()

def carregar_dados_cliente(cliente_id=None):
    query = """
        SELECT customer_id, issue_date, SUM(total) as total_gasto
        FROM sales
        GROUP BY customer_id, issue_date
    """
    with engine.connect() as conn:
        rows = conn.execute(text(query)).fetchall()

    df = pd.DataFrame(rows, columns=["cliente_id", "data", "total_gasto"])
    df["data"] = pd.to_datetime(df["data"])
    data_ultima_banco = df["data"].max()

    ultima_compra = df.groupby("cliente_id")["data"].max().reset_index()
    ultima_compra.columns = ["cliente_id", "ultima_compra"]

    total_gasto = df.groupby("cliente_id")["total_gasto"].sum().reset_index()
    total_gasto.columns = ["cliente_id", "total_gasto"]

    qtd_compras = df.groupby("cliente_id").size().reset_index(name="qtd_compras")

    df_final = ultima_compra.merge(total_gasto, on="cliente_id")
    df_final = df_final.merge(qtd_compras, on="cliente_id")

    df_final["dias_desde_ultima"] = (data_ultima_banco - df_final["ultima_compra"]).dt.days

    if cliente_id is not None:
        df_final = df_final[df_final["cliente_id"] == cliente_id]
        if df_final.empty:
            return None

    return df_final[["cliente_id", "total_gasto", "qtd_compras", "dias_desde_ultima"]]

# Modelos Pydantic para validação e tipagem

class Cliente(BaseModel):
    cliente_id: int
