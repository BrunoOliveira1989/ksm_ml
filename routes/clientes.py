from fastapi import APIRouter, HTTPException
from models.models import (
    Cliente,
    carregar_dados_cliente,
    clf_chance,
    scaler,
)

router = APIRouter()

@router.post("/compra/probabilidade/top10")
def clientes_top10():
    df_clientes = carregar_dados_cliente()
    X = df_clientes[["total_gasto", "qtd_compras", "dias_desde_ultima"]]
    X_scaled = scaler.transform(X)

    probs = clf_chance.predict_proba(X_scaled)[:, 1]
    df_clientes = df_clientes.copy()
    df_clientes["chance_compra"] = probs

    top10 = df_clientes.sort_values("chance_compra", ascending=False).head(10)
    return [
        {"cliente_id": int(row["cliente_id"]), "chance_compra": float(row["chance_compra"])}
        for _, row in top10.iterrows()
    ]

@router.post("/compra/probabilidade")
def cliente_chance(cliente: Cliente):
    cliente_id = cliente.cliente_id
    df_clientes = carregar_dados_cliente(cliente_id)
    if df_clientes is None:
        raise HTTPException(status_code=404, detail="Cliente não encontrado")

    X = df_clientes[["total_gasto", "qtd_compras", "dias_desde_ultima"]]
    X_scaled = scaler.transform(X)
    probs = clf_chance.predict_proba(X_scaled)[:, 1]

    return {
        "cliente_id": cliente_id,
        "chance_compra": float(probs[0])
    }
