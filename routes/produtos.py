from fastapi import APIRouter, HTTPException, Query
import numpy as np
from models.models import (
    Cliente,
    historico_compras,
    clf_recom,
    mlb,
    le,
    produtos_dict,
)

router = APIRouter()

@router.post("/venda/sugerir")
def sugerir_por_cliente(
    cliente: Cliente,
    n: int = Query(3, ge=1, le=10, description="Quantidade de sugestões")
):
    cliente_id = cliente.cliente_id
    produtos_cliente = next(
        (h["produtos"] for h in historico_compras if h["cliente_id"] == cliente_id),
        None,
    )
    if not produtos_cliente:
        raise HTTPException(404, "Cliente não encontrado ou sem histórico suficiente")

    x_input = mlb.transform([produtos_cliente])
    probs = clf_recom.predict_proba(x_input)[0]
    top_n_idx = np.argsort(probs)[::-1][:n]
    sugestoes = le.inverse_transform(top_n_idx)

    sugestoes_com_nome = [
        {"produto_id": pid, "nome": produtos_dict.get(pid, "Nome desconhecido")}
        for pid in sugestoes
    ]

    return {
        "cliente_id": cliente_id,
        "sugestoes": sugestoes_com_nome,
    }
