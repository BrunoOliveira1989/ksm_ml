# treino_chance_de_compra.py

from sqlalchemy import create_engine, text
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import pandas as pd
import joblib

# Importa engine do banco a partir da config (recomendado)
from config.database import engine  # Ajuste esse import conforme sua estrutura

def carregar_dados_clientes():
    # Busca as vendas agrupadas por cliente e data
    query = text("""
        SELECT customer_id, issue_date, SUM(total) as total_gasto
        FROM sales
        GROUP BY customer_id, issue_date
    """)
    with engine.connect() as conn:
        rows = conn.execute(query).fetchall()

    df = pd.DataFrame(rows, columns=["cliente_id", "data", "total_gasto"])
    df["data"] = pd.to_datetime(df["data"])

    # Última data total do banco para usar como referência "hoje"
    data_ultima_banco = df["data"].max()

    # Última data de compra por cliente
    ultima_compra = df.groupby("cliente_id")["data"].max().reset_index()
    ultima_compra.columns = ["cliente_id", "ultima_compra"]

    # Total gasto por cliente
    total_gasto = df.groupby("cliente_id")["total_gasto"].sum().reset_index()
    total_gasto.columns = ["cliente_id", "total_gasto"]

    # Quantidade de compras por cliente
    qtd_compras = df.groupby("cliente_id").size().reset_index(name="qtd_compras")

    # Junta todas as informações
    df_final = ultima_compra.merge(total_gasto, on="cliente_id")
    df_final = df_final.merge(qtd_compras, on="cliente_id")

    # Calcula dias desde a última compra considerando a última data do banco
    df_final["dias_desde_ultima"] = (data_ultima_banco - df_final["ultima_compra"]).dt.days

    # Definir target: vai comprar = 1 se comprou nos últimos 30 dias até a última data do banco
    janela_dias = 30
    df_final["vai_comprar"] = (df_final["dias_desde_ultima"] <= janela_dias).astype(int)

    # Seleciona features e target
    X = df_final[["total_gasto", "qtd_compras", "dias_desde_ultima"]]
    y = df_final["vai_comprar"]
    clientes = df_final["cliente_id"]

    print(f"Total clientes: {len(clientes)}")
    print(f"Clientes com compra recente (últimos {janela_dias} dias): {y.sum()}")

    return X, y, clientes

def treinar_modelo(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    clf.fit(X_train, y_train)

    BASE_DIR = Path(__file__).parent.resolve()
    DATA_DIR = BASE_DIR.parent / "data"        # pasta data um nível acima
    DATA_DIR.mkdir(exist_ok=True)               # cria a pasta data se não existir

    joblib.dump(clf, DATA_DIR / "modelo_chance.pkl", compress=3)
    joblib.dump(scaler, DATA_DIR / "scaler.pkl")

    print("✅ Modelo treinado e salvo em data/ com sucesso.")

if __name__ == "__main__":
    X, y, clientes = carregar_dados_clientes()
    treinar_modelo(X, y)
