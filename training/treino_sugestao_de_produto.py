# treino_sugestao_de_produto.py

from sqlalchemy import create_engine, text
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from pathlib import Path
import joblib

# Importa engine do banco a partir da config (recomendado)
from config.database import engine  # Ajuste esse import conforme sua estrutura

# Carregar histórico de compras dos clientes
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

# Treinar modelo de recomendação
def treinar_modelo(historico):
    X, y = [], []
    for compra in historico:
        produtos = compra["produtos"]
        for produto_alvo in produtos:
            features = [p for p in produtos if p != produto_alvo]
            if features:
                X.append(features)
                y.append(produto_alvo)

    # Binarização dos dados de entrada (produtos comprados juntos)
    mlb = MultiLabelBinarizer()
    X_encoded = mlb.fit_transform(X)

    # Codificação do rótulo (produto-alvo)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Diagnóstico do tamanho das matrizes
    #print(f"Shape de X_encoded: {X_encoded.shape}")
    #print(f"Shape de y_encoded: {y_encoded.shape}")

    # Treinamento do modelo com limitação de profundidade para evitar arquivos enormes
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,  # evita árvores profundas demais
        random_state=42
    )
    clf.fit(X_encoded, y_encoded)

    # Caminhos para salvar os arquivos
    BASE_DIR = Path(__file__).parent.resolve()
    DATA_DIR = BASE_DIR.parent / "data"       # pasta data um nível acima
    DATA_DIR.mkdir(exist_ok=True)              # cria a pasta data se não existir

    modelo_path = DATA_DIR / "modelo_treinado.pkl"
    mlb_path = DATA_DIR / "mlb.pkl"
    le_path = DATA_DIR / "le.pkl"

    # Salvando os arquivos com compressão para reduzir tamanho
    joblib.dump(clf, modelo_path, compress=3)
    joblib.dump(mlb, mlb_path)
    joblib.dump(le, le_path)

    print("✅ Modelo treinado e salvo em data/ com sucesso.")

# Execução principal
if __name__ == "__main__":
    historico = carregar_historico()
    treinar_modelo(historico)
