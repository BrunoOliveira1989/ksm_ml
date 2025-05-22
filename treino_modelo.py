# treino_modelo.py

from sqlalchemy import create_engine, text
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from pathlib import Path
import joblib

# Conexão com o banco de dados
DATABASE_URL = (
    "postgresql+psycopg2://kodiak_pocket_owner:"
    "k0rm1fPEwAyU@ep-long-surf-a5z0iq90-pooler."
    "us-east-2.aws.neon.tech/ksm?sslmode=require"
)
engine = create_engine(DATABASE_URL)

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
    print(f"Shape de X_encoded: {X_encoded.shape}")
    print(f"Shape de y_encoded: {y_encoded.shape}")

    # Treinamento do modelo com limitação de profundidade para evitar arquivos enormes
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,  # evita árvores profundas demais
        random_state=42
    )
    clf.fit(X_encoded, y_encoded)

    # Caminhos para salvar os arquivos
    BASE_DIR = Path(__file__).parent.resolve()
    modelo_path = BASE_DIR / "modelo_treinado.pkl"
    mlb_path = BASE_DIR / "mlb.pkl"
    le_path = BASE_DIR / "le.pkl"

    # Salvando os arquivos com compressão para reduzir tamanho
    joblib.dump(clf, modelo_path, compress=3)
    joblib.dump(mlb, mlb_path)
    joblib.dump(le, le_path)

    print("✅ Modelo treinado e salvo com sucesso.")

# Execução principal
if __name__ == "__main__":
    historico = carregar_historico()
    treinar_modelo(historico)
