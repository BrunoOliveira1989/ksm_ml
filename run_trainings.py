# run_trainings.py

from training.treino_chance_de_compra import carregar_dados_clientes, treinar_modelo as treinar_chance
from training.treino_sugestao_de_produto import carregar_historico, treinar_modelo as treinar_sugestao

def executar():
    print("\n=== Treinando modelo de CHANCE DE COMPRA ===")
    X, y, _ = carregar_dados_clientes()
    treinar_chance(X, y)

    print("\n=== Treinando modelo de SUGESTÃO DE PRODUTO ===")
    historico = carregar_historico()
    treinar_sugestao(historico)

    print("\n✅ Todos os modelos foram treinados com sucesso.")

if __name__ == "__main__":
    executar()
