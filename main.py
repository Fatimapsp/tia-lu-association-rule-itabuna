import pandas as pd
import numpy as np
import itertools
from fp_growth import build_fp_tree, mine_fp_tree

# Fase 1: Carregamento e Pré-Processamento

print(f"\n--- INÍCIO DA FASE 1: Pré-Processamento ---")
try:
    df = pd.read_csv('vendas_dataset.csv')
    print(f"Dados originais carregados. Total de {len(df)} transações.")

except FileNotFoundError:
    print("ERRO: O arquivo 'vendas_dataset.csv' não foi encontrado.")
    exit()

# 2. LIMPEZA (Tratando o Achado Crítico: Dados Ausentes)
df_limpo = df.dropna(subset=['descricao_produtos'])
print(f"Removendo transações com dados nulos. \nTotal agora: {len(df_limpo)} transações.")
print("---")

# 3. TRANSFORMAÇÃO (Tratando o Achado Crítico: Split por ';')
# Vamos criar uma nova coluna 'itens' que conterá uma LISTA de strings.
nova_coluna_de_listas = df_limpo['descricao_produtos'].apply(lambda x: [item.strip() for item in x.split(';')])

# 4. CRIAÇÃO DA LISTA DE TRANSAÇÕES FINAL
# É uma lista de listas
transacoes_prontas = nova_coluna_de_listas.tolist()

#Verificação:
print("Verificação: As 5 primeiras transações pré-processadas:")
for i in range(5):
    print(f"Transação {i+1}: {transacoes_prontas[i]}")

print("---")
print("Pré-processamento concluído. Os dados estão prontos para a Fase de Mineração.")

#Fase 2: Mineração FP-Growth (Parte 1: Primeira Varredura)

# 1. Definir o limiar de suporte
min_support_percent = 0.01
total_transacoes = len(transacoes_prontas)

min_support_count = np.ceil(total_transacoes * min_support_percent)

print(f"\n--- INÍCIO DA FASE 2: Mineração ---")
print(f"Configuração da Mineração:")
print(f"Total de transações: {total_transacoes}")
print(f"Suporte mínimo (percentual): {min_support_percent * 100}%")
print(f"Suporte mínimo (absoluto): {min_support_count} transações")
print("---")

# 2. Contar a frequência de todos os itens individuais (1ª Varredura)
item_counts = {}
for transacao in transacoes_prontas:
    for item in transacao:
        # .get(item, 0) é uma forma segura de pegar a contagem.
        # Se o 'item' não existe no dict, ele retorna 0, e então somamos 1.
        item_counts[item] = item_counts.get(item, 0) + 1

print(f"Primeira Varredura: {len(item_counts)} itens únicos encontrados no total.")

# 3. Filtrar itens infrequentes
frequent_items_counts = {item: count for item, count in item_counts.items() if count >= min_support_count}
print(f"Filtragem: {len(frequent_items_counts)} itens frequentes retidos (>= {min_support_count} vendas).")

# 4. Criar a "Header Table" e a Lista Ordenada de Itens
# O FP-Growth exige que os itens sejam processados em ordem de frequência (do maior para o menor).

# A. Lista Ordenada:
sorted_frequent_tuples = sorted(frequent_items_counts.items(), key=lambda x: x[1], reverse=True)

# B. Header Table (Dicionário):
header_table = {item: [count, None] for item, count in frequent_items_counts.items()}

print("---")
print("Header Table (Top 10 Itens Mais Frequentes):")
for item, count in sorted_frequent_tuples[:10]:
    print(f"  {item}: {count}")

print("\nFase 2.1 (Primeira Varredura) concluída.")
print("Próximo passo: Fase 2.2 (Construção da Árvore FP-Tree).")


fp_tree_root = build_fp_tree(transacoes_prontas, header_table, min_support_count)

print(f"\n--- INÍCIO DA FASE 2.3: Mineração da FP-Tree ---")

# 1. Prepara o dict para guardar os resultados
all_frequent_itemsets = {}

# 2. Prepara o prefixo inicial (vazio)
initial_prefix = set()

# 3. Inicia a mineração!
mine_fp_tree(header_table, min_support_count, initial_prefix, all_frequent_itemsets)

print("Mineração da Árvore concluída.")
print(f"Total de {len(all_frequent_itemsets)} itemsets frequentes encontrados (com suporte >= {min_support_count}).")
print("---")

# Inspeciona os resultados
print("Top 15 Itemsets Frequentes Encontrados (por suporte):")
# Ordena os resultados pela contagem
sorted_itemsets = sorted(all_frequent_itemsets.items(), key=lambda x: x[1], reverse=True)

for itemset, count in sorted_itemsets[:15]:
    # list(itemset) transforma o frozenset de volta em lista para impressão
    print(f"  {list(itemset)}: {count}")

print("\nFase 2 (FP-Growth) concluída.")
print("Próximo passo: Fase 3 (Geração das Regras de Associação).")

# Fase 3: Regras de Associação

print(f"\n--- INÍCIO DA FASE 3: Geração de Regras ---")


# 50% (0.5) - queremos regras que "funcionem" pelo menos metade das vezes.
min_confidence = 0.5

print(f"Lógica de Geração:")
print(f"Iterando sobre {len(all_frequent_itemsets)} itemsets frequentes.")
print(f"Mantendo apenas regras com Confiança >= {min_confidence * 100}%")
print(f"Total de transações para cálculo do Lift: {total_transacoes}")
print("---")

final_rules = []

for itemset, support_count_itemset in all_frequent_itemsets.items():

    if len(itemset) < 2:
        continue

    for k in range(1, len(itemset)):

        antecedent_candidates = itertools.combinations(itemset, k)

        for ant_tuple in antecedent_candidates:
            antecedent = frozenset(ant_tuple)
            consequent = itemset - antecedent

            # --- Cálculo das Métricas ---

            # A. Confiança
            support_count_antecedent = all_frequent_itemsets[antecedent]

            confidence = support_count_itemset / support_count_antecedent

            # B. Filtrar por Confiança Mínima
            if confidence >= min_confidence:
                # C. Lift (Apenas para regras fortes)
                support_count_consequent = all_frequent_itemsets[consequent]

                support_percent_consequent = support_count_consequent / total_transacoes
                lift = confidence / support_percent_consequent

                final_rules.append((list(antecedent), list(consequent), confidence, lift, support_count_itemset))

print(f"Geração concluída. Total de {len(final_rules)} regras fortes encontradas.")
print("---")

# 4. Ordenar e Apresentar as Regras Finais
sorted_rules = sorted(final_rules, key=lambda x: (x[3], x[2]), reverse=True)  # Ordena por Lift, depois Confiança

print(
    f"Top Regras de Associação (Combos) Encontradas (Suporte >= {min_support_count}, Confiança >= {min_confidence * 100}%):")
print(f"{'Regra (Antecedente -> Consequente)':<80} | {'Confiança':<10} | {'Lift':<10} | {'Suporte (Vendas)':<10}")
print("-" * 120)

for ant, con, conf, lift, support in sorted_rules:
    # Formata a string para ficar legível
    regra_str = f"{str(ant):<38} -> {str(con):<38}"
    conf_str = f"{conf * 100:,.2f}%"
    lift_str = f"{lift:,.2f}"
    support_str = f"{support:,.0f}"

    print(f"{regra_str:<80} | {conf_str:<10} | {lift_str:<10} | {support_str:<10}")

if len(sorted_rules) == 0:
    print("Nenhuma regra de associação forte foi encontrada com os limiares de suporte (1%) e confiança (50%).")
    print("Justificativa (para o relatório): Isso indica que, embora 3 pares de itens sejam frequentes,")
    print("a confiança (força) da associação entre eles é fraca (menor que 50%).")
