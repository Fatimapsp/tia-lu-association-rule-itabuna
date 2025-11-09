import numpy as np

class FPNode:
    """
    Esta classe define a estrutura de um Nó na FP-Tree.
    É a "peça" fundamental da nossa árvore.
    """

    def __init__(self, item, count, parent):
        self.item = item  # O nome do item
        self.count = count  # A contagem de vezes que este item apareceu "neste caminho"
        self.parent = parent  # Uma referência ao nó pai
        self.children = {}  # Um dicionário de filhos {nome_item: FPNode}

        # O node_link é crucial para a mineração. Ele aponta para o próximo
        # nó com o MESMO nome de item (ex: todos os nós 'Camisa').
        self.node_link = None

    def increment_count(self, count):
        """Um método simples para somar à contagem do nó."""
        self.count += count


def update_header_table(item, target_node, header_table):
    """
    Função auxiliar para atualizar o 'node_link' na header_table.
    Ela "anexa" o target_node no final da lista ligada.
    """
    header_entry = header_table.get(item)
    if header_entry is None:
        return  # Item não está na header table (acontece em árvores condicionais)

    if header_entry[1] is None:
        header_entry[1] = target_node
    else:
        current_node = header_entry[1]
        while current_node.node_link is not None:
            current_node = current_node.node_link
        current_node.node_link = target_node


def insert_tree(transaction, node, header_table, count_to_add):
    """
    Insere recursivamente uma transação (lista de itens) na árvore.
    """
    first_item = transaction[0]

    if first_item in node.children:
        child_node = node.children[first_item]
        child_node.increment_count(count_to_add)  # Modificado
    else:
        child_node = FPNode(item=first_item, count=count_to_add, parent=node)  # Modificado
        node.children[first_item] = child_node
        update_header_table(first_item, child_node, header_table)

    remaining_transaction = transaction[1:]
    if len(remaining_transaction) > 0:
        insert_tree(remaining_transaction, child_node, header_table, count_to_add)  # Modificado


def build_fp_tree(transacoes, header_table, min_support_count):
    """
    Orquestra a construção da árvore (A Segunda Varredura).
    """
    print(f"\n--- Construindo FP-Tree ---")

    root_node = FPNode('root', 1, None)


    if len(transacoes) == 0:
        print("Base de padrão condicional vazia. Encerrando esta ramificação.")
        return root_node


    is_conditional_tree = isinstance(transacoes[0], tuple)

    if is_conditional_tree:
        print(f"Construindo árvore condicional...")

        # 1. Contar itens na base condicional
        cond_item_counts = {}
        for path, count in transacoes:
            for item in path:
                cond_item_counts[item] = cond_item_counts.get(item, 0) + count

        # 2. Filtrar por min_support
        cond_frequent_counts = {item: count for item, count in cond_item_counts.items() if count >= min_support_count}

        # 3. Criar a header_table condicional (passada como argumento 'header_table')
        header_table.clear()
        for item, count in cond_frequent_counts.items():
            header_table[item] = [count, None]

        # 4. Inserir na árvore
        for path, count in transacoes:
            # Filtra e ordena o caminho com base nas *novas* contagens
            filtered_path = [item for item in path if item in header_table]
            sorted_path = sorted(filtered_path, key=lambda item: header_table[item][0], reverse=True)

            if len(sorted_path) > 0:
                insert_tree(sorted_path, root_node, header_table, count)

    else:
        #Construção da Árvore Principal:
        print(f"Construindo árvore principal...")
        for transacao in transacoes:
            filtered_tx = [item for item in transacao if item in header_table]
            sorted_tx = sorted(filtered_tx, key=lambda item: header_table[item][0], reverse=True)

            if len(sorted_tx) > 0:
                insert_tree(sorted_tx, root_node, header_table, 1)

    print("Construção da Árvore concluída.")
    return root_node


# Mineração da Árvore FP-Tree (Recursão)

def ascend_tree(node):
    """
    Sobe na árvore a partir de um nó para encontrar seu caminho (prefixo).
    """
    path = []
    while node.parent is not None and node.parent.item != 'root':
        node = node.parent
        path.append(node.item)
    return path

def find_conditional_pattern_base(item, header_table):
    """
    Encontra todos os caminhos (prefixos) que terminam no 'item'.
    """
    conditional_patterns = []

    # 1. Encontra o primeiro nó na lista ligada
    current_node = header_table[item][1]

    # 2. Itera por todos os nós com o nome 'item'
    while current_node is not None:
        # 3. Sobe na árvore para pegar o caminho
        prefix_path = ascend_tree(current_node)

        if len(prefix_path) > 0:
            # Adiciona o (caminho, contagem_do_nó)
            conditional_patterns.append((prefix_path, current_node.count))

        # 4. Pula para o próximo nó na lista ligada
        current_node = current_node.node_link

    return conditional_patterns


def mine_fp_tree(header_table, min_support_count, current_prefix, all_frequent_itemsets):
    """
    A principal função recursiva de mineração.
    header_table: A tabela da árvore (ou mini-árvore) atual.
    min_support_count: O limiar de contagem (ex: 50).
    current_prefix: O itemset que estamos "condicionando"
    all_frequent_itemsets: O dict global para armazenar os resultados.
    """

    # 1. Ordena os itens do menos frequente para o mais frequente
    sorted_items = sorted(header_table.items(), key=lambda x: x[1][0])

    # 2. Itera por cada item (de baixo para cima)
    for item, data in sorted_items:
        count = data[0]

        # 3. Gera o novo itemset frequente
        new_itemset = current_prefix.copy()
        new_itemset.add(item)

        # 4. Salva o resultado
        all_frequent_itemsets[frozenset(new_itemset)] = count

        # 5. Encontra a Base de Padrão Condicional para o 'item'
        cond_pattern_base = find_conditional_pattern_base(item, header_table)

        # 6. Cria a Header Table da Árvore Condicional
        cond_header_table = {}

        # 7. Constrói a Árvore FP Condicional
        cond_tree_root = build_fp_tree(cond_pattern_base, cond_header_table, min_support_count)

        # 8. Recursão: Minera a Árvore Condicional
        if len(cond_header_table) > 0:
            # Se a mini-árvore tiver itens, minere-a
            # O novo prefixo é o itemset que acabamos de encontrar
            mine_fp_tree(cond_header_table, min_support_count, new_itemset, all_frequent_itemsets)
