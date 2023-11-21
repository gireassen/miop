import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

G = nx.karate_club_graph()
data = nx.to_numpy_array(G)

def apr_func(data):
    te = TransactionEncoder()
    data_encoded = te.fit_transform(data)
    data_encoded_df = pd.DataFrame(data_encoded, columns=te.columns_)
    min_support = 0.02
    frequent_itemsets = apriori(data_encoded_df, min_support=min_support, use_colnames=True)
    min_conf = 0.8
    rules = association_rules(frequent_itemsets, min_threshold=min_conf)
    fname = 'Karate'
    n_rules = 300
    rules[:n_rules].to_csv(fname+'.csv', index=False)

apr_func(data)

G = nx.Graph(data)
degree = G.degree()
matrix = np.zeros((34, 34))
for u, v in G.edges():
    if degree[u] > degree[v] or degree[u] < degree[v]:
        matrix[u][v] = 1
        matrix[v][u] = 1
for row in matrix:
    print(''.join(str(int(val)) for val in row))

print('\n')

def find_associations(data):
    n = len(data)
    associations = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if data[i][j] != 0:
                for k in range(n):
                    if data[j][k] != 0 and data[k][i] != 0:
                        associations[i][j] = 1
                        associations[j][k] = 1
                        associations[k][i] = 1
    return associations

found_associations = find_associations(data)
for row in found_associations:
    print(''.join(str(int(val)) for val in row))

G_associations = nx.from_numpy_array(found_associations)
for node in G_associations.nodes():
    G_associations.nodes[node]['club'] = 'Mr. Hi' if node < 17 else 'Others'
pos = nx.spring_layout(G_associations)

# Отрисовка графа
nx.draw(G_associations, pos, with_labels=True, node_color='lightblue', node_size=500, edge_color='gray')
plt.show()

def set_club_colors(G_associations):
    for each in G_associations.nodes():
        if G_associations.nodes[each]['club'] == 'Mr. Hi':
            G_associations.nodes[each]['color'] = 'lightblue'
        else:
            G_associations.nodes[each]['color'] = 'red'
    return G_associations

data_2 = nx.to_numpy_array(G)
G_associations = set_club_colors(G_associations)

colors = [node[1]['color'] for node in G_associations.nodes(data=True)]
pos = nx.spring_layout(G_associations)

plt.figure(figsize=(8, 6))
nx.draw(G_associations, pos, node_color=colors, with_labels=True, edge_color='gray')
plt.title('Фракции Захариева клуба карате')
plt.show()
