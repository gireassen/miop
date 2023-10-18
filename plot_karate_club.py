import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

'''
Этот код выполняет две основные задачи:
1. он рисует сетевой граф каратейского клуба Захари;
2. затем он перекрашивает узлы в этом графе в зависимости от того, к какому клубу принадлежит каждый узел,
и снова рисует граф, уже с перекрашенными узлами.
Вот более подробное объяснение:
- Библиотека `networkx` используется для создания и манипулирования структурами данных графа,
    в то время как библиотека `matplotlib.pyplot` используется для визуализации графов.
- Функция `nx.karate_club_graph()` возвращает граф социальной сети каратейского клуба "Захари",
    который был сформирован в 1977 году Уэйном Захари (Wayne W. Zachary),
    когда он проводил антропологическое исследование 34 членов каратейского клуба,
    которые были разделены во время спора между администратором клуба "Mr. Hi" и инструктором занятий.
- Граф затем визуализируется с помощью `nx.draw()`, а свойства `with_labels=True`, `node_color='lightblue'`,
    и `edge_color='gray'` добавляются для упрощения чтения графа.
- Функция `set_club_colors(G)` затем проходит через каждый узел в графе и устанавливает цвет узла в зависимости от клуба,
    к которому он принадлежит, присваивая "lightblue" узлам, принадлежащим "Mr. Hi" и "pink" всем остальным узлам. 
- Граф затем отображается снова, но на этот раз с узлами, окрашенными в соответствии с их принадлежностью к клубу.
Код использует `nx.spring_layout(G)` для автоматического определения позиций узлов во время визуализации,
так что узлы с множеством связей будут более центрированы, а менее связанные узлы - периферийны.
'''

G = nx.karate_club_graph()

# Рисуем граф Karate Club
plt.figure(figsize=(8, 6))
nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray')
plt.title('Захариев клуб карате')
plt.show()
def set_club_colors(G):
    for each in G.nodes():
        if G.nodes[each]['club'] == 'Mr. Hi':
            G.nodes[each]['color'] = 'lightblue'
        else:
            G.nodes[each]['color'] = 'red'
    return G

G = set_club_colors(G)

colors = [node[1]['color'] for node in G.nodes(data=True)]
pos = nx.spring_layout(G)

plt.figure(figsize=(8, 6))
nx.draw(G, pos, node_color=colors, with_labels=True, edge_color='gray')
plt.title('Фракции Захариева клуба карате')
plt.show()

def create_adjacency_table(G):
    adjacency_matrix = nx.adjacency_matrix(G)
    adjacency_df = pd.DataFrame(adjacency_matrix.todense(), index=G.nodes(), columns=G.nodes())
    return adjacency_df

adjacency_df = create_adjacency_table(G)
print(adjacency_df)
