import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score

def generateTsne(embeddings,optimal_k,iteration,texts):
    # Выбор оптимального числа кластеров (например, k=4)
    kmeans = KMeans(n_clusters=optimal_k, random_state=42,algorithm="elkan",init='k-means++')
    clusters = kmeans.fit_predict(embeddings)

    colors = cm.tab10.colors
    clusters_colors = [colors[i% len(colors)] for i in clusters]

    # Уменьшение размерности для визуализации (PCA или t-SNE)
    tsne = TSNE(n_components=2)
    embeddings_2d = tsne.fit_transform(embeddings)

    inertia = []
    silhouette_scores = []
    K_range = range(2, 10)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(embeddings)
        inertia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(embeddings, kmeans.labels_))

    # Визуализация метода локтя
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(K_range, inertia, marker='o')
    plt.title('Метод локтя')
    plt.xlabel('Количество кластеров')
    plt.ylabel('Inertia')

    plt.subplot(1, 2, 2)
    plt.plot(K_range, silhouette_scores, marker='o')
    plt.title('Silhouette Score')
    plt.xlabel('Количество кластеров')
    plt.ylabel('Score')
    plt.show()


    # Визуализация кластеров
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        embeddings_2d[:, 0], 
        embeddings_2d[:, 1], 
        c=clusters_colors,  # Используем фиксированные цвета
        alpha=0.7,
        edgecolors='w',
        linewidths=0.5
    )

    # Добавляем легенду
    unique_clusters = np.unique(clusters)
    legend_elements = [
        plt.Line2D([0], [0], 
        marker='o', 
        color='w', 
        label=f'Cluster {i}',
        markerfacecolor=colors[i % len(colors)], 
        markersize=10) for i in unique_clusters
    ]

    plt.legend(
        handles=legend_elements, 
        title='Clusters',
        bbox_to_anchor=(1.05, 1),
        loc='upper left'
    )

    plt.title('Кластеризация отзывов (T-SNE)')
    plt.xlabel('Компонента 1')
    plt.ylabel('Компонента 2')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    index_map = {value: idx for idx, value in enumerate(texts)}
    # Примеры текстов из каждого кластера
    for cluster_id in range(optimal_k):
        cluster_texts = {"texts":[text for text, cluster in zip(texts, clusters) if cluster == cluster_id]}
        cluster_ids = {"ids":[index_map.get(item) for item in cluster_texts['texts']]}
        cluster_results = {**cluster_texts,**cluster_ids}
        with open(f'clusters/cluster{iteration}{cluster_id}.json', 'w', encoding='utf-8') as cluster:
            json.dump(cluster_results, cluster, ensure_ascii=False)



# Загрузка данных
with open('data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
with open('embeddings.json', 'r', encoding='utf-8') as fi:
    dataEmbeddings = json.load(fi)

embeddings= np.array(data['array'])
texts = dataEmbeddings['documents']
# Генерация эмбеддингов
#model = SentenceTransformer('all-MiniLM-L6-v2')
#embeddings = model.encode(texts, show_progress_bar=True)

# Определение оптимального числа кластеров (метод локтя)

iteration = 0
k = 2
generateTsne(embeddings,k,iteration,texts)
for i in range(k):
    with open(f'clusters/cluster0{i}.json', 'r', encoding='utf-8') as clusterFirst:
        clusterIteration = json.load(clusterFirst)
    clusterIterationIds = clusterIteration['ids']
    clusterIterationTexts = clusterIteration['texts']
    embeddingsIteration = embeddings[clusterIterationIds]
    if i == 0:
        generateTsne(embeddingsIteration,2,i+1,clusterIterationTexts)
    else:
        generateTsne(embeddingsIteration,4,i+1,clusterIterationTexts)
