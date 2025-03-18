import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# 모델 로드
def load_model(pkl_file):
    with open(pkl_file, "rb") as f:
        params = pickle.load(f)
    return params


# t-SNE 변환 및 시각화
def visualize_tsne(pkl_file, num_words=200):
    params = load_model(pkl_file)
    word_vecs = params["word_vecs"]
    id_to_word = params["id_to_word"]

    # 일부 단어만 선택 (너무 많으면 시각화가 어려움)
    selected_indices = np.random.choice(len(word_vecs), num_words, replace=False)
    selected_vectors = word_vecs[selected_indices]
    selected_labels = [id_to_word[i] for i in selected_indices]

    # t-SNE 적용
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced_vectors = tsne.fit_transform(selected_vectors)

    # 시각화
    plt.figure(figsize=(12, 8))
    plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], alpha=0.7)

    # 단어 라벨 추가
    for i, label in enumerate(selected_labels):
        plt.annotate(
            label, (reduced_vectors[i, 0], reduced_vectors[i, 1]), fontsize=9, alpha=0.7
        )

    plt.title("t-SNE Word Vector Visualization")
    plt.show()


# 사용 예시 (파일명 변경 필요)
pkl_file = "cbow_params.pkl"  # 또는 'skipgram_params.pkl'
visualize_tsne(pkl_file)
