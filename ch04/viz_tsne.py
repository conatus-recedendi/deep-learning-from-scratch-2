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
def visualize_tsne(
    pkl_file,
    output_file="cbow_tsne.png",
    query_words=[],
    analogy_words=[],
    analogy_word_keys=[],
    analogy_word_keys_answer=[]
    seed=1000,
):
    params = load_model(pkl_file)
    word_vecs = params["word_vecs"]
    id_to_word = params["id_to_word"]
    word_to_id = params["word_to_id"]
    np.random.seed(seed)

    # 단어 벡터 추출
    query_indices = [word_to_id[word] for word in query_words if word in word_to_id]
    analogy_indices = [word_to_id[word] for word in analogy_words if word in word_to_id]
    key_indices = [
        word_to_id[word]
        for pair in analogy_word_keys
        for word in pair
        if word in word_to_id
    ]

    key_indices_answer = [
        word_to_id[word]
        for pair in analogy_word_keys_answer
        for word in pair
        if word in word_to_id
    ]

    all_indices = query_indices + analogy_indices + key_indices + key_indices_answer
    selected_vectors = word_vecs[all_indices]
    selected_labels = [id_to_word[i] for i in all_indices]

    # 색상 지정
    selected_colors = (
        ["blue"] * len(query_indices)
        + ["red"] * len(analogy_indices)
        + ["green"] * len(key_indices)
        + ["purple"] * len(key_indices_answer)
    )

    # t-SNE 적용
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    reduced_vectors = tsne.fit_transform(selected_vectors)

    # 시각화
    plt.figure(figsize=(12, 8))
    for i, label in enumerate(selected_labels):
        plt.scatter(
            reduced_vectors[i, 0],
            reduced_vectors[i, 1],
            color=selected_colors[i],
            alpha=0.7,
        )
        plt.annotate(
            label, (reduced_vectors[i, 0], reduced_vectors[i, 1]), fontsize=9, alpha=0.7
        )

    plt.title("t-SNE Word Vector Visualization")
    plt.savefig(output_file, dpi=300)
    plt.show()


query_words = [
    "you",
    "year",
    "car",
    "toyota",
    "yourself",
    "i",
    "we",
    "anything",
    "anybody",
    "your",
    "weird",
    "somebody",
    "earlier",
    "month",
    "quarter",
    "week",
    "summer",
    "spring",
    "decade",
    "period",
    "luxury",
    "cars",
    "auto",
    "mazda",
    "truck",
    "motor",
    "honda",
    "nissan",
    "infiniti",
    "lexus",
]

analogy_word_keys = [
    ["king", "man"],
    ["take", "took"],
    ["car", "cars"],
    ["good", "better"],
]

analogy_word_keys_answer = [
    ["queen", "woman"],
    ["go", "went"],
    ["child", "children"],
    ["bad", "worse"],
]

analogy_words = [
    "peace",
    "freeway",
    "bikers",
    "teacher",
    "born",
    "a.m",
    "fbi",
    "brother",
    "answers",
    "pricings",
    "non-u.s.",
    "eurodollars",
    "'re",
    "were",
    "went",
    "was",
    "feet",
    "yards",
    "ran",
    "rape",
    "incest",
    "daffynition",
    "bond-equivalent",
    "non-violent",
    "pregnant",
    "adjustable",
    "always",
    "rather",
    "more",
    "less",
    "greater",
    "faster",
    "impressive",
    "swings",
    "vary",
    "ever",
]

# 사용 예시 (파일명 변경 필요)
pkl_file = "cbow_params.pkl"  # 또는 'skipgram_params.pkl'
visualize_tsne(
    pkl_file,
    "cbow_tsne.png",
    query_words=[],
    analogy_words=analogy_words,
    analogy_word_keys=analogy_word_keys,
    analogy_word_keys_answer=analogy_word_keys_answer,
)
pkl_file = "skip_gram_params.pkl"
visualize_tsne(
    pkl_file,
    "skip_gram_tsne.png",
    query_words=[],
    analogy_words=analogy_words,
    analogy_word_keys=analogy_word_keys,
    analogy_word_keys_answer=analogy_word_keys_answer,
)
