# 코드 재실행: skip-gram 유추 평가 함수 정의
import os, sys

sys.path.append("../../")
import csv
import pickle
import numpy as np
from typing import List, Tuple
from dataset.ptb import load_data


def load_model(pkl_file: str):
    """Skip-gram 모델 파라미터 불러오기"""
    with open(pkl_file, "rb") as f:
        params = pickle.load(f)
    corpus, word_to_id, id_to_word = load_data("train")
    return params[0], word_to_id, id_to_word


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-8)


def find_top_k_similar(
    vec: np.ndarray, word_vecs: np.ndarray, id_to_word: dict, top_k: int
) -> List[Tuple[str, float]]:
    similarities = np.dot(word_vecs, vec) / (
        np.linalg.norm(word_vecs, axis=1) * np.linalg.norm(vec) + 1e-8
    )
    top_k_ids = similarities.argsort()[::-1][:top_k]
    return [(id_to_word[int(i)], similarities[i]) for i in top_k_ids]


def evaluate_analogy(csv_path: str, model_path: str, top_k: int = 5):
    word_vecs, word_to_id, id_to_word = load_model(model_path)

    correct = 0
    total = 0

    with open(csv_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) < 6:
                continue  # 데이터 부족 시 스킵
            word_a, word_b, word_c, word_d = row[2], row[3], row[4], row[5]

            if all(w in word_to_id for w in [word_a, word_b, word_c, word_d]):
                vec_a = word_vecs[word_to_id[word_a]]
                vec_b = word_vecs[word_to_id[word_b]]
                vec_c = word_vecs[word_to_id[word_c]]
                vec_d = word_vecs[word_to_id[word_d]]

                analogy_vec = vec_b - vec_a + vec_c
                top_k_words = find_top_k_similar(
                    analogy_vec, word_vecs, id_to_word, top_k
                )

                predicted_words = [w for w, _ in top_k_words]
                if word_d in predicted_words:
                    correct += 1
                total += 1

    print(f"총 평가 수: {total}")
    print(f"정답 수: {correct}")
    print(
        f"정확도: {correct / total * 100:.2f}%"
        if total > 0
        else "평가할 데이터가 없습니다."
    )


# 사용 예시 (사용 전 모델과 CSV 경로를 지정):
evaluate_analogy(
    "../dataset/google-analogies.csv",
    "./skip_gram_params.pkl",
    top_k=5,
)
