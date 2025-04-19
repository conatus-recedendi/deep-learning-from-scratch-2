# 코드 재실행: cbow 유추 평가 함수 정의
import os, sys

sys.path.append("../")
import csv
import pickle
import numpy as np
from typing import List, Tuple
from dataset.ptb import load_data


def load_model(pkl_file: str):
    """cbow 모델 파라미터 불러오기"""
    with open(pkl_file, "rb") as f:
        params = pickle.load(f)
    # corpus, word_to_id, id_to_word = load_data("train")
    word_vecs = params["word_vecs"]
    word_to_id = params["word_to_id"]
    id_to_word = params["id_to_word"]
    return word_vecs, word_to_id, id_to_word


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
    print("start")
    word_vecs, word_to_id, id_to_word = load_model(model_path)
    print("end load_model")
    total_correct = 0
    total_count = 0

    syntactic_correct = 0
    syntactic_count = 0

    semantic_correct = 0
    semantic_count = 0

    i = 0
    with open(csv_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)

        for row in reader:
            i += 1
            print(i)
            if len(row) < 6:
                continue

            category = row[1]
            word_a, word_b, word_c, word_d = row[2], row[3], row[4], row[5]

            if (
                word_a not in word_to_id
                or word_b not in word_to_id
                or word_c not in word_to_id
                or word_d not in word_to_id
            ):
                total_count += 1
                if category.startswith("gram"):
                    syntactic_count += 1
                else:
                    semantic_count += 1
                continue

            vec_a = word_vecs[word_to_id[word_a]]
            vec_b = word_vecs[word_to_id[word_b]]
            vec_c = word_vecs[word_to_id[word_c]]

            analogy_vec = vec_b - vec_a + vec_c
            top_k_words = find_top_k_similar(analogy_vec, word_vecs, id_to_word, top_k)
            predicted_words = [w for w, _ in top_k_words]

            is_correct = word_d in predicted_words

            # 전체 정확도
            total_count += 1
            if is_correct:
                total_correct += 1

            # 범주 구분
            if category.startswith("gram"):
                syntactic_count += 1
                if is_correct:
                    syntactic_correct += 1
            else:
                semantic_count += 1
                if is_correct:
                    semantic_correct += 1

    # 출력
    print(f"모델 경로: {model_path}")
    print(
        f"총 평가 수: {total_count}, 정확도: {total_correct / total_count * 100:.2f}%"
        if total_count
        else "평가할 데이터 없음"
    )
    print(
        f"Syntactic: {syntactic_count}, 정확도: {syntactic_correct / syntactic_count * 100:.2f}%"
        if syntactic_count
        else "Syntactic 없음"
    )
    print(
        f"Semantic: {semantic_count}, 정확도: {semantic_correct / semantic_count * 100:.2f}%"
        if semantic_count
        else "Semantic 없음"
    )
    print("-" * 50)
    return (
        total_correct,
        total_count,
        syntactic_correct,
        syntactic_count,
        semantic_correct,
        semantic_count,
    )


# 사용 예시 (사용 전 모델과 CSV 경로를 지정):
evaluate_analogy(
    "../dataset/google-analogies.csv",
    "./skip_gram_params.pkl",
    top_k=5,
)
