import random
from pathlib import Path


def generate_addition_dataset(size: int, filename: str):
    assert size <= 1_000_000, "size는 1,000,000 이하이어야 합니다 (중복 없이 생성 가능)"

    # 모든 가능한 (a, b) 쌍 생성
    all_pairs = [(a, b) for a in range(1000) for b in range(1000)]

    # 무작위로 섞고, size만큼 추출
    random.shuffle(all_pairs)
    sampled_pairs = all_pairs[:size]

    def format_addition(a, b):
        result = a + b
        addition = f"{a}+{b}"
        return f"{addition:<7}_{result:<5}"  # 자리수 맞춤 및 구분자 '_'

    with open(filename, "w") as f:
        for a, b in sampled_pairs:
            f.write(format_addition(a, b) + "\n")


# 사이즈별 파일 생성
output_dir = Path("./")
output_dir.mkdir(exist_ok=True)

sizes = {
    "100K": 100_000,
    "250K": 250_000,
    "500K": 500_000,
    "1M": 1_000_000,
    "1K": 1_000,
}

for label, size in sizes.items():
    generate_addition_dataset(size, output_dir / f"addition_{label}.txt")
