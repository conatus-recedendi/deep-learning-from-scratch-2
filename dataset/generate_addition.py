import random
from pathlib import Path


def generate_addition_dataset(size: int, filename: str):
    def random_addition():
        a = random.randint(0, 999)
        b = random.randint(0, 999)
        result = a + b
        addition = f"{a}+{b}"
        return f"{addition:<7}_{result:<5}"

    # 생성
    data = [random_addition() for _ in range(size)]

    # 저장
    with open(filename, "w") as f:
        for line in data:
            f.write(line + "\n")


# 사이즈별 파일 생성
output_dir = Path("./")
output_dir.mkdir(exist_ok=True)

sizes = {"100K": 100_000, "250K": 250_000, "500K": 500_000, "1M": 1_000_000}

for label, size in sizes.items():
    generate_addition_dataset(size, output_dir / f"addition_{label}.txt")

output_dir.listdir()
