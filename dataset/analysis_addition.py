import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
from pathlib import Path


# 여러 파일의 자릿수 분포 비교
def analyze_multiple_files_digit_lengths(filepaths):
    results = []

    for filepath in filepaths:
        with open(filepath, "r") as f:
            lines = f.readlines()

        def get_input_digit_lengths(line):
            input_part = line.split("_")[0]
            a_str, b_str = input_part.split("+")
            a_digits = len(a_str.strip())
            b_digits = len(b_str.strip())
            return (a_digits, b_digits)

        length_pairs = [get_input_digit_lengths(line) for line in lines]
        counter = Counter(length_pairs)

        for pair, count in counter.items():
            results.append(
                {
                    "File": filepath.name,
                    "a_digits": pair[0],
                    "b_digits": pair[1],
                    "Count": count,
                }
            )

    return pd.DataFrame(results)


output_dir = Path("./")
# 모든 파일 경로
all_files = [
    output_dir / "addition.txt",
    output_dir / "addition_100K.txt",
    output_dir / "addition_250K.txt",
    output_dir / "addition_500K.txt",
    output_dir / "addition_1M.txt",
]

# 분석 및 결과 정리
df_all = analyze_multiple_files_digit_lengths(all_files)
pivot_df = df_all.pivot_table(
    index=["a_digits", "b_digits"], columns="File", values="Count", fill_value=0
)
pivot_df = pivot_df.sort_index()

# 시각화를 위해 각 조합을 문자열로 변환
df_all["Digit Pair"] = df_all.apply(
    lambda row: f"{row['a_digits']},{row['b_digits']}", axis=1
)

# 막대그래프 시각화
plt.figure(figsize=(14, 6))
bar_width = 0.2
digit_pairs = sorted(df_all["Digit Pair"].unique())
x = range(len(digit_pairs))

# 각 파일에 대해 막대 위치를 조금씩 옮겨 그리기
for i, file in enumerate(df_all["File"].unique()):
    subset = (
        df_all[df_all["File"] == file]
        .set_index("Digit Pair")
        .reindex(digit_pairs)
        .fillna(0)
    )
    counts = subset["Count"].values
    plt.bar([pos + i * bar_width for pos in x], counts, width=bar_width, label=file)

plt.title("Digit Length Pair Distribution (Bar Chart)")
plt.xlabel("Digit Pair (a_digits, b_digits)")
plt.ylabel("Count")
plt.xticks([pos + 1.5 * bar_width for pos in x], digit_pairs, rotation=45)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()
