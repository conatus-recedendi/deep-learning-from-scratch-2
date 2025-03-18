# coding: utf-8
import sys

sys.path.append("..")
from common import config
import numpy as np
import matplotlib.pyplot as plt
from dataset import sequence
from common.optimizer import Adam
from common.trainer import Trainer
from common.util import eval_seq2seq, cast_to_single_value
from seq2seq import Seq2seq
from peeky_seq2seq import PeekySeq2seq
import wandb

config.GPU = True

wandb.init(
    project="RNN",
    name="seq2seq",
    config={
        "seed": {"value": 1000},
        "gradient_descent": {"value": "SGD"},
        "learning_rate": {"value": 5.0},
        "epochs": {"value": 25},
        "batch_size": 128,
        "model": {"value": "seq2seq"},
        "max_grad": {"value": 0.25},
        "is_reverse": {"value": False},
        "model_params": {
            "value": {
                "hidden_size": 128,
                "wordvec_size": 16,
            },
        },
        "dataset": {"value": "addition"},
        "gpu": {"value": True},
        "baseline": True,
    },
)


# 데이터셋 읽기
(x_train, t_train), (x_test, t_test) = sequence.load_data("addition.txt")
char_to_id, id_to_char = sequence.get_vocab()

# 입력 반전 여부 설정 =============================================
is_reverse = wandb.config.is_reverse  # True
if is_reverse:
    x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]
# ================================================================


# 하이퍼파라미터 설정
vocab_size = len(char_to_id)

# 일반 혹은 엿보기(Peeky) 설정 =====================================
model = Seq2seq(
    vocab_size,
    wandb.config.model_params["wordvec_size"],
    wandb.config.model_params["hidden_size"],
)
# model = PeekySeq2seq(vocab_size, wordvec_size, hidden_size)
# ================================================================
optimizer = Adam()
trainer = Trainer(model, optimizer)

acc_list = []
for epoch in range(wandb.config.epochs):
    trainer.fit(
        x_train,
        t_train,
        max_epoch=1,
        batch_size=wandb.config.batch_size,
        max_grad=wandb.config.max_grad,
    )

    correct_num = 0
    for i in range(len(x_test)):
        question, correct = x_test[[i]], t_test[[i]]
        verbose = i < 10
        correct_num += eval_seq2seq(
            model, question, correct, id_to_char, verbose, is_reverse
        )

    acc = float(correct_num) / len(x_test)
    acc_list.append(acc)
    wandb.log(
        {
            "Test Accuracy": cast_to_single_value(acc),
        }
    )
    print("검증 정확도 %.3f%%" % (acc * 100))

# 그래프 그리기
x = np.arange(len(acc_list))
plt.plot(x, acc_list, marker="o")
plt.xlabel("에폭")
plt.ylabel("정확도")
plt.ylim(0, 1.0)
plt.show()
