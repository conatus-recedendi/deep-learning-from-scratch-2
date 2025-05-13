# coding: utf-8
import sys

sys.path.append("..")
from common import config

config.GPU = True
import matplotlib.pyplot as plt
from common.np import *
from dataset import sequence
from common.optimizer import Adam
from common.trainer import Trainer
from common.util import eval_seq2seq, cast_to_single_value, to_gpu
from seq2seq import Seq2seq
from peeky_seq2seq import PeekySeq2seq
import wandb


def run():
    wandb.init()

    # 데이터셋 읽기
    (x_train, t_train), (x_test, t_test) = sequence.load_data("addition_100K.txt")
    char_to_id, id_to_char = sequence.get_vocab()

    by_digit = wandb.config.by_digit
    # 입력 반전 여부 설정 =============================================
    is_reverse = wandb.config.is_reverse  # True
    if is_reverse:
        x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]
    # ================================================================

    if config.GPU:
        x_train, t_train = to_gpu(x_train), to_gpu(t_train)
        x_test, t_test = to_gpu(x_test), to_gpu(t_test)

    # 하이퍼파라미터 설정
    vocab_size = len(char_to_id)

    # 일반 혹은 엿보기(Peeky) 설정 =====================================
    if wandb.config.is_peeky:
        model = PeekySeq2seq(
            vocab_size,
            wandb.config.model_params["wordvec_size"],
            wandb.config.model_params["hidden_size"],
        )
    else:
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
            print("test acc" + str(i))
            question, correct = x_test[[i]], t_test[[i]]
            verbose = i < 10
            correct_num += eval_seq2seq(
                model,
                question,
                correct,
                id_to_char,
                verbose,
                is_reverse,
                by_digit,
            )

        max_iters = len(x_test) // wandb.config.batch_size
        total_loss = 0
        total_count = 0
        for iters in range(max_iters):
            print("test loss" + str(iters))
            batch_x = x_test[
                iters * wandb.config.batch_size : (iters + 1) * wandb.config.batch_size
            ]
            batch_t = t_test[
                iters * wandb.config.batch_size : (iters + 1) * wandb.config.batch_size
            ]
            total_loss += model.forward(batch_x, batch_t)
            total_count += 1
        loss_test = total_loss / total_count

        acc = float(correct_num) / len(x_test)
        acc_list.append(acc)

        correct_num_train = 0
        for i in range(max(len(x_train), len(x_test))):
            print("train acc" + str(i))
            question, correct = x_train[[i]], t_train[[i]]
            verbose = i < 10
            correct_num_train += eval_seq2seq(
                model, question, correct, id_to_char, verbose, is_reverse
            )
        acc_train = float(correct_num_train) / len(x_train)

        wandb.log(
            {
                "Test Accuracy": cast_to_single_value(acc * 100),
                "Train Accuracy": cast_to_single_value(acc_train * 100),
                "Test Loss": cast_to_single_value(loss_test),
            }
        )
        print("검증 정확도 %.3f%%" % (acc * 100))


# 그래프 그리기
# x = np.arange(len(acc_list))
# plt.plot(x, acc_list, marker="o")
# plt.xlabel("에폭")
# plt.ylabel("정확도")
# plt.ylim(0, 1.0)
# plt.show()


wandb_sweep_config = {
    "name": "seq2seq",
    "method": "grid",
    "metric": {"name": "train_loss", "goal": "minimize"},
    "parameters": {
        "seed": {"values": [1000, 2000, 3000]},
        # "seed": {"value": 1000},
        "gradient_descent": {"value": "SGD"},
        "learning_rate": {"value": 5.0},
        "epochs": {"value": 100},
        "batch_size": {"value": 128},
        "model": {"value": "seq2seq"},
        "max_grad": {"value": 0.25},
        "is_reverse": {"value": False},
        "is_peeky": {"value": False},
        "by_digit": {"values": [False, True]},
        "model_params": {
            "values": [
                {"hidden_size": 128, "wordvec_size": 16},
                # {"hidden_size": 100, "window_size": 2},
                # {"hidden_size": 100, "window_size": 3},
            ]
        },
        "gpu": {"value": config.GPU},
        "dataset": {
            "values": [
                "addition_100K.txt",
                # "addition_250K.txt",
                # "addition_500K.txt",
                # "addition_1M.txt",
            ]
        },
        "baseline": {"value": False},
        # "batch_norm": {"value": False},
        # "weight_decay_lambda": {"value": 0},
        # "dataset": {"value": ""},
        # "activation": {"value": "relu"},
        # "weight_init_std": {"value": "he"},
        # "dropout": {"value": 0.15},
    },
}

sweep_id = wandb.sweep(sweep=wandb_sweep_config, project="RNN")

wandb.agent(sweep_id, function=run)
