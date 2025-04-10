# coding: utf-8
import sys

sys.path.append("..")
from common import config

# GPU에서 실행하려면 아래 주석을 해제하세요(CuPy 필요).
# ==============================================
config.GPU = True
# ==============================================
from common.optimizer import SGD
from common.trainer import RnnlmTrainer
from common.util import eval_perplexity, to_gpu, cast_to_single_value
from dataset import ptb
from better_rnnlm import BetterRnnlm

import wandb

wandb.init(
    project="RNN",
    name="Better_RNNLM w/o weight_typing",
    config={
        "seed": {"value": 1000},
        "gradient_descent": {"value": "SGD"},
        "learning_rate": {"value": 20},
        "epochs": {"value": 40},
        "batch_size": 20,
        "model": {"value": "Better_RNNLM"},
        "max_grad": {"value": 0.25},
        "dropout": {"value": 0.5},
        "is_weight_typing": {"value": False},
        "model_params": {
            "value": {
                "hidden_size": 650,
                "wordvec_size": 650,
                "time_size": 35,
                "num_lstm_layer": 2,  # only 2, 4
            },
        },
        "dataset": {"value": "PTB"},
        "gpu": {"value": True},
        "baseline": True,
    },
)

# 학습 데이터 읽기
corpus, word_to_id, id_to_word = ptb.load_data("train")
corpus_val, _, _ = ptb.load_data("val")
corpus_test, _, _ = ptb.load_data("test")

if config.GPU:
    corpus = to_gpu(corpus)
    corpus_val = to_gpu(corpus_val)
    corpus_test = to_gpu(corpus_test)

vocab_size = len(word_to_id)
xs = corpus[:-1]
ts = corpus[1:]

model = BetterRnnlm(
    vocab_size,
    wandb.config.model_params["wordvec_size"],
    wandb.config.model_params["hidden_size"],
    wandb.config.dropout,
    is_weight_typing=wandb.config.is_weight_typing,
    num_lstm_layer=wandb.config.model_params["num_lstm_layer"],
)
optimizer = SGD(wandb.config.learning_rate)
trainer = RnnlmTrainer(model, optimizer)

best_ppl = float("inf")
lr = wandb.config.learning_rate
for epoch in range(wandb.config.epochs):
    trainer.fit(
        xs,
        ts,
        max_epoch=1,
        batch_size=wandb.config.batch_size,
        time_size=wandb.config.model_params["time_size"],
        max_grad=wandb.config.max_grad,
    )

    model.reset_state()
    ppl = eval_perplexity(model, corpus_val)
    print("검증 퍼플렉서티: ", ppl)
    wandb.log({"Validation Perplexity": cast_to_single_value(ppl)})

    if best_ppl > ppl:
        best_ppl = ppl
        model.save_params()
    else:
        lr /= 4.0
        optimizer.lr = lr

    model.reset_state()
    print("-" * 50)


# 테스트 데이터로 평가
model.reset_state()
ppl_test = eval_perplexity(model, corpus_test)
print("테스트 퍼플렉서티: ", ppl_test)
wandb.log({"Test Perplexity": cast_to_single_value(ppl_test)})
