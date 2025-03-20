# coding: utf-8
import sys


sys.path.append("..")
from common import config

config.GPU = True
from common.np import *
from common.optimizer import SGD
from common.trainer import RnnlmTrainer
from common.util import eval_perplexity, to_gpu, cast_to_single_value
from dataset import ptb
from rnnlm import Rnnlm
import wandb

# GPU에서 실행하려면 아래 주석을 해제하세요(CuPy 필요).
# ==============================================
# ==============================================


wandb.init(
    project="RNN",
    name="RNNLM",
    config={
        "seed": 1000,
        "gradient_descent": "SGD",
        "learning_rate": 20,
        "epochs": 4,
        "batch_size": 20,
        "model": "RNNLM",
        "max_grad": 0.25,
        "model_params": {
            "hidden_size": 100,
            "wordvec_size": 100,
            "time_size": 35,
        },
        "dataset": "PTB",
        "gpu": True,
        "baseline": True,
    },
)


# 학습 데이터 읽기
corpus, word_to_id, id_to_word = ptb.load_data("train")
corpus_test, _, _ = ptb.load_data("test")
vocab_size = len(word_to_id)
if config.GPU:
    corpus = to_gpu(corpus)
    corpus_test = to_gpu(corpus_test)
xs = corpus[:-1]
ts = corpus[1:]


# 모델 생성
model = Rnnlm(
    vocab_size,
    wandb.config.model_params["wordvec_size"],
    wandb.config.model_params["hidden_size"],
)
optimizer = SGD(wandb.config.learning_rate)
trainer = RnnlmTrainer(model, optimizer)

# 기울기 클리핑을 적용하여 학습
trainer.fit(
    xs,
    ts,
    wandb.config.epochs,
    wandb.config.batch_size,
    wandb.config.model_params["time_size"],
    wandb.config.max_grad,
    eval_interval=20,
)
trainer.plot(ylim=(0, 500))

# 테스트 데이터로 평가
model.reset_state()
ppl_test = eval_perplexity(model, corpus_test)
wandb.log({"Test Perplexity": cast_to_single_value(ppl_test)})
print("테스트 퍼플렉서티: ", ppl_test)

# 매개변수 저장
model.save_params()
