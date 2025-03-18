# coding: utf-8
import sys

sys.path.append("..")
from common.optimizer import SGD
from common.trainer import RnnlmTrainer
from dataset import ptb
from simple_rnnlm import SimpleRnnlm
import wandb
from common.config import GPU


wandb.init(
    project="RNN",
    name="SimpleRNNLM",
    config={
        "seed": {"value": 1000},
        "gradient_descent": {"value": "SGD"},
        "learning_rate": {"value": 0.1},
        "epochs": {"value": 100},
        "batch_size": 10,
        "model": {"value": "SimpleRNNLM"},
        "model_params": {
            "value": {
                "hidden_size": 100,
                "wordvec_size": 100,
                "time_size": 5,
                "power": 0.75,
            },
        },
        "dataset": {"value": "PTB-1000"},
        "gpu": {"value": False},
        "baseline": True,
    },
)

# 학습 데이터 읽기
corpus, word_to_id, id_to_word = ptb.load_data("train")
corpus_size = 1000  # 테스트 데이터셋을 작게 설정
corpus = corpus[:corpus_size]
vocab_size = int(max(corpus) + 1)
xs = corpus[:-1]  # 입력
ts = corpus[1:]  # 출력（정답 레이블）

# 모델 생성
model = SimpleRnnlm(
    vocab_size,
    wandb.config.model_params["wordvec_size"],
    wandb.config.model_params["hidden_size"],
)
optimizer = SGD(wandb.config.learning_rate)
trainer = RnnlmTrainer(model, optimizer)

trainer.fit(
    xs,
    ts,
    wandb.config.epochs,
    wandb.config.batch_size,
    wandb.config.model_params["time_size"],
)
trainer.plot()
