# coding: utf-8
import sys

sys.path.append("..")
from common import config


config.GPU = True
from common.optimizer import SGD
from common.trainer import RnnlmTrainer
from common.util import to_gpu, eval_perplexity, cast_to_single_value
from dataset import ptb
from simple_rnnlm import SimpleRnnlm
import wandb


wandb.init(
    project="RNN",
    name="SimpleRNNLM",
    config={
        "seed": 1000,
        "gradient_descent": "SGD",
        "learning_rate": 0.1,
        "epochs": 100,
        "batch_size": 10,
        "model": "SimpleRNNLM",
        "model_params": {
            "hidden_size": 100,
            "wordvec_size": 100,
            "time_size": 5,
            "power": 0.75,
        },
        "dataset": "PTB",
        "gpu": config.GPU,
        "baseline": True,
    },
)

# 학습 데이터 읽기
corpus, word_to_id, id_to_word = ptb.load_data("train")
corpus_test, _, _ = ptb.load_data("test")
corpus_size = len(corpus)  # 테스트 데이터셋을 작게 설정
corpus = corpus[:corpus_size]
vocab_size = int(max(corpus) + 1)
if config.GPU:
    corpus = to_gpu(corpus)
    corpus_test = to_gpu(corpus_test)
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

model.reset_state()
ppl_test = eval_perplexity(model, corpus_test)
wandb.log({"Test Perplexity": cast_to_single_value(ppl_test)})
