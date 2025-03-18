# coding: utf-8
import sys

sys.path.append("..")
import numpy as np
from common import config

# GPU에서 실행하려면 아래 주석을 해제하세요(CuPy 필요).
# ===============================================
config.GPU = True
# ===============================================
import pickle
from common.trainer import Trainer
from common.optimizer import Adam
from cbow import CBOW
from skip_gram import SkipGram
from common.util import create_contexts_target, to_cpu, to_gpu
from dataset import ptb
import wandb


wandb.init(
    project="Word2Vec",
    name="Negative_All_Sample_CBOW",
    config={
        "seed": {"value": 1000},
        "gradient_descent": {"value": "Adam"},
        "learning_rate": {"value": 0.001},
        "epochs": {"value": 10},
        "batch_size": {"value": 100},
        "model": {"value": "Negative_All_Sample_CBOW"},
        "model_params": {
            "value": {
                "hidden_size": 100,
                "window_size": 5,
                "power": 0.75,
            },
        },
        "dataset": {"value": "PTB"},
        "gpu": {"value": config.GPU},
        # "batch_norm": {"value": False},
        # "weight_decay_lambda": {"value": 0},
        # "dataset": {"value": ""},
        # "activation": {"value": "relu"},
        # "weight_init_std": {"value": "he"},
        # "dropout": {"value": 0.15},},
    },
)

np.random.seed(wandb.config.seed)

# 데이터 읽기
corpus, word_to_id, id_to_word = ptb.load_data("train")
vocab_size = len(word_to_id)

contexts, target = create_contexts_target(
    corpus, window_size=wandb.config.model_params["window_size"]
)
if config.GPU:
    contexts, target = to_gpu(contexts), to_gpu(target)

# 모델 등 생성
model = CBOW(
    vocab_size,
    hidden_size=wandb.config.model_params["hidden_size"],
    window_size=wandb.config.model_params["window_size"],
    corpus=corpus,
)
# model = SkipGram(vocab_size, hidden_size, window_size, corpus)
optimizer = Adam()
trainer = Trainer(model, optimizer)

# 학습 시작
trainer.fit(
    contexts, target, max_epoch=wandb.config.epochs, batch_size=wandb.config.batch_size
)
trainer.plot()

# 나중에 사용할 수 있도록 필요한 데이터 저장
word_vecs = model.word_vecs
if config.GPU:
    word_vecs = to_cpu(word_vecs)
params = {}
params["word_vecs"] = word_vecs.astype(np.float16)
params["word_to_id"] = word_to_id
params["id_to_word"] = id_to_word
pkl_file = "cbow_params.pkl"  # or 'skipgram_params.pkl'
with open(pkl_file, "wb") as f:
    pickle.dump(params, f, -1)
