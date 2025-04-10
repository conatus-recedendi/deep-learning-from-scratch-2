# coding: utf-8
import sys

sys.path.append("..")
from common import config

# GPU에서 실행하려면 아래 주석을 해제하세요(CuPy 필요).
# ===============================================
config.GPU = True
# ===============================================
import pickle
from common.trainer_our import Trainer
from common.optimizer import Adam
from cbow import CBOW
from skip_gram_our import SkipGram
from common.util import create_contexts_target, to_cpu, to_gpu
from common.np import *
from dataset import ptb
import wandb
import json

wandb.init(
    project="Word2Vec",
    name="b_skip_gram",
    config={
        "seed": 1000,
        "gradient_descent": "Adam",
        "learning_rate": 0.001,
        "epochs": 1,
        "batch_size": 100,
        "model": "skip-gram",
        "model_params": {
            "value": {
                "hidden_size": 100,
                "window_size": 5,
                "power": 0.75,
            },
        },
        "dataset": "PTB",
        "gpu": config.GPU,
        "baseline": True,
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
# model = CBOW(
#    vocab_size,
#    hidden_size=wandb.config.model_params["hidden_size"],
#    window_size=wandb.config.model_params["window_size"],
#    corpus=corpus,
# )

model = SkipGram(
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
# trainer.plot()
for i in range(len(model.loss_layers)):
    print(f"loss_layer {i} : ", model.loss_layers[i].loss_cache)
times = model.training_time()
with open("train_loss.txt", "w") as f:
    for layer, timing in times.items():
        f.write(f"{layer}: {json.dumps(timing)}\n")
# 나중에 사용할 수 있도록 필요한 데이터 저장
word_vecs = model.word_vecs
if config.GPU:
    word_vecs = to_cpu(word_vecs)
params = {}
params["word_vecs"] = word_vecs.astype(np.float16)
params["word_to_id"] = word_to_id
params["id_to_word"] = id_to_word
pkl_file = "skip_gram_params.pkl"  # or 'skipgram_params.pkl'
with open(pkl_file, "wb") as f:
    pickle.dump(params, f, -1)
