# coding: utf-8
import sys

sys.path.append("..")  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from common import config
from common.np import *
from common.layers import *
from ch04.negative_sampling_layer_our import NegativeSamplingLoss
import pickle


class SkipGram:
    def __init__(
        self,
        vocab_size,
        hidden_size=100,
        window_size=1,
        sample_size=5,
        corpus=None,
        weight_init_std=0.01,
        power=0.75,
        word_to_id=None,
        id_to_word=None,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.sample_size = sample_size
        self.corpus = corpus
        self.power = power

        self.word_to_id = word_to_id
        self.id_to_word = id_to_word

        self.params = []
        self.grads = []

        W_in = weight_init_std * np.random.randn(vocab_size, hidden_size).astype("f")
        W_out = weight_init_std * np.random.randn(vocab_size, hidden_size).astype("f")

        self.in_layer = Embedding(W_in)
        # self.out_layer = Embedding(W_out)

        self.loss_layers = [
            NegativeSamplingLoss(
                W=W_out,
                sample_size=self.sample_size,
                corpus=corpus,
                power=self.power,
            )
            for _ in range(window_size * 2)
        ]

        self.layers = [self.in_layer] + self.loss_layers
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

        self.word_vecs = W_in

    def forward(self, contexts, target):
        h = self.in_layer.forward(target)
        # s = self.out_layer.forward(h)

        return h

    # def loss(self, contexts, target):
    #     s = self.forward(contexts, target)
    #     loss = 0
    #     for i, layer in enumerate(self.loss_layers):
    #         loss += layer.forward(s, contexts[:, i].reshape(-1))

    #     # l1 = self.loss_layer_1.forward(s, contexts[:, 0])
    #     # l2 = self.loss_layer_2.forward(s, contexts[:, 1])

    #     return loss

    def forward(self, contexts, target):
        h = self.in_layer.forward(target)

        loss = 0
        for i, layer in enumerate(self.loss_layers):
            loss += layer.forward(h, contexts[:, i])
        return loss

    # def backward(self, dout=1):
    #     dl = 0
    #     for layer in self.loss_layers:
    #         dl += layer.backward(dout)

    #     # dl1 = self.loss_layer_1.backward(dout)
    #     # dl2 = self.loss_layer_2.backward(dout)

    #     ds = dl
    #     # dh = self.out_layer.backward(ds)
    #     self.in_layer.backward(ds)
    #     return None

    def backward(self, dout=1):
        dh = 0
        for i, layer in enumerate(self.loss_layers):
            dh += layer.backward(dout)
        self.in_layer.backward(dh)
        return None

    def accuracy(self, x, target, batch_size=32):
        return 0.0

    def memory_usage(self):
        pass

    def training_time(self):
        time_usage = {}
        total_forward = 0
        total_backward = 0
        for i, layer in enumerate(self.layers):
            if hasattr(layer, "time"):
                time_usage[f"Layer {i} ({layer.__class__.__name__})"] = {
                    "forward": layer.time["forward"],
                    "backward": layer.time["backward"],
                }
                total_forward += layer.time["forward"]
                total_backward += layer.time["backward"]
        time_usage["Total"] = {"forward": total_forward, "backward": total_backward}
        return time_usage

    def save_model(self, file_name):
        word_vecs = self.word_vecs
        if config.GPU:
            word_vecs = to_cpu(self.word_vecs)
        params = {}
        params["word_vecs"] = word_vecs.astype(np.float16)
        params["word_to_id"] = self.word_to_id
        params["id_to_word"] = self.id_to_word
        pkl_file = file_name
        with open(pkl_file, "wb") as f:
            pickle.dump(params, f)
