# coding: utf-8
import sys

sys.path.append("..")
from common import config
from common.np import *  # import numpy as np
from common.layers import Embedding, SigmoidWithLoss
import collections
import time


class EmbeddingDot:
    def __init__(self, W):
        self.embed = Embedding(W)
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None

    def forward(self, h, idx):
        target_W = self.embed.forward(idx)
        out = np.sum(target_W * h, axis=1)

        self.cache = (h, target_W)
        return out

    def backward(self, dout):
        h, target_W = self.cache
        dout = dout.reshape(dout.shape[0], 1)

        dtarget_W = dout * h
        self.embed.backward(dtarget_W)
        dh = dout * target_W
        return dh


class UnigramSampler:
    def __init__(self, corpus, power, sample_size):
        self.sample_size = sample_size
        self.vocab_size = None
        self.word_p = None

        counts = collections.Counter()
        for word_id in corpus:
            counts[word_id] += 1

        vocab_size = len(counts)
        self.vocab_size = vocab_size

        self.word_p = np.zeros(vocab_size)
        for i in range(vocab_size):
            self.word_p[i] = counts[i]

        self.word_p = np.power(self.word_p, power)
        self.word_p /= np.sum(self.word_p)

    def get_negative_sample(self, target):
        batch_size = target.shape[0]
        if not config.GPU:
            negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32)

            for i in range(batch_size):
                p = self.word_p.copy()
                target_idx = target[i]
                p[target_idx] = 0
                p /= p.sum()
                negative_sample[i, :] = np.random.choice(
                    self.vocab_size, size=self.sample_size, replace=False, p=p
                )
        else:
            # GPU(cupy）로 계산할 때는 속도를 우선한다.
            # 부정적 예에 타깃이 포함될 수 있다.
            negative_sample = np.random.choice(
                self.vocab_size,
                size=(batch_size, self.sample_size),
                p=self.word_p,
            )

        return negative_sample


class NegativeSamplingLoss:
    def __init__(self, W, corpus, power=0.75, sample_size=5):
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power, sample_size)
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size + 1)]
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size + 1)]

        self.params, self.grads = [], []
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads
        self.time = { "forward": 0, "backward": 0}
        self.loss_cache = { "positive": 0, "negative": 0 }

    def forward(self, h, target):
        start = time.perf_counter()
        batch_size = target.shape[0]
        negative_sample = self.sampler.get_negative_sample(target)

        # 긍정적 예 순전파
        score = self.embed_dot_layers[0].forward(h, target)
        correct_label = np.ones(batch_size, dtype=np.int32)
        loss = self.loss_layers[0].forward(score, correct_label)
        self.loss_cache["positive"] = loss.copy()

        # 부정적 예 순전파
        negative_label = np.zeros(batch_size, dtype=np.int32)
        #for i in range(self.sample_size):
        #    negative_target = negative_sample[:, i]
        #    score = self.embed_dot_layers[1 + i].forward(h, negative_target)
        #    loss += self.loss_layers[1 + i].forward(score, negative_label)
        negative_loss = 0
        for i in range(1, len(self.embed_dot_layers)):
            negative_out = self.embed_dot_layers[i].forward(h, negative_sample[:, i - 1])
            negative_loss += self.loss_layers[i].forward(
                negative_out, negative_label
            )
        loss += negative_loss
        self.time["forward"] += time.perf_counter() - start
        self.loss_cache["negative"] = loss - self.loss_cache["positive"]
        return loss

    def backward(self, dout=1):
        start = time.perf_counter()
        dh = 0
        for l0, l1 in zip(self.loss_layers, self.embed_dot_layers):
            dscore = l0.backward(dout)
            dh += l1.backward(dscore)
        self.time["backward"] += time.perf_counter() - start
        return dh
