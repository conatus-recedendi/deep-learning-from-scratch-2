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
        self.power = power
        self.corpus = corpus
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
        negative_sample = np.random.choice(
            self.vocab_size, (target.shape[0], self.sample_size), p=self.word_p
        )
        return negative_sample


class NegativeSamplingLoss:
    def __init__(self, W, sample_size, corpus, power=0.75):
        self.sampler = UnigramSampler(corpus, power, sample_size)
        self.out_layers = [EmbeddingDot(W) for _ in range(sample_size + 1)]
        self.loss_funcs = [SigmoidWithLoss() for _ in range(sample_size + 1)]
        self.params, self.grads = [], []
        for layer in self.out_layers:
            self.params += layer.params
            self.grads += layer.grads

        self.time = {"forward": 0, "backward": 0}
        self.loss_cache = {"positive": 0, "negative": 0}

    def forward(self, h, target):
        """
        :params h: hidden layer output -> (batch_size, hidden_size)
        :params target: label data -> (batch_size,)
        :return: loss
        """
        start = time.perf_counter()
        positive_loss = 0

        positive_out = self.out_layers[0].forward(h, target)
        positive_loss = self.loss_funcs[0].forward(
            positive_out, np.ones(h.shape[0], dtype=np.int32)
        )

        negative_loss = 0

        # negative_target = np.zeros((target.shape[0], 1), dtype=np.int32)
        negative_sample = self.sampler.get_negative_sample(target)
        for i in range(1, len(self.out_layers)):
            negative_out = self.out_layers[i].forward(h, negative_sample[:, i - 1])
            negative_loss += self.loss_funcs[i].forward(
                negative_out, np.zeros(h.shape[0], dtype=np.int32)
            )

        self.time["forward"] += time.perf_counter() - start
        self.loss_cache["positive"] = positive_loss
        self.loss_cache["negative"] = negative_loss
        return positive_loss + negative_loss

    def backward(self, dout=1):

        start = time.perf_counter()
        total_dout = 0
        for l0, l1 in zip(self.loss_funcs, self.out_layers):
            ds = l0.backward(dout)
            total_dout += l1.backward(ds)

        self.time["backward"] += time.perf_counter() - start
        return total_dout

    def memory_usage(self):
        pass
