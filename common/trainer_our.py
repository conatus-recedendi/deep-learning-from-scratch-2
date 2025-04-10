import os
import wandb
import pickle
import time
from common.np import *
from common.util import *


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_list = []
        self.mem_usage = {}
        self.train_acc_history = []
        self.test_acc_history = []
        self.current_epoch = 0
        # self.seed = seed
        # self.output_name = output_name

    def fit(self, x, t, max_epoch=10, batch_size=32, max_grad=None):
        data_size = len(x)
        max_iters = data_size // batch_size

        model, optimizer = self.model, self.optimizer
        for epoch in range(max_epoch):
            total_loss = 0.0
            loss_count = 0
            idx = np.random.permutation(len(x))
            sample_x = x[idx]
            sample_t = t[idx]
            for iters in range(max_iters):

                batch_x = sample_x[iters * batch_size : (iters + 1) * batch_size]
                batch_t = sample_t[iters * batch_size : (iters + 1) * batch_size]

                # 기울기 구해 매개변수 갱신
                loss = model.forward(batch_x, batch_t)
                model.backward()
                params, grads = remove_duplicate(
                    model.params, model.grads
                )  # 공유된 가중치를 하나로 모음
                if max_grad is not None:
                    clip_grads(grads, max_grad)
                optimizer.update(params, grads)
                total_loss += loss
                loss_count += 1
            # acc_test = model.accuracy(x_test, t_test, batch_size)
            acc_train = model.accuracy(sample_x, sample_t, batch_size)
            # acc_test = 0
            # acc_train = 0
            # if loss_count > 0

            print("end acc")
            wandb.log(
                {
                    "train_loss": cast_to_single_value(
                        total_loss / (loss_count + 1e-7)
                    ),
                    "train_acc": cast_to_single_value(acc_train),
                    "test_acc": cast_to_single_value(acc_test),
                }
            )
            self.loss_list.append(total_loss / (loss_count + 1e-7))
            self.train_acc_history.append(acc_train)
            self.test_acc_history.append(acc_test)
            # last_saved_model = (
            #     f"{self.output_name}/output_seed={self.seed}_epoch={epoch-1}.pkl"
            # )
            # if os.path.exists(last_saved_model) and epoch > 0:
            #     os.remove(last_saved_model)
            # pkl_file = f"{self.output_name}/output_seed={self.seed}_epoch={epoch}.pkl"
            # with open(pkl_file, "wb") as f:
            #     pickle.dump(self.model.params, f)
            total_loss = 0
            loss_count = 0
            self.current_epoch += 1


def remove_duplicate(params, grads):
    """
    매개변수 배열 중 중복되는 가중치를 하나로 모아
    그 가중치에 대응하는 기울기를 더한다.
    """
    params, grads = params[:], grads[:]  # copy list

    while True:
        find_flg = False
        L = len(params)

        for i in range(0, L - 1):
            for j in range(i + 1, L):
                # 가중치 공유 시
                if params[i] is params[j]:
                    grads[i] += grads[j]  # 경사를 더함
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)
                # 가중치를 전치행렬로 공유하는 경우(weight tying)
                elif (
                    params[i].ndim == 2
                    and params[j].ndim == 2
                    and params[i].T.shape == params[j].shape
                    and np.all(params[i].T == params[j])
                ):
                    grads[i] += grads[j].T
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)

                if find_flg:
                    break
            if find_flg:
                break

        if not find_flg:
            break

    return params, grads


class RnnlmTrainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.time_idx = None
        self.ppl_list = None
        self.eval_interval = None
        self.current_epoch = 0

    def get_batch(self, x, t, batch_size, time_size):
        batch_x = np.empty((batch_size, time_size), dtype="i")
        batch_t = np.empty((batch_size, time_size), dtype="i")

        data_size = len(x)
        jump = data_size // batch_size
        offsets = [
            i * jump for i in range(batch_size)
        ]  # 배치에서 각 샘플을 읽기 시작하는 위치

        for time in range(time_size):
            for i, offset in enumerate(offsets):
                batch_x[i, time] = x[(offset + self.time_idx) % data_size]
                batch_t[i, time] = t[(offset + self.time_idx) % data_size]
            self.time_idx += 1
        return batch_x, batch_t

    def fit(
        self,
        xs,
        ts,
        max_epoch=10,
        batch_size=20,
        time_size=35,
        max_grad=None,
        eval_interval=20,
    ):
        data_size = len(xs)
        max_iters = data_size // (batch_size * time_size)
        self.time_idx = 0
        self.ppl_list = []
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_count = 0

        start_time = time.time()
        for epoch in range(max_epoch):
            ppl = 0
            loss = 0
            for iters in range(max_iters):
                batch_x, batch_t = self.get_batch(xs, ts, batch_size, time_size)

                # 기울기를 구해 매개변수 갱신
                loss = model.loss(batch_x, batch_t)
                model.backward()
                params, grads = remove_duplicate(
                    model.params, model.grads
                )  # 공유된 가중치를 하나로 모음
                if max_grad is not None:
                    clip_grads(grads, max_grad)
                optimizer.update(params, grads)
                total_loss += loss
                loss_count += 1

                # 퍼플렉서티 평가
                if (eval_interval is not None) and (iters % eval_interval) == 0:
                    ppl = np.exp(total_loss / loss_count)
                    elapsed_time = time.time() - start_time
                    print(
                        "| 에폭 %d |  반복 %d / %d | 시간 %d[s] | 퍼플렉서티 %.2f"
                        % (
                            self.current_epoch + 1,
                            iters + 1,
                            max_iters,
                            elapsed_time,
                            ppl,
                        )
                    )
                    self.ppl_list.append(float(ppl))
                    total_loss, loss_count = 0, 0

            wandb.log(
                {
                    "Perplexity": cast_to_single_value(ppl),
                    # "train_acc": cast_to_single_value(acc_train),
                    # "test_acc": cast_to_single_value(acc_test),
                }
            )
            self.current_epoch += 1

    # def plot(self, ylim=None):
    #     x = np.arange(len(self.ppl_list))
    #     if config.GPU:
    #         x = to_cpu(x)
    #     if ylim is not None:
    #         plt.ylim(*ylim)
    #     plt.plot(x, self.ppl_list, label="train")
    #     plt.xlabel("반복 (x" + str(self.eval_interval) + ")")
    #     plt.ylabel("퍼플렉서티")
    #     plt.show()
