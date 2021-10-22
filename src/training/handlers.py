import csv
import numpy as np
from ignite.engine import Events
from ignite.handlers import *
from tqdm import tqdm
from collections import deque

import sys

class LRScheduler(object):
    def __init__(self, scheduler, loss):
        self.scheduler = scheduler
        self.loss = loss

    def __call__(self, engine):
        loss_val = engine.state.metrics[self.loss]
        self.scheduler.step(loss_val)

    def attach(self, engine):
        engine.add_event_handler(Events.COMPLETED, self)
        return self


class Tracer(object):
    def __init__(self, metrics=None):
        self.metrics = metrics
        self.trace = []
        if metrics is None:
            setattr(self, 'batch_losses', [])
            setattr(self, 'ninstances', 0)

    def attach(self, engine):
        if self.metrics is None:
            def _trace(engine):
                self.trace.append(np.sum(self.batch_losses) / self.ninstances)

            def _save_batch_ouput(engine):
                batch_size = engine.state.batch[0].size(0)
                self.batch_losses.append(engine.state.output * batch_size)
                self.ninstances += batch_size

            def _initialize_batch_trace(engine):
                getattr(self, 'batch_losses').clear()
                setattr(self, 'ninstances', 0)

            engine.add_event_handler(Events.EPOCH_STARTED, _initialize_batch_trace)
            engine.add_event_handler(Events.ITERATION_COMPLETED, _save_batch_ouput)
        else:
            def _trace(engine):
                metrics_values = engine.state.metrics
                self.trace.append(tuple(metrics_values[m] for m in self.metrics))

        engine.add_event_handler(Events.EPOCH_COMPLETED, _trace)

        return self

    def save(self, save_path):
        loss = 'training' if self.metrics is None else 'validation'
        with open('{}/{}.csv'.format(save_path, loss), mode='w') as f:
            wr = csv.writer(f, quoting=csv.QUOTE_ALL)
            for i, v in enumerate(self.trace):
                try:
                    v = list(v)
                except TypeError:
                    v = [v]
                wr.writerow([i + 1] + v)


class ProgressLog(object):
    def __init__(self, n_batches, log_interval, pbar=None, desc=None):
        self.desc = 'iteration-loss: {:.5f}' if desc is None else desc
        self.pbar = pbar or tqdm(
            initial=0, leave=False, total=n_batches,
            desc=self.desc.format(0)
        )
        self.log_interval = log_interval
        self.running_loss = 0
        self.n_instances = 0
        self.n_batches = n_batches

    def attach(self, trainer, evaluator=None, metrics=None):
        def _log_batch(engine):
            batch_size = engine.state.batch[0].size(0)
            self.running_loss += engine.state.output * batch_size
            self.n_instances += batch_size

            iter = engine.state.iteration % self.n_batches
            if iter % self.log_interval == 0:
                self.pbar.desc = self.desc.format(
                    engine.state.output)
                self.pbar.update(self.log_interval)

        def _log_epoch(engine):
            self.pbar.refresh()
            tqdm.write("Epoch: {} - avg loss: {:.5f}"
                .format(engine.state.epoch, self.running_loss / self.n_instances))
            self.running_loss = 0
            self.n_instances = 0
            self.pbar.n = self.pbar.last_print_n = 0

        trainer.add_event_handler(Events.ITERATION_COMPLETED, _log_batch)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, _log_epoch)
        trainer.add_event_handler(Events.COMPLETED, lambda x: self.pbar.close())

        if evaluator is not None and metrics is None:
            raise ValueError('')

        if evaluator is not None:
            self.evaluator = evaluator
            def _log_validation(engine):
                metrics = self.evaluator.state.metrics

                message = []
                for k, v in metrics.items():
                    message.append("{}: {:.5f}".format(k, v))
                tqdm.write('\tvalidation: ' + ' - '.join(message))

            trainer.add_event_handler(Events.EPOCH_COMPLETED, _log_validation)

        return self


class ConvergenceStopping(object):
    def __init__(self, tol=1e-5, patience=50):
        self.tol = tol
        self.patience = patience
        self._loss_history = deque()

    def attach(self, trainer):
        def _reset(engine):
            self._loss_history.clear()

        def _updated_history(engine):
            self._loss_history.append(engine.state.output)
            if len(self._loss_history) > self.patience:
                self._loss_history.popleft()

        def _evaluate_convergence(engine):
            if len(self._loss_history) < self.patience:
                return

            last_loss = self._loss_history[-1]

            for loss in self._loss_history:
                if abs(loss - last_loss) > self.tol:
                    return

            engine.terminate()

        trainer.add_event_handler(Events.STARTED, _reset)
        trainer.add_event_handler(Events.ITERATION_COMPLETED, _updated_history)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, _evaluate_convergence)
