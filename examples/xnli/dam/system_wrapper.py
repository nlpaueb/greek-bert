import pytorch_wrapper as pw
import torch
import os
import uuid

from torch import nn
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.optim import AdamW
from itertools import product

from .model import XNLIDAMModel
from .dataset import XNLIDAMDataset


class XNLIDAMSystemWrapper:

    def __init__(self, embeddings, w2i, model_params):

        self._w2i = w2i
        model = XNLIDAMModel(embeddings, **model_params)
        if torch.cuda.is_available():
            self._system = pw.System(model, last_activation=nn.Softmax(dim=-1), device=torch.device('cuda'))
        else:
            self._system = pw.System(model, last_activation=nn.Softmax(dim=-1), device=torch.device('cpu'))

    def train(self, train_dataset_file, val_dataset_file, lr, batch_size, grad_accumulation_steps, run_on_multi_gpus):
        torch.manual_seed(0)
        train_dataset = XNLIDAMDataset(train_dataset_file, self._w2i)
        val_dataset = XNLIDAMDataset(val_dataset_file, self._w2i)
        self._train_impl(train_dataset, val_dataset, lr, batch_size, grad_accumulation_steps, run_on_multi_gpus)

    def _train_impl(self,
                    train_dataset,
                    val_dataset,
                    lr,
                    batch_size,
                    grad_accumulation_steps,
                    run_on_multi_gpus):

        train_dataloader = DataLoader(
            train_dataset,
            sampler=RandomSampler(train_dataset),
            batch_size=batch_size,
            collate_fn=XNLIDAMDataset.collate_fn
        )

        val_dataloader = DataLoader(
            val_dataset,
            sampler=SequentialSampler(val_dataset),
            batch_size=batch_size,
            collate_fn=XNLIDAMDataset.collate_fn
        )

        loss_wrapper = pw.loss_wrappers.GenericPointWiseLossWrapper(nn.CrossEntropyLoss())
        optimizer = AdamW(self._system.model.parameters(), lr=lr)

        base_es_path = f'/tmp/{uuid.uuid4().hex[:30]}/'
        os.makedirs(base_es_path, exist_ok=True)

        train_method = self._system.train_on_multi_gpus if run_on_multi_gpus else self._system.train

        _ = train_method(
            loss_wrapper,
            optimizer,
            train_data_loader=train_dataloader,
            evaluation_data_loaders={'val': val_dataloader},
            evaluators={'macro-f1': pw.evaluators.MultiClassF1Evaluator(average='macro')},
            gradient_accumulation_steps=grad_accumulation_steps,
            callbacks=[
                pw.training_callbacks.EarlyStoppingCriterionCallback(
                    patience=50,
                    evaluation_data_loader_key='val',
                    evaluator_key='macro-f1',
                    tmp_best_state_filepath=f'{base_es_path}/temp.es.weights'
                )
            ]
        )

    def evaluate(self, eval_dataset_file, batch_size, run_on_multi_gpus):
        eval_dataset = XNLIDAMDataset(eval_dataset_file, self._w2i)
        return self._evaluate_impl(eval_dataset, batch_size, run_on_multi_gpus)

    def _evaluate_impl(self, eval_dataset, batch_size, run_on_multi_gpus):

        eval_dataloader = DataLoader(
            eval_dataset,
            sampler=SequentialSampler(eval_dataset),
            batch_size=batch_size,
            collate_fn=XNLIDAMDataset.collate_fn
        )

        evaluators = {

            'acc': pw.evaluators.MultiClassAccuracyEvaluator(),
            'macro-prec': pw.evaluators.MultiClassPrecisionEvaluator(average='macro'),
            'macro-rec': pw.evaluators.MultiClassRecallEvaluator(average='macro'),
            'macro-f1': pw.evaluators.MultiClassF1Evaluator(average='macro')
        }

        if run_on_multi_gpus:
            return self._system.evaluate_on_multi_gpus(eval_dataloader, evaluators)
        else:
            return self._system.evaluate(eval_dataloader, evaluators)

    @staticmethod
    def tune(embeddings, w2i, train_dataset_file, val_dataset_file, run_on_multi_gpus):
        lrs = [0.01, 0.001]
        batch_size = [16, 32, 64]
        dp = [0, 0.1, 0.2, 0.3]
        hs = [100, 200, 300]
        params = list(product(lrs, dp, batch_size, hs))
        grad_accumulation_steps = 1

        train_dataset = XNLIDAMDataset(train_dataset_file, w2i)
        val_dataset = XNLIDAMDataset(val_dataset_file, w2i)

        results = []
        for i, (lr, dp, batch_size, hs) in enumerate(params):
            print(f'{i + 1}/{len(params)}')
            torch.manual_seed(0)
            current_system_wrapper = XNLIDAMSystemWrapper(
                embeddings,
                w2i,
                {
                    'rnn_dp': dp,
                    'mlp_dp': dp,
                    'rnn_hidden_size': hs,
                }
            )
            current_system_wrapper._train_impl(
                train_dataset,
                val_dataset,
                lr,
                batch_size,
                grad_accumulation_steps,
                run_on_multi_gpus
            )

            current_results = current_system_wrapper._evaluate_impl(val_dataset, batch_size, run_on_multi_gpus)
            results.append([current_results['macro-f1'].score, (lr, dp, batch_size, hs)])

        return results
