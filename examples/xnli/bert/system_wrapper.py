import pytorch_wrapper as pw
import torch
import os
import uuid

from torch import nn
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from itertools import product
from transformers import AutoTokenizer, AutoModel, AdamW
from functools import partial

from .model import XNLIBERTModel
from .dataset import XNLIBERTDataset


class XNLIBERTSystemWrapper:

    def __init__(self, pretrained_bert_name, model_params):

        self._pretrained_bert_name = pretrained_bert_name
        bert_model = AutoModel.from_pretrained(pretrained_bert_name)
        model = XNLIBERTModel(bert_model, **model_params)

        if torch.cuda.is_available():
            self._system = pw.System(model, last_activation=nn.Softmax(dim=-1), device=torch.device('cuda'))
        else:
            self._system = pw.System(model, last_activation=nn.Softmax(dim=-1), device=torch.device('cpu'))

    def train(self,
              train_dataset_file,
              val_dataset_file,
              lr,
              batch_size,
              grad_accumulation_steps,
              run_on_multi_gpus,
              preprocessing_function,
              verbose=True,
              seed=0):
        torch.manual_seed(seed)
        tokenizer = AutoTokenizer.from_pretrained(self._pretrained_bert_name)
        train_dataset = XNLIBERTDataset(train_dataset_file, tokenizer, preprocessing_function)
        val_dataset = XNLIBERTDataset(val_dataset_file, tokenizer, preprocessing_function)
        self._train_impl(
            train_dataset,
            val_dataset,
            lr,
            batch_size,
            grad_accumulation_steps,
            run_on_multi_gpus,
            tokenizer.pad_token_id,
            verbose
        )

    def _train_impl(self,
                    train_dataset,
                    val_dataset,
                    lr,
                    batch_size,
                    grad_accumulation_steps,
                    run_on_multi_gpus,
                    pad_value,
                    verbose=True):

        train_dataloader = DataLoader(
            train_dataset,
            sampler=RandomSampler(train_dataset),
            batch_size=batch_size,
            collate_fn=partial(XNLIBERTDataset.collate_fn, pad_value=pad_value)
        )

        val_dataloader = DataLoader(
            val_dataset,
            sampler=SequentialSampler(val_dataset),
            batch_size=batch_size,
            collate_fn=partial(XNLIBERTDataset.collate_fn, pad_value=pad_value)
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
                    patience=3,
                    evaluation_data_loader_key='val',
                    evaluator_key='macro-f1',
                    tmp_best_state_filepath=f'{base_es_path}/temp.es.weights'
                )
            ],
            verbose=verbose
        )

    def evaluate(self, eval_dataset_file, batch_size, run_on_multi_gpus, preprocessing_function, verbose=True):
        tokenizer = AutoTokenizer.from_pretrained(self._pretrained_bert_name)
        eval_dataset = XNLIBERTDataset(eval_dataset_file, tokenizer, preprocessing_function)
        return self._evaluate_impl(eval_dataset, batch_size, run_on_multi_gpus, tokenizer.pad_token_id, verbose)

    def _evaluate_impl(self, eval_dataset, batch_size, run_on_multi_gpus, pad_value, verbose=True):

        eval_dataloader = DataLoader(
            eval_dataset,
            sampler=SequentialSampler(eval_dataset),
            batch_size=batch_size,
            collate_fn=partial(XNLIBERTDataset.collate_fn, pad_value=pad_value)
        )

        evaluators = {

            'acc': pw.evaluators.MultiClassAccuracyEvaluator(),
            'macro-prec': pw.evaluators.MultiClassPrecisionEvaluator(average='macro'),
            'macro-rec': pw.evaluators.MultiClassRecallEvaluator(average='macro'),
            'macro-f1': pw.evaluators.MultiClassF1Evaluator(average='macro'),
            'micro-prec': pw.evaluators.MultiClassPrecisionEvaluator(average='micro'),
            'micro-rec': pw.evaluators.MultiClassRecallEvaluator(average='micro'),
            'micro-f1': pw.evaluators.MultiClassF1Evaluator(average='micro')
        }

        if run_on_multi_gpus:
            return self._system.evaluate_on_multi_gpus(eval_dataloader, evaluators, verbose=verbose)
        else:
            return self._system.evaluate(eval_dataloader, evaluators, verbose=verbose)

    def save_model_state(self, path):
        self._system.save_model_state(path)

    @staticmethod
    def tune(pretrained_bert_name, train_dataset_file, val_dataset_file, run_on_multi_gpus, preprocessing_function):
        lrs = [5e-5, 3e-5, 2e-5]
        dp = [0, 0.1, 0.2]
        grad_accumulation_steps = [4, 8]
        batch_size = 4
        params = list(product(lrs, dp, grad_accumulation_steps))

        tokenizer = AutoTokenizer.from_pretrained(pretrained_bert_name)

        train_dataset = XNLIBERTDataset(train_dataset_file, tokenizer, preprocessing_function)
        val_dataset = XNLIBERTDataset(val_dataset_file, tokenizer, preprocessing_function)

        results = []
        for i, (lr, dp, grad_accumulation_steps) in enumerate(params):
            print(f'{i + 1}/{len(params)}')
            torch.manual_seed(0)
            current_system_wrapper = XNLIBERTSystemWrapper(pretrained_bert_name, {'dp': dp})
            current_system_wrapper._train_impl(
                train_dataset,
                val_dataset,
                lr,
                batch_size,
                grad_accumulation_steps,
                run_on_multi_gpus
            )

            current_results = current_system_wrapper._evaluate_impl(val_dataset, batch_size, run_on_multi_gpus)
            results.append([current_results['macro-f1'].score, (lr, dp, grad_accumulation_steps)])

        return results
