import torch

from abc import abstractmethod
from pytorch_wrapper.evaluators import AbstractEvaluator, GenericEvaluatorResults


def convert_tags_to_entities(seq):
    # for nested list
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]

    prev_tag = 'O'
    prev_tag_type = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']):
        tag = chunk[0]
        tag_type = '-'.join(chunk.split('-')[1:])
        if _is_end_of_chunk(prev_tag, tag, prev_tag_type, tag_type):
            chunks.append((prev_tag_type, begin_offset, i - 1))
        if _is_start_of_chunk(prev_tag, tag, prev_tag_type, tag_type):
            begin_offset = i
        prev_tag = tag
        prev_tag_type = tag_type

    return chunks


def _is_end_of_chunk(prev_tag, tag, prev_tag_type, tag_type):
    return (
            prev_tag == 'E' or
            prev_tag == 'S' or
            prev_tag == 'B' and tag == 'B' or
            prev_tag == 'B' and tag == 'S' or
            prev_tag == 'B' and tag == 'O' or
            prev_tag == 'I' and tag == 'B' or
            prev_tag == 'I' and tag == 'S' or
            prev_tag == 'I' and tag == 'O' or
            prev_tag != 'O' and prev_tag_type != tag_type
    )


def _is_start_of_chunk(prev_tag, tag, prev_tag_type, tag_type):
    return (
            tag == 'B' or
            tag == 'S' or
            prev_tag == 'E' and tag == 'E' or
            prev_tag == 'E' and tag == 'I' or
            prev_tag == 'S' and tag == 'E' or
            prev_tag == 'S' and tag == 'I' or
            prev_tag == 'O' and tag == 'E' or
            prev_tag == 'O' and tag == 'I' or
            tag != 'O' and prev_tag_type != tag_type
    )


class MaskedTokenLabelingEvaluatorWrapper(AbstractEvaluator):

    def __init__(self, evaluator, batch_input_key='input', model_output_key=None,
                 batch_target_key='target', batch_mask_key='mask'):
        self._evaluator = evaluator
        super(MaskedTokenLabelingEvaluatorWrapper, self).__init__()
        self._batch_input_key = batch_input_key
        self._model_output_key = model_output_key
        self._batch_target_key = batch_target_key
        self._batch_mask_key = batch_mask_key
        self.reset()

    def reset(self):
        self._evaluator.reset()

    def step(self, output, batch, last_activation=None):
        if self._model_output_key is not None:
            output = output[self._model_output_key]

        mask = batch[self._batch_mask_key].to(output.device)

        output_extra_dims = output.dim() - mask.dim()
        output_mask_new_shape = list(mask.shape) + [1] * output_extra_dims
        output_extra_dims_shape = list(output.shape)[mask.dim():]
        output = torch.masked_select(output, mask.view(*output_mask_new_shape))
        output = output.view(-1, *output_extra_dims_shape)

        target = batch[self._batch_target_key].to(output.device)
        target_extra_dims = target.dim() - mask.dim()
        target_mask_new_shape = list(mask.shape) + [1] * target_extra_dims
        target_extra_dims_shape = list(target.shape)[mask.dim():]
        target = torch.masked_select(target, mask.view(*target_mask_new_shape))
        target = target.view(-1, *target_extra_dims_shape)

        new_batch = {k: batch[k] for k in batch if k != self._batch_target_key}
        new_batch[self._batch_target_key] = target

        self._evaluator.step(output, new_batch, last_activation)

    def calculate(self):
        return self._evaluator.calculate()


class AbstractMaskedTokenEntityLabelingEvaluator(AbstractEvaluator):

    def __init__(self, i2l, batch_input_key='input', model_output_key=None, batch_target_key='target',
                 batch_mask_key='mask'):
        self._batch_input_key = batch_input_key
        self._model_output_key = model_output_key
        self._batch_target_key = batch_target_key
        self._batch_mask_key = batch_mask_key
        self._i2l = i2l
        self._labels = [l[2:] for l in i2l if l[:2] == 'B-']
        super(AbstractMaskedTokenEntityLabelingEvaluator, self).__init__()

    def reset(self):
        self._tp = {}
        self._fp = {}
        self._fn = {}

        for l in self._labels:
            self._tp[l] = 0
            self._fp[l] = 0
            self._fn[l] = 0

    def step(self, output, batch, last_activation=None):
        if self._model_output_key is not None:
            output = output[self._model_output_key]

        output = output.argmax(dim=-1)
        mask = batch[self._batch_mask_key].to(output.device)
        targets = batch[self._batch_target_key].to(output.device)

        for i in range(output.shape[0]):
            cur_out = torch.masked_select(output[i], mask[i]).tolist()
            cur_targets = torch.masked_select(targets[i], mask[i]).tolist()

            converted_out = set(convert_tags_to_entities([self._i2l[o] for o in cur_out]))
            converted_targets = set(convert_tags_to_entities([self._i2l[t] for t in cur_targets]))

            cur_tp = converted_out.intersection(converted_targets)
            cur_fp = converted_out.difference(converted_targets)
            cur_fn = converted_targets.difference(converted_out)

            for l in self._labels:
                self._tp[l] += len([c for c in cur_tp if c[0] == l])
                self._fp[l] += len([c for c in cur_fp if c[0] == l])
                self._fn[l] += len([c for c in cur_fn if c[0] == l])

    @abstractmethod
    def calculate(self):
        pass


class MultiClassPrecisionEvaluatorMaskedTokenEntityLabelingEvaluator \
            (AbstractMaskedTokenEntityLabelingEvaluator):

    def __init__(self, i2l, average='macro', batch_input_key='input', model_output_key=None, batch_target_key='target',
                 batch_mask_key='mask'):
        super(
            MultiClassPrecisionEvaluatorMaskedTokenEntityLabelingEvaluator,
            self
        ).__init__(
            i2l,
            batch_input_key,
            model_output_key,
            batch_target_key,
            batch_mask_key,
        )

        self._average = average

    def calculate(self):
        if self._average == 'macro':
            per_class_score = {}
            for l in self._labels:
                denominator = self._tp[l] + self._fp[l]
                per_class_score[l] = (self._tp[l] / denominator) if denominator != 0 else 0
            score = sum(per_class_score[l] for l in per_class_score) / len(per_class_score)
        else:
            global_tp = sum(self._tp[l] for l in self._tp)
            global_fp = sum(self._fp[l] for l in self._fp)
            denominator = global_tp + global_fp
            score = (global_tp / denominator) if denominator > 0 else 0

        return GenericEvaluatorResults(score, self._average + '-precision', '%5.4f', is_max_better=True)


class MultiClassRecallEvaluatorMaskedTokenEntityLabelingEvaluator \
            (AbstractMaskedTokenEntityLabelingEvaluator):

    def __init__(self, i2l, average='macro', batch_input_key='input', model_output_key=None, batch_target_key='target',
                 batch_mask_key='mask'):
        super(
            MultiClassRecallEvaluatorMaskedTokenEntityLabelingEvaluator,
            self
        ).__init__(
            i2l,
            batch_input_key,
            model_output_key,
            batch_target_key,
            batch_mask_key,
        )

        self._average = average

    def calculate(self):
        if self._average == 'macro':
            per_class_score = {}
            for l in self._labels:
                denominator = self._tp[l] + self._fn[l]
                per_class_score[l] = (self._tp[l] / denominator) if denominator != 0 else 0
            score = sum(per_class_score[l] for l in per_class_score) / len(per_class_score)
        else:
            global_tp = sum(self._tp[l] for l in self._tp)
            global_fn = sum(self._fn[l] for l in self._fn)
            denominator = global_tp + global_fn
            score = (global_tp / denominator) if denominator > 0 else 0

        return GenericEvaluatorResults(score, self._average + '-recall', '%5.4f', is_max_better=True)


class MultiClassF1EvaluatorMaskedTokenEntityLabelingEvaluator \
            (AbstractMaskedTokenEntityLabelingEvaluator):

    def __init__(self, i2l, average='macro', batch_input_key='input', model_output_key=None, batch_target_key='target',
                 batch_mask_key='mask'):
        super(
            MultiClassF1EvaluatorMaskedTokenEntityLabelingEvaluator,
            self
        ).__init__(
            i2l,
            batch_input_key,
            model_output_key,
            batch_target_key,
            batch_mask_key,
        )

        self._average = average

    def calculate(self):
        if self._average == 'macro':
            per_class_score = {}
            for l in self._labels:
                pr_denominator = self._tp[l] + self._fp[l]
                pr_score = (self._tp[l] / pr_denominator) if pr_denominator > 0 else 0
                rec_denominator = self._tp[l] + self._fn[l]
                rec_score = (self._tp[l] / rec_denominator) if rec_denominator > 0 else 0
                denominator = pr_score + rec_score
                per_class_score[l] = (2 * pr_score * rec_score) / denominator if denominator > 0 else 0
            score = sum(per_class_score[l] for l in per_class_score) / len(per_class_score)
        else:
            global_tp = sum(self._tp[l] for l in self._tp)
            global_fn = sum(self._fn[l] for l in self._fn)
            global_fp = sum(self._fp[l] for l in self._fp)
            pr_denominator = global_tp + global_fp
            pr_score = (global_tp / pr_denominator) if pr_denominator > 0 else 0
            rec_denominator = global_tp + global_fn
            rec_score = (global_tp / rec_denominator) if rec_denominator > 0 else 0
            denominator = pr_score + rec_score
            score = (2 * pr_score * rec_score) / denominator if denominator > 0 else 0

        return GenericEvaluatorResults(score, self._average + '-f1', '%5.4f', is_max_better=True)
