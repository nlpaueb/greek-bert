import torch

from pytorch_wrapper.evaluators import AbstractEvaluator


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
