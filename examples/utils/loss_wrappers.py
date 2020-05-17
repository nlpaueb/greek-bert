import torch

from pytorch_wrapper.loss_wrappers import AbstractLossWrapper


class MaskedTokenLabelingGenericPointWiseLossWrapper(AbstractLossWrapper):

    def __init__(self, loss, batch_input_key='input', model_output_key=None,
                 batch_target_key='target', batch_mask_key='mask', perform_last_activation=False):

        super(MaskedTokenLabelingGenericPointWiseLossWrapper, self).__init__()
        self._loss = loss
        self._batch_input_key = batch_input_key
        self._batch_mask_key = batch_mask_key
        self._model_output_key = model_output_key
        self._batch_target_key = batch_target_key
        self._perform_last_activation = perform_last_activation

    def calculate_loss(self, output, batch, training_context, last_activation=None):

        if self._model_output_key is not None:
            output = output[self._model_output_key]

        if last_activation is not None and self._perform_last_activation:
            output = last_activation(output)

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

        return self._loss(output, target)


class PassThroughLossWrapper(AbstractLossWrapper):
    """
    Dummy adapter that returns the loss as returned by the model. Useful when the loss is calculated inside the model's
        forward method.
    """

    def __init__(self, model_loss_key=None):
        """
        :param model_loss_key: Key where the dict returned by the model contains the calculated loss. Leave None if the
            model returns only the loss.
        """
        super(PassThroughLossWrapper, self).__init__()
        self._model_loss_key = model_loss_key

    def calculate_loss(self, output, batch, training_context, last_activation=None):
        """
        Calculates the loss for a single batch.
        :param output: Output of the model.
        :param batch: Dict that contains all information needed by the loss wrapper.
        :param training_context: Dict containing information regarding the training process.
        :param last_activation: Last activation provided to the System.
        :return: Output of the loss function/module.
        """
        if self._model_loss_key is None:
            return output
        else:
            return output[self._model_loss_key]

    def to(self, device):
        pass
