#----------------------------------------------------------------------------
#                                  Imports
#----------------------------------------------------------------------------

import torchvision
from torchvision.transforms import ToTensor

from transformers import ViTModel
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn as nn
import torch.nn.functional as F

from transformers import ViTFeatureExtractor
import torch.nn as nn
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import ViTForImageClassification
from typing import Dict, List, Optional, Set, Tuple, Union
from modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ImageClassifierOutput, MaskedLMOutput

#----------------------------------------------------------------------------
#                                  Functions
#----------------------------------------------------------------------------

def vit_load_model(path):

    model = torch.load(path)
    model.eval()

    return model

#----------------------------------------------------------------------------
#                           Vision Transformer Class
#----------------------------------------------------------------------------

# class ViTForImageClassification(nn.Module):
#     def __init__(self, num_labels=2):
#         super(ViTForImageClassification, self).__init__()
#         self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
#         self.dropout = nn.Dropout(0.1)
#         self.classifier = nn.Linear(self.vit.config.hidden_size, num_labels)
#         self.num_labels = num_labels
#         #self.softmax = nn.Softmax()

#     def forward(self, pixel_values):#, labels):
#         outputs = self.vit(pixel_values=pixel_values)
#         output = self.dropout(outputs.last_hidden_state[:,0])
#         logits = self.classifier(output)

#         loss = None
#         # if labels is not None:
#         #     loss_fct = nn.CrossEntropyLoss()
#         #     loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#         if loss is not None:
#             return logits, loss.item()
#         else:
#             return logits, None

class ViTForImageClassification(nn.Module):#$ViTPreTrainedModel):
    def __init__(self, num_labels=2):
        super(ViTForImageClassification, self).__init__()

        self.num_labels = num_labels
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

        # Classifier head
        self.classifier = nn.Linear(self.config.hidden_size, num_labels) #if config.num_labels > 0 else nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()

    # @add_start_docstrings_to_model_forward(VIT_INPUTS_DOCSTRING)
    # @add_code_sample_docstrings(
    #     processor_class=_FEAT_EXTRACTOR_FOR_DOC,
    #     checkpoint=_IMAGE_CLASS_CHECKPOINT,
    #     output_type=ImageClassifierOutput,
    #     config_class=_CONFIG_FOR_DOC,
    #     expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    #)
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        #return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.classifier(sequence_output[:, 0, :])

        loss = None
        # if labels is not None:
        #     if self.config.problem_type is None:
        #         if self.num_labels == 1:
        #             self.config.problem_type = "regression"
        #         elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
        #             self.config.problem_type = "single_label_classification"
        #         else:
        #             self.config.problem_type = "multi_label_classification"

        #     if self.config.problem_type == "regression":
        #         loss_fct = MSELoss()
        #         if self.num_labels == 1:
        #             loss = loss_fct(logits.squeeze(), labels.squeeze())
        #         else:
        #             loss = loss_fct(logits, labels)
        #     elif self.config.problem_type == "single_label_classification":
        #         loss_fct = CrossEntropyLoss()
        #         loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        # self.config.problem_type == "multi_label_classification":
        # loss_fct = BCEWithLogitsLoss()
        # loss = loss_fct(logits)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )            