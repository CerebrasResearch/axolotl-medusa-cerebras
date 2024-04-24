from transformers import (
    PretrainedConfig,
    TrainerCallback,
)
import logging
import warnings
from functools import partial
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from axolotl.utils.dict import DictDefault
from axolotl.utils.distributed import is_main_process
import axolotl
from transformers.trainer_pt_utils import LabelSmoother
IGNORE_TOKEN_ID = LabelSmoother.ignore_index
import types
import math
import wandb
import transformers

logger = LOG = logging.getLogger("axolotl.monkeypatch.medusa")

class MedusaConfig(PretrainedConfig):
    """
    Configuration class for Medusa model.

    Args:
        medusa_num_heads (int, optional): Number of heads for the Medusa layer. Default is 2.
        medusa_num_layers (int, optional): Number of Medusa layers. Default is 1.
        base_model_name_or_path (str, optional): The name or path of the base model. Default is "lmsys/vicuna-7b-v1.3".
        num_unfreezed_layers (int, optional): Number of layers to unfreeze. Default is 0.
        **kwargs: Additional keyword arguments to be passed to the parent class constructor.
    """

    def __init__(
        self,
        medusa_num_heads=4,
        medusa_num_layers=1,
        base_model_name_or_path="lmsys/vicuna-7b-v1.3",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.medusa_num_heads = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        self.base_model_name_or_path = base_model_name_or_path

        print(torch.randint(10, (2, 2)))


class ResBlock(nn.Module):
    """
    A Residual Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.

    Args:
        hidden_size (int): The size of the hidden layers in the block.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        # Initialize as an identity mapping
        nn.init.zeros_(self.linear.weight)
        # Use SiLU activation to keep consistent with the Llama model
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Forward pass of the ResBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the residual connection and activation.
        """
        return x + self.act(self.linear(x))

def add_medusa_heads(
    self,
    medusa_num_heads=4,
    medusa_num_layers=0,
):
    """
    Args:
        self (nn.Module): The base language model to be used.
        medusa_num_heads (int, optional): Number of additional tokens to predict. Defaults to 3.
        medusa_num_layers (int, optional): Number of ResBlock layers for each Medusa head. Defaults to 0.
    """
    hidden_size = self.lm_head.weight.shape[-1]
    vocab_size = self.lm_head.weight.shape[0]
    self.config.medusa_num_layers = medusa_num_layers
    self.config.medusa_num_heads = medusa_num_heads
    self.medusa_num_heads = medusa_num_heads
    # Create a list of Medusa heads
    self.medusa_head = nn.ModuleList(
        [
            nn.Sequential(
                *([ResBlock(hidden_size)] * medusa_num_layers),
                nn.Linear(hidden_size, vocab_size, bias=False),
            )
            for _ in range(medusa_num_heads)
        ]
    )

    # Ensure medusa_head's dtype and device align with the base_model
    self.medusa_head.to(self.dtype).to(self.device)

    # Copy weights over from base model
    for i in range(medusa_num_heads):
        # Initialize the weights of each medusa_head using the base model's weights
        self.medusa_head[i][-1].weight.data[:] = self.lm_head.weight.data[:]

    self.old_forward = self.forward

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        medusa_return: bool = False,
        medusa_only_heads: bool = False,
    ):
        """Forward pass of the MedusaModel.
        Returns:
            torch.Tensor: A tensor containing predictions from all Medusa heads.
            (Optional) Original predictions from the base model's LM head.
        """
        # LOG.debug("medusa_return: %s", medusa_return)
        if not medusa_return:
            return self.old_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # Pass input through the base model
        if medusa_only_heads:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                hidden_states = outputs[0]
                medusa_logits = [self.lm_head(hidden_states)]
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states = outputs[0]
            medusa_logits = [self.lm_head(hidden_states)]
        for i in range(self.medusa_num_heads):
            medusa_logits.append(self.medusa_head[i](hidden_states))
        return torch.stack(medusa_logits, dim=0)
    
    self.forward = types.MethodType(forward, self)

class ResBlock_Hydra(nn.Module):
    """
    A Residual Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.

    Args:
        hidden_size (int): The size of the hidden layers in the block.
    """

    def __init__(self, hidden_size, num_condition=0):
        super().__init__()
        self.linear = nn.Linear(hidden_size * (num_condition + 1), hidden_size)
        # Handling residual connection when reducing dim
        if num_condition > 0:
            self.res_connection = nn.Linear(hidden_size * (num_condition + 1), hidden_size)
        else:
            self.res_connection = nn.Identity()
        # Initialize as an identity mapping
        torch.nn.init.zeros_(self.linear.weight)
        # Use SiLU activation to keep consistent with the Llama model
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Forward pass of the ResBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the residual connection and activation.
        """
        return self.res_connection(x) + self.act(self.linear(x))

class HydraMLP(nn.Module):
    """
    A MLP module as the Hydra head.

    Args:
        hidden_size (int): The size of the hidden layers in the MLP.
        num_layers (int): The number of hidden layers in the MLP.
    """

    def __init__(
        self,
        hydra_num_layers, 
        hydra_num_heads, 
        grounded_heads, 
        input_embed_fn,
        base_config,
        lm_head_init_weight=None,
    ):
        super().__init__()

        self.hidden_size = base_config.hidden_size
        self.vocab_size = base_config.vocab_size
        
        self.hydra_num_layers = hydra_num_layers
        self.hydra_num_heads = hydra_num_heads
        self.grounded_heads = grounded_heads
        self.input_embed_fn = input_embed_fn

        assert self.hydra_num_layers > 0, "Hydra MLP must have at least one layer."

        if grounded_heads:
            self.hydra_mlp = nn.ModuleList([
                nn.Sequential(
                    ResBlock_Hydra(self.hidden_size, hydra_head_idx + 1),
                    *([ResBlock(self.hidden_size)] * (self.hydra_num_layers - 1))
                ) for hydra_head_idx in range(self.hydra_num_heads)
            ])
        else:
            self.hydra_mlp = nn.ModuleList([
                nn.Sequential(
                    *([ResBlock_Hydra(self.hidden_size)] * self.hydra_num_layers)
                ) for hydra_head_idx in range(self.hydra_num_heads)
            ])
        
        self.hydra_lm_head = nn.ModuleList([
            nn.Linear(self.hidden_size, self.vocab_size) for _ in range(self.hydra_num_heads)
        ])
        if lm_head_init_weight is not None:
            print("Initializing HydraLM head with pretrained weights...")
            for i in range(hydra_num_heads):
            # Initialize the weights of each hydra_head using the base model's weights
                self.hydra_lm_head[i].weight.data[:] = lm_head_init_weight[:]
        # else:
        #     print("Initializing HydraLM head with backbone model weights...")
        #     for i in range(hydra_num_heads):
        #     # Initialize the weights of each hydra_head using the base model's weights
        #         self.hydra_lm_head[i].weight.data[:] = self.lm_head.weight.data[:]

    def forward(self, base_hidden_states, input_ids=None, noise=None):
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the MLP.
        """

        hydra_hidden_states = []
        if self.grounded_heads:
            assert input_ids is not None, "Input ids must be provided for grounded heads"
            with torch.inference_mode():
                input_embeds = self.input_embed_fn(input_ids)
            if noise is not None:
                input_embeds = input_embeds + noise
            hydra_inputs = [base_hidden_states]
            for i in range(self.hydra_num_heads):
                # Move input embeddings back one spot for each hydra head idx
                hydra_inputs.append(torch.roll(input_embeds, shifts=-(i+1), dims=1))
            
            for i in range(self.hydra_num_heads):
                head_input = torch.cat(hydra_inputs[:i + 2], dim=-1) 
                hydra_hidden_states.append(self.hydra_mlp[i](head_input))
        else:
            for i in range(self.hydra_num_heads):
                hydra_hidden_states.append(self.hydra_mlp[i](base_hidden_states))
        
        hydra_logits = []
        for i in range(self.hydra_num_heads):
            hydra_logits.append(self.hydra_lm_head[i](hydra_hidden_states[i]))
        
        return hydra_logits, hydra_hidden_states

    def _ungrounded_proposal(self, input_logits, base_hidden_states, hydra_buffers):
        hydra_logits = []
        for i in range(self.hydra_num_heads):
            hydra_hidden_state = self.hydra_mlp[i](base_hidden_states)
            hydra_logits.append(self.hydra_lm_head[i](hydra_hidden_state))
        hydra_logits = torch.stack(hydra_logits, dim=0)

        # Greedy decoding: Select the most probable candidate from the original logits.
        candidates_logit = torch.argmax(input_logits[:, -1]).unsqueeze(0)

        # Extract the TOPK candidates from the hydra logits.
        candidates_hydra_logits = []
        for hydra_head, beam_size in enumerate(hydra_buffers["beam_sizes"]):
            candidates_hydra_logits.append(torch.topk(hydra_logits[hydra_head, 0, -1], beam_size, dim = -1).indices)
        candidates_hydra_logits = torch.cat(candidates_hydra_logits)

        # Combine the selected candidate from the original logits with the topk hydra logits.
        candidates = torch.cat([candidates_logit, candidates_hydra_logits.view(-1)], dim=-1)

        # Map the combined candidates to the tree indices to get tree candidates.
        tree_candidates = candidates[hydra_buffers["tree_indices"]]

        # Extend the tree candidates by appending a zero.
        tree_candidates_ext = torch.cat([tree_candidates, torch.zeros((1), dtype=torch.long, device=tree_candidates.device)], dim=0)

        # Retrieve the cartesian candidates using the retrieve indices.
        cart_candidates = tree_candidates_ext[hydra_buffers["retrieve_indices"]]

        # Unsqueeze the tree candidates for dimension consistency.
        tree_candidates = tree_candidates.unsqueeze(0)
        return cart_candidates, tree_candidates
    
    def _grounded_proposal(self, input_logits, base_hidden_states, hydra_buffers):
        children_per_head = hydra_buffers["children_per_head"]
        children_to_expand_per_head = hydra_buffers["children_to_expand_per_head"]
        retrieve_indices = hydra_buffers["retrieve_indices"]

        candidate_id = torch.argmax(input_logits[:, -1]).unsqueeze(0)
        candidate_embedding = self.input_embed_fn(candidate_id).unsqueeze(0)

        candidates = torch.tensor([candidate_id], device=candidate_id.device)[None, ...]
        candidates_embeddings = torch.cat([base_hidden_states[:, -1:], candidate_embedding], dim=-1)

        for head_idx, (head_num_children, head_children_to_expand) in enumerate(zip(children_per_head, children_to_expand_per_head)):
            hydra_hidden_state = self.hydra_mlp[head_idx](candidates_embeddings)
            hydra_preds = self.hydra_lm_head[head_idx](hydra_hidden_state)
            next_head_embeddings = []

            for path_idx, (num_children, children_to_expand) in enumerate(zip(head_num_children, head_children_to_expand)):

                hydra_candidates = torch.topk(hydra_preds[:, path_idx], num_children, dim=-1).indices
                candidates = torch.cat([candidates, hydra_candidates], dim=-1)
                
                if children_to_expand > 0:
                    children_embeddings = self.input_embed_fn(hydra_candidates)[:, :children_to_expand]
                    repeat_slice = [path_idx] * children_to_expand
                    path_embeddings = candidates_embeddings[:, repeat_slice]
                    next_head_embeddings.append(torch.cat([path_embeddings, children_embeddings], dim=-1))
            
            if len(next_head_embeddings):
                # TODO (Zack): Determine assertion error about next_head_embeddings being empty before finishing tree
                candidates_embeddings = torch.cat(next_head_embeddings, dim=1)

        # TODO (Zack): Only selecting first batch element for now, change when doing bs > 1
        cart_candidates = candidates[0, retrieve_indices]

        return cart_candidates, candidates

    def proposal(
            self,
            input_logits,
            base_hidden_states,
            hydra_buffers,
            past_key_values=None, # Not actually used but consistent with other proposal functions,
            input_ids = None
        ):
        if self.grounded_heads:
            return self._grounded_proposal(input_logits, base_hidden_states, hydra_buffers)
        else:
            return self._ungrounded_proposal(input_logits, base_hidden_states, hydra_buffers)


def add_hydra_heads(
    self,
    medusa_num_heads=4,
    medusa_num_layers=0,
):
    """
    Args:
        self (nn.Module): The base language model to be used.
        medusa_num_heads (int, optional): Number of additional tokens to predict. Defaults to 3.
        medusa_num_layers (int, optional): Number of ResBlock layers for each Medusa head. Defaults to 0.
    """
    hidden_size = self.lm_head.weight.shape[-1]
    vocab_size = self.lm_head.weight.shape[0]
    self.config.hydra_num_layers = medusa_num_layers
    self.config.hydra_num_heads = medusa_num_heads
    self.hydra_num_heads = medusa_num_heads
    self.hidden_state_offset=0
    self.dropout_rate = 0.0
    self.grounded_heads = False
    self.tokenizer = None
    # AutoTokenizer.from_pretrained(self.base_model_name_or_path)
    # self.model = self.base_model.model
    # Create a list of Medusa heads
    # self.medusa_head = nn.ModuleList(
    #     [
    #         nn.Sequential(
    #             *([ResBlock(hidden_size)] * medusa_num_layers),
    #             nn.Linear(hidden_size, vocab_size, bias=False),
    #         )
    #         for _ in range(medusa_num_heads)
    #     ]
    # )
    self.hydra = HydraMLP(medusa_num_layers,
             self.hydra_num_heads,
             True,
             self.model.embed_tokens,
             self.config,
            #  lm_head_init_weight=self.lm_head.weight.data[:]
             )
    print(self.config)
    self.medusa_head = self.hydra.hydra_lm_head
    self.hydra_head = self.hydra.hydra_mlp

    # Ensure medusa_head's dtype and device align with the base_model
    self.hydra.hydra_lm_head.to(self.dtype).to(self.device)

    # Copy weights over from base model
    for i in range(medusa_num_heads):
        # Initialize the weights of each medusa_head using the base model's weights
        self.hydra.hydra_lm_head[i].weight.data[:] = self.lm_head.weight.data[:]

    self.old_forward = self.forward

    # TODO: need to fix this! use the hydraMLP class directly

    # def forward(
    #     self,
    #     input_ids: torch.LongTensor = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     position_ids: Optional[torch.LongTensor] = None,
    #     past_key_values: Optional[List[torch.FloatTensor]] = None,
    #     inputs_embeds: Optional[torch.FloatTensor] = None,
    #     labels: Optional[torch.LongTensor] = None,
    #     use_cache: Optional[bool] = None,
    #     output_attentions: Optional[bool] = None,
    #     output_hidden_states: Optional[bool] = None,
    #     return_dict: Optional[bool] = None,
    #     medusa_return: bool = False,
    #     medusa_only_heads: bool = False,
    # ):
    #     """Forward pass of the MedusaModel.
    #     Returns:
    #         torch.Tensor: A tensor containing predictions from all Medusa heads.
    #         (Optional) Original predictions from the base model's LM head.
    #     """
    #     # LOG.debug("medusa_return: %s", medusa_return)
    #     if not medusa_return:
    #         return self.old_forward(
    #             input_ids=input_ids,
    #             attention_mask=attention_mask,
    #             position_ids=position_ids,
    #             past_key_values=past_key_values,
    #             inputs_embeds=inputs_embeds,
    #             use_cache=use_cache,
    #             output_attentions=output_attentions,
    #             output_hidden_states=output_hidden_states,
    #             return_dict=return_dict,
    #         )
    #     # Pass input through the base model
    #     if medusa_only_heads:
    #         with torch.no_grad():
    #             outputs = self.model(
    #                 input_ids=input_ids,
    #                 attention_mask=attention_mask,
    #                 position_ids=position_ids,
    #                 past_key_values=past_key_values,
    #                 inputs_embeds=inputs_embeds,
    #                 use_cache=use_cache,
    #                 output_attentions=output_attentions,
    #                 output_hidden_states=output_hidden_states,
    #                 return_dict=return_dict,
    #             )
    #             hidden_states = outputs[0]
    #             medusa_logits = [self.lm_head(hidden_states)]
    #     else:
    #         outputs = self.model(
    #             input_ids=input_ids,
    #             attention_mask=attention_mask,
    #             position_ids=position_ids,
    #             past_key_values=past_key_values,
    #             inputs_embeds=inputs_embeds,
    #             use_cache=use_cache,
    #             output_attentions=output_attentions,
    #             output_hidden_states=output_hidden_states,
    #             return_dict=return_dict,
    #         )
    #         hidden_states = outputs[0]
    #         medusa_logits = [self.lm_head(hidden_states)]
    #     for i in range(self.medusa_num_heads):
    #         medusa_logits.append(self.medusa_head[i](hidden_states))
    #     return torch.stack(medusa_logits, dim=0)
    

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        past_key_values=None,
        output_orig=False,
        position_ids=None,
        run_hydra_head=True,
        base_hidden_states=None,
        noise_alpha=0.0,
        medusa_return=False,
        medusa_only_heads=False,
        hydra_logits_only=True
    ):
        """Forward pass of the HydraModel.
        # TODO, graft this onto axolotl

        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            labels (torch.Tensor, optional): Ground truth labels for loss computation.
            past_key_values (tuple, optional): Tuple containing past key and value states for attention.
            output_orig (bool, optional): Whether to also output predictions from the original LM head.
            position_ids (torch.Tensor, optional): Position IDs.

        Returns:
            torch.Tensor: A tensor containing predictions from all Hydra heads.
            (Optional) Original predictions from the base model's LM head.
        """
        if not medusa_return:
            return self.old_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                # **kwargs,
            )
        if base_hidden_states is not None:
            with torch.inference_mode():
                outputs = None
                if output_orig:
                    orig_logits = self.orig_lm_head(base_hidden_states)
        else:
            with torch.inference_mode():
                # Pass input through the base model
                outputs = self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    output_hidden_states=self.hidden_state_offset != 0,
                )

                if output_orig:
                    orig_logits = self.base_model.lm_head(outputs[0])

            # Clone the output hidden states
            if self.hidden_state_offset == 0:
                base_hidden_states = outputs[0].clone()
            else:
                base_hidden_states = outputs[1][-(self.hidden_state_offset + 1)].clone()
        
        # Hydra heads only queried in model forward during training
        if not run_hydra_head:
            assert output_orig, "Must output original predictions if not running Hydra head."
            return None, outputs, orig_logits, base_hidden_states
        
        # From NEFT-tune
        # model_dim = base_hidden_states.shape[-1]
        # seq_len = (input_ids != self.tokenizer.pad_token_id).sum(dim=-1).clamp(min=1).unsqueeze(1).unsqueeze(2)
        # denom = torch.sqrt(seq_len * model_dim)

        # noise = (torch.rand_like(base_hidden_states) * 2 - 1) * noise_alpha / denom
        # noise = noise.to(base_hidden_states.dtype)
        # input_base_hidden_states = base_hidden_states + noise


        hydra_logits, hydra_hidden_states = self.hydra(
            base_hidden_states=base_hidden_states, input_ids=input_ids
        )
        if hydra_logits_only:
            return torch.stack(hydra_logits, dim=0)
        if output_orig:
            return torch.stack(hydra_logits, dim=0), torch.stack(hydra_hidden_states, dim=0), outputs, orig_logits, base_hidden_states
        return torch.stack(hydra_logits, dim=0), torch.stack(hydra_hidden_states, dim=0), outputs


    self.forward = types.MethodType(forward, self)





def replace_compute_loss(
    medusa_heads_coefficient,
    medusa_decay_coefficient, 
    medusa_scheduler="constant",
    medusa_logging=False,
    medusa_only_heads=False,
    medusa_distillation_regularization=0.0,
    medusa_self_distillation=False,
):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute the training loss for the model.

        Args:
            model (torch.nn.Module): The model for which to compute the loss.
            inputs (dict): The input data, including input IDs, attention mask, and labels.
            return_outputs (bool): Whether to return model outputs along with the loss.

        Returns:
            Union[float, Tuple[float, torch.Tensor]]: The computed loss, optionally with model outputs.
        """
        if medusa_self_distillation:
            from peft.tuners.tuners_utils import BaseTunerLayer
            with torch.inference_mode():
                # Get the output of the original model for distillation
                for module in model.modules():
                    if isinstance(module, (BaseTunerLayer)):
                        module.enable_adapters(False)
                # original lm logits, TODO: print this to investigate batch size
                # print inputs as well.
                original_logits = model(
                    **inputs,
                    medusa_return=False,
                ).logits

                for module in model.modules():
                    if isinstance(module, (BaseTunerLayer)):
                        module.enable_adapters(True)
        # original lm + heads batched
        # print("inputs")
        # print(inputs)
        # print(inputs["labels"].shape)
        original_logits = model(
                    **inputs,
                    medusa_return=False,
                ).logits
        # print("original logits")
        # print(original_logits)
        # print(original_logits.shape)
        logits = model(
            **inputs,
            medusa_return=True,
            medusa_only_heads=medusa_only_heads,
        )
        labels = inputs["labels"]
        # Shift so that tokens < n predict n
        loss = 0
        loss_fct = CrossEntropyLoss()
        log = {}
        medusa = logits.shape[0]
        for i in range(medusa):
            # 
            # print("logits")
            # print(logits)
            # print(logits.shape)
            # print("labels")
            # print(labels)
            # print(labels.shape)
            # logits: [ith medusa head, batch, context length, vocab size]
            # logits: [6, 1, 4096, 32000]
            medusa_logits = logits[i, :, : -(1 + i)].contiguous()
            # labels: [1, 4096]
            medusa_labels = labels[..., 1 + i :].contiguous()
           
            medusa_logits = medusa_logits.view(-1, logits.shape[-1])
            medusa_labels = medusa_labels.view(-1)
            medusa_labels = medusa_labels.to(medusa_logits.device)

            # print(f"medusas logits processed (head {i})")
            # print(medusa_logits)
            # print(medusa_logits.shape)
            # print("labels processed")
            # print(medusa_labels)
            # print(medusa_labels.shape)
            if i == 0:
                if medusa_self_distillation:
                    original_logits = original_logits[:, :-1].contiguous().view(-1, original_logits.shape[-1])
                    mask = medusa_labels.ne(IGNORE_TOKEN_ID)
                    soft_labels = F.softmax(original_logits[mask], dim=-1)
                    loss_i = F.kl_div(
                        F.log_softmax(medusa_logits[mask], dim=-1),
                        soft_labels,
                        reduction="sum",
                    ) / medusa_logits.shape[0]
                elif medusa_distillation_regularization > 0:
                    # use soft labels
                    mask = medusa_labels.ne(IGNORE_TOKEN_ID)
                    soft_labels = F.softmax(medusa_logits[mask], dim=-1) * medusa_distillation_regularization + \
                        F.one_hot(medusa_labels[mask], num_classes=medusa_logits.shape[-1]) * (1 - medusa_distillation_regularization)
                    loss_i = F.kl_div(
                        F.log_softmax(medusa_logits[mask], dim=-1),
                        soft_labels,
                        reduction="sum",
                    ) / medusa_logits.shape[0]
                else:
                    loss_i = loss_fct(medusa_logits, medusa_labels)
            else:
                loss_i = loss_fct(medusa_logits, medusa_labels)
            # Compute the coefficient for medusa losses
            if medusa_scheduler == "sine":
                medusa_scheduler_coefficient = math.sin(
                    self.state.global_step / self.state.max_steps * math.pi / 2
                )
            elif medusa_scheduler == "linear":
                medusa_scheduler_coefficient = (
                    self.state.global_step / self.state.max_steps
                )
            elif medusa_scheduler == "constant":
                medusa_scheduler_coefficient = 1
            elif medusa_scheduler.startswith("sine"):
                ratio = float(medusa_scheduler.split("_")[1])
                if self.state.global_step / self.state.max_steps < ratio:
                    medusa_scheduler_coefficient = math.sin(
                        self.state.global_step / self.state.max_steps / ratio * math.pi / 2
                    )
                else:
                    medusa_scheduler_coefficient = 1
            else:
                raise ValueError(
                    f"Invalid medusa_scheduler: {medusa_scheduler}. "
                    "Must be one of 'sine', 'linear', or 'constant'."
                )
            # Add decay coefficient to the loss
            if i == 0:
                if not medusa_only_heads:
                    loss += loss_i
            else:
                loss += loss_i * medusa_decay_coefficient ** i * medusa_heads_coefficient * medusa_scheduler_coefficient
            not_ignore = medusa_labels.ne(IGNORE_TOKEN_ID)
            medusa_labels = medusa_labels[not_ignore]

            # Add top-k accuracy
            for k in range(1, 10):
                _, topk = medusa_logits.topk(k, dim=-1)
                topk = topk[not_ignore]
                correct = topk.eq(medusa_labels.unsqueeze(-1)).any(-1)
                log[f"medusa{i}_top{k}"] = correct.float().mean().item()

            log[f"medusa{i}_loss"] = loss_i.item()
            log["medusa_scheduler_coefficient"] = medusa_scheduler_coefficient
        # self.log(log)
        # Add prefix to the log
        if model.training:
            prefix = "train"
        else:
            prefix = "eval"
        log = {f"{prefix}/{k}": v for k, v in log.items()}
        if medusa_logging and self.state.is_world_process_zero:
            # Hardcoded for now
            wandb.log({
                **log,
                "train/global_step": self.state.global_step,
            })
        return (loss, logits) if return_outputs else loss
    transformers.trainer.Trainer.compute_loss = compute_loss

def replace_create_optimizer(
    medusa_lr_multiplier,
):
    # Copy from transformers.Trainer.create_optimizer
    from transformers.trainer import is_sagemaker_mp_enabled, Trainer, ShardedDDPOption
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            # Separately set lr for medusa_head
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and "medusa_head" not in n)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and "medusa_head" in n)
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate * medusa_lr_multiplier,
                },
                
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
                if optimizer_cls.__name__ == "Adam8bit":
                    import bitsandbytes

                    manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                    skipped = 0
                    for module in opt_model.modules():
                        if isinstance(module, nn.Embedding):
                            skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                            logger.info(f"skipped {module}: {skipped/2**20}M params")
                            manager.register_module_override(module, "weight", {"optim_bits": 32})
                            logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                    logger.info(f"skipped: {skipped/2**20}M params")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer
    transformers.trainer.Trainer.create_optimizer = create_optimizer

    # Fix deepspeed's optimizer
    def deepspeed_init(trainer, num_training_steps, inference=False):
        """
        Init DeepSpeed, after updating the DeepSpeed configuration with any relevant Trainer's args.

        If `resume_from_checkpoint` was passed then an attempt to resume from a previously saved checkpoint will be made.

        Args:
            trainer: Trainer object
            num_training_steps: per single gpu
            resume_from_checkpoint: path to a checkpoint if to resume from after normal DeepSpeedEngine load
            inference: launch in inference mode (no optimizer and no lr scheduler)

        Returns: optimizer, lr_scheduler

        We may use `deepspeed_init` more than once during the life of Trainer, when we do - it's a temp hack based on:
        https://github.com/microsoft/DeepSpeed/issues/1394#issuecomment-937405374 until Deepspeed fixes a bug where it
        can't resume from a checkpoint after it did some stepping https://github.com/microsoft/DeepSpeed/issues/1612

        """
        from deepspeed.utils import logger as ds_logger
        from transformers.integrations.deepspeed import deepspeed_optim_sched

        model = trainer.model
        args = trainer.args

        hf_deepspeed_config = trainer.accelerator.state.deepspeed_plugin.hf_ds_config

        # resume config update - some bits like `model` and `num_training_steps` only become available during train
        hf_deepspeed_config.trainer_config_finalize(args, model, num_training_steps)

        # set the Deepspeed log level consistent with the Trainer
        ds_logger.setLevel(args.get_process_log_level())

        if inference:
            # only Z3 makes sense for the inference
            if not hf_deepspeed_config.is_zero3():
                raise ValueError("ZeRO inference only makes sense with ZeRO Stage 3 - please adjust your config")

            # in case the training config is re-used for inference
            hf_deepspeed_config.del_config_sub_tree("optimizer")
            hf_deepspeed_config.del_config_sub_tree("lr_scheduler")
            optimizer, lr_scheduler = None, None
            model_parameters = None
        else:
            trainer.optimizer = None  # important for when deepspeed_init is used as re-init
            self = trainer
            opt_model = model
            decay_parameters = self.get_decay_parameter_names(opt_model)
            model_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and "medusa_head" not in n)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and "medusa_head" in n)
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate * medusa_lr_multiplier,
                },
                
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]
            
            # list(filter(lambda p: p.requires_grad, model.parameters()))
            optimizer, lr_scheduler = deepspeed_optim_sched(
                trainer, hf_deepspeed_config, args, num_training_steps, model_parameters
            )

        # keep for quick debug:
        # from pprint import pprint; pprint(config)

        return optimizer, lr_scheduler
    transformers.integrations.deepspeed.deepspeed_init = deepspeed_init