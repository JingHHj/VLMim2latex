import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Literal, Union
from safetensors.torch import save_file

        
class LoRALinear(nn.Linear):
    def __init__(self, 
                 in_features, 
                 out_features,
                 bias=True, 
                 rank=8, 
                 lora_alpha=8, 
                 lora_dropout=0.0,
                 use_rslora=True,
                 **kwargs):
        
        nn.Linear.__init__(self, in_features, out_features, bias=bias, **kwargs)
        assert rank > 0

        self.rank = rank
        self.lora_alpha = lora_alpha 
        self.scaling = self.lora_alpha / self.rank**0.5 if use_rslora else self.lora_alpha / self.rank
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else lambda x: x

        self.weight.requires_grad = False
        
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def _merge_weights(self):

        """
        xW^T + xAB = x(W^T + AB)
        """

        merged_weight = self.weight.data + (self.lora_A @ self.lora_B).T * self.scaling
        state_dict = {"weight": merged_weight}
        if self.bias is not None:
            state_dict["bias"] = self.bias

        merged_linear = nn.Linear(self.in_features, 
                                  self.out_features,
                                  bias=True if self.bias is not None else False)
        
        merged_linear.load_state_dict(state_dict)

        return merged_linear
    
    def _load_pretrained_weights(self, state_dict):
        self.weight.data = state_dict["weight"]
        if "bias" in state_dict.keys():
            self.bias.data = state_dict["bias"]

    def forward(self, x):
        orig_layer_out = F.linear(x, self.weight, bias=self.bias)
        lora_mult = (self.lora_A @ self.lora_B) * self.scaling
        low_rank_out = self.lora_dropout(x) @ lora_mult
        output = orig_layer_out + low_rank_out
        return output
    


@dataclass
class LoraConfig:
    rank: int = 8
    target_modules: Optional[Union[list[str], str]] = None
    lora_alpha: float = 8.0
    lora_dropout: float = 0.0
    bias: Literal["none", "all", "lora_only"] = "none"
    use_rslora: bool = True

class LoraModel(nn.Module):

    def __init__(self, model, config):
        super(LoraModel, self).__init__()

        self.lora_model = model
        self.config = config
        ### Ensure Taraget Modules/Exclude Modules are Lists ###
        if self.config.target_modules is None:
            self.config.target_modules = []
        elif isinstance(self.config.target_modules, str):
            self.config.target_modules = [self.config.target_modules]
        if self.config.exclude_modules is None:
            self.config.exclude_modules = []
        elif isinstance(self.config.exclude_modules, str):
            self.config.exclude_modules = [self.config.exclude_modules]

        orig_trainable_params = self._compute_trainable_parameters()
        self._disable_all_grads()
        self._apply_lora(self.lora_model)
        self._toggle_bias_grad()
        lora_trainable_params = self._compute_trainable_parameters()

        # Print out to show the difference 
        print_string = ""
        print_string += f"Initial Parameters : {orig_trainable_params} || "
        print_string += f"LoRA Parameters : {lora_trainable_params} || "
        print_string += f"Trainable Proportion : {round(lora_trainable_params*100/orig_trainable_params, 2)}%"

        print(print_string)

    def forward(self, *inputs, **kwargs):

        """
        The forward function is the same, so a catchall here
        to pass all of our stuff from the forward methdod into
        our models forward method
        """

        return self.lora_model(*inputs, **kwargs)
    
    def _exclude_module_name_check(self, name):
        return any([ex in name for ex in self.config.exclude_modules])
    
    def _target_module_name_check(self, name):
        return any([tgt in name for tgt in self.config.target_modules])
    
    def _apply_lora(self, module):

        """
        Method to recursively replace all the layers in a model with LoraLayers
        """

        # Recursively Go Through Model and Find Layers To Convert
        for name, child in module.named_children():
            # Check if Layer is Included to Convert to LoRA
            if self._target_module_name_check(name):
                if isinstance(child, nn.Linear):  # Convert Linear to LoRA
                    new_layer = LoRALinear(in_features=child.in_features, 
                                           out_features=child.out_features, 
                                           bias=True if child.bias is not None else False,
                                           rank=self.config.rank,
                                           lora_alpha=self.config.lora_alpha, 
                                           lora_dropout=self.config.lora_dropout, 
                                           use_rslora=self.config.use_rslora)

                    new_layer._load_pretrained_weights(child.state_dict())
                    setattr(module, name, new_layer)
            # If there are more children and its not an exclusion module, Recurse into them 
            if (len(list(child.children())) > 0) and not any([ex in name for ex in self.config.exclude_modules]):
                self._apply_lora(child)

    def _toggle_bias_grad(self):

        """
        Method to turn off bias gradients depending on:
            - none:  Dont train any biases
            - all: train all biases
            - lora_only: train biases only in lora layers
        """

        for name, param in self.lora_model.named_parameters():
            if not self._exclude_module_name_check(name):
                if ".bias" in name:
                    if self.config.bias == "none":
                        param.requires_grad = False
                    elif self.config.bias == "all":
                        param.requires_grad = True
                    elif (self.config.bias == "lora_only") and self._target_module_name_check(name):
                        param.requires_grad = True

    def _disable_all_grads(self):
        
        """
        Helper function to disable all gradients 
        """

        for name, param in self.lora_model.named_parameters():
            if not self._exclude_module_name_check(name):
                param.requires_grad = False

    def _compute_trainable_parameters(self):

        """
        Helper function to compute all parameters with gradients
        """

        total_learnable_params = 0
        for param in self.lora_model.parameters():
            if param.requires_grad:
                total_learnable_params += param.numel()

        return total_learnable_params

    def _merge_weights(self, module):

        """
        Recursively trigger weight merging and replace in model 
        """

        for name, child in module.named_children():

            if isinstance(child, (LoRALinear)):
                 
                 # Merge the Layer 
                 merged_layer = child._merge_weights()
                 # Replace LoRA Layer with Merged 
                 setattr(module, name, merged_layer)
            else:
                if len(list(child.children())) > 0:
                    self._merge_weights(child)

    def save_model(self, path, merge_weights=False):

        """
        Method to save model safetensors to the given path
            - merge_weights -> True: Merge LoRA weights and save
            - merge_weights -> False: Only save trainable weights
        """

        def _detach_cpu(param):
            return param.detach().cpu()
    
        if merge_weights:
            self._merge_weights(self.lora_model)
            state_dict = {name.replace("lora_model.", ""): _detach_cpu(param) for (name, param) in self.named_parameters()}
        else:
            state_dict = {name: _detach_cpu(param) for (name, param) in self.named_parameters() if (param.requires_grad)}

        save_file(state_dict, path)

