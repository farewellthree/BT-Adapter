import os
import torch
import torch.nn as nn
from transformers import Trainer
from typing import Optional
from transformers.trainer import is_sagemaker_mp_enabled, get_parameter_names, ALL_LAYERNORM_LAYERS, \
    ShardedDDPOption, logger#, smp#, OSS


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model


class BTAdapterLLavaTrainer(Trainer):
    
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        _state_dict = state_dict
        if _state_dict is None:
            # Only save the model itself if we are using distributed training
            model_to_save = unwrap_model(self.model)
            _state_dict = model_to_save.state_dict()
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            # Save the model
            weight_to_save = {}
            keys_to_match = ['mm_projector', 'embed_tokens', 'embed_in']
            for k, v in _state_dict.items():
                if any(key_match in k for key_match in keys_to_match):
                    weight_to_save[k] = v

            current_folder = output_dir.split('/')[-1]
            parent_folder = os.path.dirname(output_dir)
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'), )

        if getattr(self.args, 'mm_vision_tower', False):
            weight_to_save = {}
            keys_to_match = ['btadapter']
            for k, v in _state_dict.items():
                if any(key_match in k for key_match in keys_to_match):
                    weight_to_save[k] = v
            current_folder = output_dir.split('/')[-1]
            parent_folder = os.path.dirname(output_dir)
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "btadapter_weight")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'btadapter_weight.bin'), )

        # super(BTAdapterLLavaTrainer, self)._save(output_dir, state_dict)
    
    
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            all_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            valid_parameter = [name for name, param in opt_model.named_parameters() if param.requires_grad]
            decay_parameters = [name for name in all_parameters if "bias" not in name]
            nodecay_parameters = [name for name in all_parameters if name not in decay_parameters]

            CLIP_para = set_weight_decay(opt_model, nodecay_parameters, (), 
                self.args.weight_decay, lr=self.args.learning_rate*self.args.clip_lr, 
                have=("vision_tower",), not_have=('btadapter',)
            )
            BTAdapter_para = set_weight_decay(opt_model, nodecay_parameters, (), 
                self.args.weight_decay, lr=self.args.learning_rate, 
                have=("btadapter",), not_have=()
            )
            proj_para = set_weight_decay(opt_model, nodecay_parameters, (), 
                self.args.weight_decay, lr=self.args.learning_rate, 
                have=("mm_projector","embed_tokens",), not_have=()
            )
            llama_para = set_weight_decay(opt_model, nodecay_parameters, (), 
                self.args.weight_decay, lr=self.args.learning_rate*self.args.llama_lr, 
                have=(), not_have=("vision_tower","mm_projector","embed_tokens",)
            )
            optimizer_grouped_parameters = CLIP_para + BTAdapter_para + proj_para + llama_para

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
    

def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin
    
def set_weight_decay(model, skip_list=(), skip_keywords=(), weight_decay=0.001, lr=2e-6, have=(), not_have=()):
    has_decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(have) > 0 and not check_keywords_in_name(name, have):
            continue
        if len(not_have) > 0 and check_keywords_in_name(name, not_have):
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
        else:
            has_decay.append(param)

    return [{'params': has_decay, 'weight_decay': weight_decay, 'lr': lr},
            {'params': no_decay, 'weight_decay': 0., 'lr': lr}]