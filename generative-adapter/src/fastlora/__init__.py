from fastlora.args import ModelArgs

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)


def get_model_and_tokenizer(model_args, device="cpu", evaluation_mode=True, return_tokenizer_only=False, **kwargs):
    """Load model and tokenizer."""
    return get_fastlora_model_and_tokenizer(
        model_args, 
        device=device, 
        evaluation_mode=evaluation_mode, 
        return_tokenizer_only=return_tokenizer_only, 
        **kwargs
    )

    # if model_args.baseline == 'longlora':
    #     return get_longlora_model_and_tokenizer(
    #         model_args, 
    #         device=device, 
    #         evaluation_mode=evaluation_mode, 
    #         return_tokenizer_only=return_tokenizer_only, 
    #         **kwargs
    #     )
    # elif model_args.baseline == 'autocompressor':
    #     return get_autocompressor_model_and_tokenizer(
    #         model_args, 
    #         device=device, 
    #         evaluation_mode=evaluation_mode, 
    #         return_tokenizer_only=return_tokenizer_only, 
    #         **kwargs
    #     )
    # else:
    #     # assert model_args.get("enable_fastlora", False) or model_args.get("enable_ultragist", False), "Please specify the model type"
    #     return get_default_model_and_tokenizer(
    #         model_args, 
    #         device=device, 
    #         evaluation_mode=evaluation_mode, 
    #         return_tokenizer_only=return_tokenizer_only, 
    #         **kwargs
    #     )

def get_fastlora_model_and_tokenizer(model_args, device="cpu", evaluation_mode=True, return_tokenizer_only=False, **kwargs):
    import torch
    from dataclasses import asdict
    from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
    from transformers.utils import logging
    from transformers.integrations import is_deepspeed_zero3_enabled
    from peft import AutoPeftModelForCausalLM

    from fastlora.config import FastLoraConfig
    from fastlora.model import FastLoraModelForCausalLM, FastLoraModel, get_peft_model_state_dict, set_peft_model_state_dict
    import peft.peft_model as peft_model
    import peft.mapping as peft_mapping
    
    ## monkey patching
    peft_model.PEFT_TYPE_TO_MODEL_MAPPING.update({"FASTLORA": FastLoraModel})
    peft_mapping.PEFT_TYPE_TO_CONFIG_MAPPING.update({"FASTLORA": FastLoraConfig})
    peft_model.get_peft_model_state_dict = get_peft_model_state_dict
    peft_model.set_peft_model_state_dict = set_peft_model_state_dict

    logger = logging.get_logger(__name__)

    model_args_dict = asdict(model_args)
    model_args_dict.update(**kwargs)

    model_name_or_path = model_args_dict["model_name_or_path"]
    cache_dir = model_args_dict["model_cache_dir"]
    access_token = model_args_dict["access_token"]

    tokenizer_kwargs = {}
    if model_args_dict["no_use_fast"]:
        tokenizer_kwargs = {"use_fast": False}

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, 
        cache_dir=cache_dir, 
        padding_side=model_args_dict["padding_side"], 
        token=access_token, 
        trust_remote_code=True,
        **tokenizer_kwargs
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if return_tokenizer_only:
        return tokenizer

    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(model_args_dict["dtype"], torch.float32)
        
    device_map = model_args_dict["device_map"]
    if device_map is None and not is_deepspeed_zero3_enabled():
        device_map = {"": device}

    attn_impl = model_args_dict["attn_impl"]

    fastlora_kwargs = {}
    for k, v in model_args_dict.items():
        if (k.startswith("fastlora_")) and v is not None:
            fastlora_kwargs[k] = v
    fastlora_config = FastLoraConfig(
        **fastlora_kwargs,
        task_type="CAUSAL_LM",
        peft_type="FASTLORA",
    )
    # if model_name_or_path is already a FastLora model, then load the model directly
    try:
        # model = AutoPeftModelForCausalLM.from_pretrained(
        #     model_name_or_path,
        #     config=fastlora_config,
        #     torch_dtype=torch_dtype,
        #     attn_implementation=attn_impl,
        # )
        from peft.config import PeftConfig
        peft_config = PeftConfig.from_pretrained(model_name_or_path, **kwargs)

        # FIXME: temporary solution to load base model
        peft_config.fastlora_training_attention_mask = fastlora_kwargs.get("fastlora_training_attention_mask", None)

        base_model_path = peft_config.base_model_name_or_path
        assert base_model_path is not None, "base_model_name_or_path should not be None"
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation=attn_impl,
            token=access_token,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )
        peft_config.task_type = "FAST_LORA_CAUSAL_LM"
        model = FastLoraModelForCausalLM.from_pretrained(
            base_model,
            model_name_or_path,
            adapter_name='default',
            is_trainable=False,
            config=peft_config,
            **kwargs,
        )
    except ValueError:
        # HACK: if model_name_or_path is a locol model, then load the model from fastlora.model.from_pretrained_v1
        if "data/" in model_name_or_path:
            from fastlora.model import from_pretrained_v1
            model = from_pretrained_v1(
                model_name_or_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
                attn_implementation=attn_impl,
            )
            print(f'[INFO] load v1 model from {model_name_or_path}')
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
                attn_implementation=attn_impl,
                token=access_token,
                trust_remote_code=True,
                cache_dir=cache_dir,
            )
            model = FastLoraModelForCausalLM(model, fastlora_config)
    
    model = model.eval()
    logger.info(model.config)

    if evaluation_mode:
        # NOTE: essential to disable all gradient in-place, so that when calling accelerator.prepare, the forward function will not be wrapped that may consume extra GPU memory
        model.requires_grad_(False)


    return model, tokenizer


# def get_default_model_and_tokenizer(model_args, device="cpu", evaluation_mode=True, return_tokenizer_only=False, **kwargs):
    
#     import torch
#     import transformers
#     from dataclasses import asdict
#     from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
#     from transformers.utils import logging
#     from transformers.integrations import is_deepspeed_zero3_enabled
#     from packaging import version

#     logger = logging.get_logger(__name__)

#     model_args: ModelArgs

#     model_args_dict = asdict(model_args)
#     model_args_dict.update(**kwargs)
    
#     model_name_or_path = model_args_dict["model_name_or_path"]
#     cache_dir = model_args_dict["model_cache_dir"]
#     access_token = model_args_dict["access_token"]

#     logger.info(f"Loading model and tokenizer from {model_name_or_path}...")

#     tokenizer_kwargs = {}
#     if model_args_dict["no_use_fast"]:
#         tokenizer_kwargs = {"use_fast": False}

#     tokenizer = AutoTokenizer.from_pretrained(
#         model_name_or_path, 
#         cache_dir=cache_dir, 
#         padding_side=model_args_dict["padding_side"], 
#         token=access_token, 
#         trust_remote_code=True,
#         **tokenizer_kwargs
#     )
#     if tokenizer.pad_token_id is None:
#         tokenizer.pad_token_id = tokenizer.eos_token_id
    
#     if return_tokenizer_only:
#         return tokenizer

#     dtype = model_args_dict["dtype"]
#     if dtype == "bf16":
#         dtype = torch.bfloat16
#     elif dtype == "fp16":
#         dtype = torch.float16
#     else:
#         dtype = torch.float32
        
#     device_map = model_args_dict["device_map"]
#     if device_map is None and not is_deepspeed_zero3_enabled():
#         device_map = {"": device}
    
#     rope_kwargs = {}
#     rope_theta = model_args_dict["rope_theta"]
#     if rope_theta is not None:
#         rope_kwargs["rope_theta"] = rope_theta
#     rope_method = model_args_dict["rope_method"]
#     if rope_method is not None:
#         rope_factor = model_args_dict["rope_factor"]
#         rope_scaling = {
#             "type": rope_method,
#             "factor": rope_factor
#         }
#         # NOTE: do not destroy the default rope_scaling of the model
#         rope_kwargs["rope_scaling"] = rope_scaling

#     attn_kwargs = {}
#     attn_impl = model_args_dict["attn_impl"]
#     if attn_impl is not None:
#         if version.parse(transformers.__version__) <= version.parse("4.36"):
#             if attn_impl == "flash_attention_2":
#                 attn_kwargs["use_flash_attention_2"] = True
#         else:
#             attn_kwargs["attn_implementation"] = attn_impl

#     # from_pretrained_kwargs = {}
#     # if attn_impl == "flash_attention_2" and version.parse(transformers.__version__) <= version.parse("4.36"):
#     #     from_pretrained_kwargs["use_flash_attention_2"] = True


#     # use architecture attribute to distinguish different models
#     probe_config = AutoConfig.from_pretrained(
#         model_name_or_path, 
#         cache_dir=cache_dir, 
#         token=access_token, 
#         trust_remote_code=True
#     )
#     architecture = probe_config.architectures[0]

#     extra_kwargs = {}
#     if model_args_dict["max_position_embeddings"] is not None:
#         extra_kwargs["max_position_embeddings"] = model_args_dict["max_position_embeddings"]
#     if architecture == "MistralForCausalLM" and model_args_dict["mistral_sliding_window"] is not None:
#         extra_kwargs["sliding_window"] = model_args_dict["mistral_sliding_window"]
#     if model_args_dict["load_in_4_bit"]:
#         extra_kwargs["quantization_config"] = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_quant_type="nf4",
#             bnb_4bit_use_double_quant=True,
#             bnb_4bit_compute_dtype=dtype,
#         )
#         device_map = None

#     if model_args_dict.get("enable_fastlora", False):
#         from fastlora.mistral import MistralForCausalLM, MistralConfig
#         ARCHITECTURE_TO_CLASS = {
#             'MistralForCausalLM': (MistralConfig, MistralForCausalLM),
#         }
#         config_class, model_class = ARCHITECTURE_TO_CLASS[architecture]

#         fastlora_kwargs = {}
#         for k, v in model_args_dict.items():
#             if (k.startswith("fastlora_") or k.startswith("lora_")) and v is not None:
#                 fastlora_kwargs[k] = v
        
#         config = config_class.from_pretrained(
#             model_name_or_path, 
#             cache_dir=cache_dir,
#             token=access_token,
#             **fastlora_kwargs,
#             **rope_kwargs,
#             **attn_kwargs,
#             **extra_kwargs,
#         )
#         model = model_class.from_pretrained(
#             model_name_or_path, 
#             config=config,
#             cache_dir=cache_dir, 
#             torch_dtype=dtype,
#             device_map=device_map, 
#             token=access_token,
#         )
#     # elif model_args_dict.get("enable_ultragist", False):
#     #     from ultragist.llama import LlamaForCausalLM, LlamaConfig
#     #     from ultragist.mistral import MistralForCausalLM, MistralConfig
#     #     ARCHITECTURE_TO_CLASS = {
#     #         'LlamaForCausalLM': (LlamaConfig, LlamaForCausalLM),
#     #         'MistralForCausalLM': (MistralConfig, MistralForCausalLM),
#     #     }

#     #     config_class, model_class = ARCHITECTURE_TO_CLASS[architecture]

#     #     ultragist_kwargs = {}
#     #     for k, v in model_args_dict.items():
#     #         if k.startswith("ultragist") and v is not None:
#     #             ultragist_kwargs[k] = v
#     #     config = config_class.from_pretrained(
#     #         model_name_or_path, 
#     #         cache_dir=cache_dir,
#     #         token=access_token,
#     #         **ultragist_kwargs,
#     #         **rope_kwargs,
#     #         **attn_kwargs,
#     #         **extra_kwargs,
#     #     )
#     #     model = model_class.from_pretrained(
#     #         model_name_or_path, 
#     #         config=config,
#     #         cache_dir=cache_dir, 
#     #         torch_dtype=dtype,
#     #         device_map=device_map, 
#     #         token=access_token,
#     #     )
#     else:
#         model = AutoModelForCausalLM.from_pretrained(
#             model_name_or_path, 
#             cache_dir=cache_dir, 
#             torch_dtype=dtype,
#             device_map=device_map,
#             token=access_token,
#             trust_remote_code=True,

#             # NOTE: do not destroy the default rope_scaling of the model
#             **rope_kwargs,
#             **attn_kwargs,
#             **extra_kwargs,
#         )

#     # load lora
#     if model_args_dict["lora"] is not None:
#         logger.info(f"loading lora from {model_args_dict['lora']}...")

#         from peft import PeftModel
#         model = PeftModel.from_pretrained(
#             model, 
#             model_args_dict["lora"],
#             torch_dtype=dtype,
#             device_map=device_map,
#         )
#         if model_args_dict["lora_unload"]:
#             model = model.merge_and_unload()


#     if model_args_dict["enable_tp"]:
#         import tensor_parallel as tp
#         logger.info("enabling tensor parallelism...")
        
#         # model = tp.tensor_parallel(model, device_ids=list(range(8)), distributed=False, sharded=False)
#         model = tp.tensor_parallel(model, sharded=True)

#         if model.generation_config.eos_token_id == 128001:
#             model.generation_config.eos_token_id = [128001, 128009]

#     model = model.eval()
#     logger.info(model.config)

#     if evaluation_mode:
#         # NOTE: essential to disable all gradient in-place, so that when calling accelerator.prepare, the forward function will not be wrapped that may consume extra GPU memory
#         model.requires_grad_(False)

#     # override the default generation config
#     generation_config = model_args.get_generation_config()
#     if len(generation_config):
#         unused_config = model.generation_config.update(**generation_config)
#         if len(unused_config):
#             logger.warning(f"The following attributes are not used when overriding the generation configurations: {unused_config}")
#     logger.info(f"Generation config: {generation_config}")

#     return model, tokenizer


# def get_longlora_model_and_tokenizer(model_args, device="cpu", evaluation_mode=True, return_tokenizer_only=False, **kwargs):
#     """Load model and tokenizer."""
#     import os
#     import torch
#     import transformers
#     from dataclasses import asdict
#     from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
#     from transformers.utils import logging
#     from transformers.integrations import is_deepspeed_zero3_enabled
#     from packaging import version

#     from .args import ModelArgs

#     logger = logging.get_logger(__name__)

#     model_args: ModelArgs

#     model_args_dict = asdict(model_args)
#     model_args_dict.update(**kwargs)
    
#     model_name_or_path = model_args_dict["model_name_or_path"]
#     cache_dir = model_args_dict["model_cache_dir"]
#     access_token = model_args_dict["access_token"]

#     logger.info(f"Loading model and tokenizer from {model_name_or_path}...")

#     tokenizer_kwargs = {}
#     if model_args_dict["no_use_fast"]:
#         tokenizer_kwargs = {"use_fast": False}

#     tokenizer = AutoTokenizer.from_pretrained(
#         model_name_or_path, 
#         cache_dir=cache_dir, 
#         padding_side=model_args_dict["padding_side"], 
#         token=access_token, 
#         trust_remote_code=True,
#         **tokenizer_kwargs
#     )
#     if tokenizer.pad_token_id is None:
#         tokenizer.pad_token_id = tokenizer.eos_token_id
    
#     if return_tokenizer_only:
#         return tokenizer

#     if model_args_dict["longlora_s2_attn"]:
#         from baselines.longlora import replace_llama_attn
#         replace_llama_attn(use_flash_attn=True, use_full=False)

#     dtype = model_args_dict["dtype"]
#     if dtype == "bf16":
#         dtype = torch.bfloat16
#     elif dtype == "fp16":
#         dtype = torch.float16
#     else:
#         dtype = torch.float32
        
#     device_map = model_args_dict["device_map"]
#     if device_map is None and not is_deepspeed_zero3_enabled():
#         device_map = {"": device}
    
#     rope_kwargs = {}
#     rope_theta = model_args_dict["rope_theta"]
#     if rope_theta is not None:
#         rope_kwargs["rope_theta"] = rope_theta
#     rope_method = model_args_dict["rope_method"]
#     if rope_method is not None:
#         rope_factor = model_args_dict["rope_factor"]
#         rope_scaling = {
#             "type": rope_method,
#             "factor": rope_factor
#         }
#         # NOTE: do not destroy the default rope_scaling of the model
#         rope_kwargs["rope_scaling"] = rope_scaling

#     attn_kwargs = {}
#     attn_impl = model_args_dict["attn_impl"]
#     if attn_impl is not None:
#         if version.parse(transformers.__version__) <= version.parse("4.36"):
#             if attn_impl == "flash_attention_2":
#                 attn_kwargs["use_flash_attention_2"] = True
#         else:
#             attn_kwargs["attn_implementation"] = attn_impl

#     if model_args_dict["max_position_embeddings"] is not None:
#         extra_kwargs["max_position_embeddings"] = model_args_dict["max_position_embeddings"]
#     if model_args_dict["load_in_4_bit"]:
#         extra_kwargs["quantization_config"] = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_quant_type="nf4",
#             bnb_4bit_use_double_quant=True,
#             bnb_4bit_compute_dtype=dtype,
#         )
#         device_map = None

#     model = AutoModelForCausalLM.from_pretrained(
#         model_name_or_path, 
#         cache_dir=cache_dir, 
#         torch_dtype=dtype,
#         device_map=device_map,
#         token=access_token,
#         trust_remote_code=True,

#         # NOTE: do not destroy the default rope_scaling of the model
#         **rope_kwargs,
#         **attn_kwargs,
#     )

#     # load lora
#     if model_args_dict["lora"] is not None:
#         logger.info(f"loading lora from {model_args_dict['lora']}...")
#         model.resize_token_embeddings(32001)

#         trainable_params = os.path.join(os.path.join(model_name_or_path, "trainable_params.bin"))
#         if os.path.isfile(trainable_params):
#             model.load_state_dict(torch.load(trainable_params, map_location=model.device), strict=False)

#         from peft import PeftModel
#         model = PeftModel.from_pretrained(
#             model, 
#             model_args_dict["lora"],
#             torch_dtype=dtype,
#             device_map=device_map,
#         )
#         if model_args_dict["lora_unload"]:
#             model = model.merge_and_unload()


#     if model_args_dict["enable_tp"]:
#         import tensor_parallel as tp
#         logger.info("enabling tensor parallelism...")
        
#         # model = tp.tensor_parallel(model, device_ids=list(range(8)), distributed=False, sharded=False)
#         model = tp.tensor_parallel(model, sharded=True)

#         if model.generation_config.eos_token_id == 128001:
#             model.generation_config.eos_token_id = [128001, 128009]

#     model = model.eval()
#     logger.info(model.config)

#     if evaluation_mode:
#         # NOTE: essential to disable all gradient in-place, so that when calling accelerator.prepare, the forward function will not be wrapped that may consume extra GPU memory
#         model.requires_grad_(False)

#     # override the default generation config
#     generation_config = model_args.get_generation_config()
#     if len(generation_config):
#         unused_config = model.generation_config.update(**generation_config)
#         if len(unused_config):
#             logger.warning(f"The following attributes are not used when overriding the generation configurations: {unused_config}")
#     logger.info(f"Generation config: {generation_config}")

#     return model, tokenizer


# def get_autocompressor_model_and_tokenizer(model_args, device="cpu", evaluation_mode=True, return_tokenizer_only=False, **kwargs):
#     import torch
#     import transformers
#     from dataclasses import asdict
#     from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
#     from transformers.utils import logging
#     from transformers.integrations import is_deepspeed_zero3_enabled
#     from packaging import version

#     from .args import ModelArgs
#     from baselines.auto_compressor import LlamaAutoCompressorModel

#     logger = logging.get_logger(__name__)

#     model_args: ModelArgs

#     model_args_dict = asdict(model_args)
#     model_args_dict.update(**kwargs)
    
#     model_name_or_path = model_args_dict["model_name_or_path"]
#     cache_dir = model_args_dict["model_cache_dir"]
#     access_token = model_args_dict["access_token"]

#     logger.info(f"Loading model and tokenizer from {model_name_or_path}...")

#     tokenizer_kwargs = {}
#     if model_args_dict["no_use_fast"]:
#         tokenizer_kwargs = {"use_fast": False}

#     tokenizer = AutoTokenizer.from_pretrained(
#         model_name_or_path, 
#         cache_dir=cache_dir, 
#         padding_side=model_args_dict["padding_side"], 
#         token=access_token, 
#         trust_remote_code=True,
#         **tokenizer_kwargs
#     )
#     if tokenizer.pad_token_id is None:
#         tokenizer.pad_token_id = tokenizer.eos_token_id
    
#     if return_tokenizer_only:
#         return tokenizer

#     dtype = model_args_dict["dtype"]
#     if dtype == "bf16":
#         dtype = torch.bfloat16
#     elif dtype == "fp16":
#         dtype = torch.float16
#     else:
#         dtype = torch.float32
        
#     device_map = model_args_dict["device_map"]
#     if device_map is None and not is_deepspeed_zero3_enabled():
#         device_map = {"": device}
 
#     model = LlamaAutoCompressorModel.from_pretrained(
#         model_name_or_path, 
#         cache_dir=cache_dir, 
#         torch_dtype=dtype,
#         device_map=device_map,
#         token=access_token,
#     )

#     # assign segment length
#     model.config.segment_lengths = model_args_dict["autocompr_segment_length"]

#     model = model.eval()
#     logger.info(model.config)

#     if evaluation_mode:
#         # NOTE: essential to disable all gradient in-place, so that when calling accelerator.prepare, the forward function will not be wrapped that may consume extra GPU memory
#         model.requires_grad_(False)

#     # override the default generation config
#     generation_config = model_args.get_generation_config()
#     if len(generation_config):
#         unused_config = model.generation_config.update(**generation_config)
#         if len(unused_config):
#             logger.warning(f"The following attributes are not used when overriding the generation configurations: {unused_config}")
#     logger.info(f"Generation config: {generation_config}")

#     return model, tokenizer
