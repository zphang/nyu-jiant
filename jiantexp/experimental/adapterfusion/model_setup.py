from typing import Dict
import jiant.proj.main.components.container_setup as container_setup
import jiant.shared.model_setup as model_setup
from jiant.shared.model_resolution import ModelArchitectures, register_from_encoder
from jiant.proj.main.modeling.model_setup import create_taskmodel, get_taskmodel_and_task_names
import jiant.proj.main.modeling.primary as primary
from jiant.tasks import Task
import jiant.utils.python.datastructures as datastructures

import jiantexp.experimental.adapterfusion.ext.modeling_bert as modeling_bert
import jiantexp.experimental.adapterfusion.ext.modeling_roberta as modeling_roberta
import jiantexp.experimental.adapterfusion.ext.modeling_xlm_roberta as modeling_xlm_roberta
import jiantexp.experimental.adapterfusion.ext.adapter_config as adapter_config_lib
import jiantexp.experimental.adapterfusion.ext.adapter_utils as adapter_utils


def get_adapter_compatible_encoder(model_type):
    model_arch = ModelArchitectures.from_model_type(model_type)
    if model_arch == ModelArchitectures.BERT:
        return modeling_bert.BertModel.from_pretrained(model_type)
    elif model_arch == ModelArchitectures.ROBERTA:
        return modeling_roberta.RobertaModel.from_pretrained(model_type)
    elif model_arch == ModelArchitectures.XLM_ROBERTA:
        return modeling_xlm_roberta.XLMRobertaModel.from_pretrained(model_type)
    else:
        raise KeyError(model_arch)


register_from_encoder(modeling_bert.BertModel, ModelArchitectures.BERT)
register_from_encoder(modeling_roberta.RobertaModel, ModelArchitectures.ROBERTA)
register_from_encoder(modeling_xlm_roberta.XLMRobertaModel, ModelArchitectures.XLM_ROBERTA)


def setup_adapterfusion_jiant_model(
    model_type: str,
    tokenizer_path: str,
    task_dict: Dict[str, Task],
    raw_adapter_config: str,
    adapter_tuning_mode: str,
    taskmodels_config: container_setup.TaskmodelsConfig,
):
    model_arch = ModelArchitectures.from_model_type(model_type)
    tokenizer = model_setup.get_tokenizer(model_type=model_type, tokenizer_path=tokenizer_path)
    encoder = get_adapter_compatible_encoder(model_type=model_type)

    # Setup adapters
    taskmodel_name_list = list(taskmodels_config.task_to_taskmodel_map.values())
    adapter_config = adapter_config_lib.AdapterConfig.load(raw_adapter_config)
    for taskmodel_name in taskmodel_name_list:
        encoder.add_adapter(taskmodel_name, adapter_utils.AdapterType.text_task, config=adapter_config)

    if adapter_tuning_mode == "single":
        encoder.set_active_adapters(taskmodel_name_list)
        encoder.train_adapter(taskmodel_name_list)
    elif adapter_tuning_mode == "fusion":
        encoder.add_fusion(taskmodel_name_list, "dynamic")
        encoder.train_fusion(taskmodel_name_list)

    taskmodels_dict = {
        taskmodel_name: create_taskmodel(
            task=task_dict[task_name_list[0]],  # Take the first task
            model_arch=model_arch,
            encoder=encoder,
            taskmodel_kwargs=taskmodels_config.get_taskmodel_kwargs(taskmodel_name),
        )
        for taskmodel_name, task_name_list in get_taskmodel_and_task_names(
            taskmodels_config.task_to_taskmodel_map
        ).items()
    }
    return primary.JiantModel(
        task_dict=task_dict,
        encoder=encoder,
        taskmodels_dict=taskmodels_dict,
        task_to_taskmodel_map=taskmodels_config.task_to_taskmodel_map,
        tokenizer=tokenizer,
    )
