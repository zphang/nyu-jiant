from typing import Dict

import transformers

import jiant.proj.main.components.container_setup as container_setup
import jiant.proj.main.modeling.primary as primary
from jiant.tasks import Task
from jiant.shared.model_resolution import ModelArchitectures
from jiant.proj.main.modeling.model_setup import create_taskmodel, get_taskmodel_and_task_names

import jiantexp.experimental.frozen.form1.models as form1_models


def setup_jiant_model(
    model_type: str,
    pooler_config: str,
    task_dict: Dict[str, Task],
    taskmodels_config: container_setup.TaskmodelsConfig,
):
    model_arch = ModelArchitectures.from_model_type(model_type)
    encoder = form1_models.create_encoder(
        model_type=model_type,
        pooler_config=pooler_config,
    )
    assert model_type.startswith("roberta-")
    tokenizer = transformers.RobertaTokenizer.from_pretrained(model_type)
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
