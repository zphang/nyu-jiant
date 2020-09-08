import os
import numpy as np
import torch
import tqdm

from dataclasses import dataclass
from typing import Sequence

import jiant.utils.zconf as zconf

import jiant.shared.initialization as initialization
import jiant.proj.main.modeling.model_setup as model_setup
import jiant.proj.main.components.container_setup as container_setup
from jiant.proj.main.modeling.primary import JiantModel, wrap_jiant_forward
import jiant.tasks as tasks
import jiant.shared.runner as shared_runner
import jiant.shared.caching as caching
import jiantexp.experimental.repr_study.cka as cka
from torch.utils.data.dataloader import DataLoader
from jiant.shared.model_setup import ModelArchitectures


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    # === Required parameters === #
    task_config_path = zconf.attr(type=str, required=True)
    task_cache_path = zconf.attr(type=str, default=None)
    indices_path = zconf.attr(type=str, default=None)
    output_dir = zconf.attr(type=str, required=True)

    # === Model parameters === #
    model_type = zconf.attr(type=str, required=True)
    model_config_path = zconf.attr(default=None, type=str)
    model_tokenizer_path = zconf.attr(default=None, type=str)

    model_a_path = zconf.attr(type=str, required=True)
    model_b_path = zconf.attr(type=str, required=True)

    # === Running Setup === #
    batch_size = zconf.attr(default=8, type=int)
    skip_b = zconf.attr(action="store_true")
    skip_cka = zconf.attr(action="store_true")
    cka_kernel = zconf.attr(default="linear", type=str)
    save_acts = zconf.attr(action="store_true")

    # Specialized config
    no_cuda = zconf.attr(action='store_true')
    fp16 = zconf.attr(action='store_true')
    local_rank = zconf.attr(default=-1, type=int)
    server_ip = zconf.attr(default='', type=str)
    server_port = zconf.attr(default='', type=str)
    seed = zconf.attr(type=int, default=-1)
    force_overwrite = zconf.attr(action="store_true")


def main(args):
    quick_init_out = initialization.quick_init(args=args, verbose=True)
    with quick_init_out.log_writer.log_context():
        task = tasks.create_task_from_config_path(
            config_path=args.task_config_path,
            verbose=True,
        )
        jiant_model = model_setup.setup_jiant_model(
            model_type=args.model_type,
            model_config_path=args.model_config_path,
            tokenizer_path=args.model_tokenizer_path,
            task_dict={task.name: task},
            taskmodels_config=container_setup.TaskmodelsConfig(
                {task.name: task.name}
            ),
        )
        model_arch = ModelArchitectures.from_model_type(model_type=args.model_type)
        if model_arch == ModelArchitectures.ROBERTA:
            jiant_model.encoder.encoder.output_hidden_states = True
        else:
            raise RuntimeError()
        data_obj = DataObj.from_path(
            task=task,
            task_cache_path=args.task_cache_path,
            indices_path=args.indices_path,
            batch_size=args.batch_size,
        )

        # === Compute === #
        act_a = compute_activations_from_path(
            data_obj=data_obj,
            task=task,
            jiant_model=jiant_model,
            model_path=args.model_a_path,
            device=quick_init_out.device,
        )
        if not args.skip_b:
            act_b = compute_activations_from_path(
                data_obj=data_obj,
                task=task,
                jiant_model=jiant_model,
                model_path=args.model_b_path,
                device=quick_init_out.device,
            )
        if not args.skip_cka:
            assert not args.skip_b
            cka_outputs = compute_cka(
                act_a=act_a,
                act_b=act_b,
                device=quick_init_out.device,
                cka_kernel=args.cka_kernel,
            )
            torch.save(cka_outputs, os.path.join(args.output_dir, "cka.p"))
        if args.save_acts:
            torch.save(act_a, os.path.join(args.output_dir, "act_a.p"))
            if not args.skip_b:
                torch.save(act_b, os.path.join(args.output_dir, "act_b.p"))


@dataclass
class DataObj:
    dataloader: DataLoader
    grouped_input_indices: Sequence
    grouped_position_indices: Sequence

    @classmethod
    def from_path(cls, task, task_cache_path, indices_path, batch_size):
        loaded = torch.load(indices_path)
        grouped_input_indices = loaded["grouped_input_indices"]
        grouped_position_indices = loaded["grouped_position_indices"]
        task_cache = caching.ChunkedFilesDataCache(task_cache_path)
        # Account for multiple-choice
        example_indices = np.array(grouped_input_indices) // get_num_inputs(task)
        dataloader = shared_runner.get_eval_dataloader_from_cache(
            eval_cache=task_cache,
            task=task,
            eval_batch_size=batch_size,
            explicit_subset=example_indices,
        )
        return cls(
            dataloader=dataloader,
            grouped_input_indices=grouped_input_indices,
            grouped_position_indices=grouped_position_indices,
        )


def get_num_inputs(task: tasks.Task):

    if task.TASK_TYPE == tasks.TaskTypes.MULTIPLE_CHOICE:
        # noinspection PyUnresolvedReferences
        return task.NUM_CHOICES
    else:
        return 1


def compute_activations_from_path(data_obj: DataObj,
                                  task: tasks.Task,
                                  jiant_model: JiantModel,
                                  model_path: str,
                                  device):
    load_model(
        jiant_model=jiant_model,
        model_path=model_path,
        device=device,
    )
    return compute_activations_from_model(
        data_obj=data_obj,
        task=task,
        jiant_model=jiant_model,
        device=device,
    )


def compute_cka(act_a, act_b, device, cka_kernel):
    assert act_a.shape[1] == act_b.shape[1]
    num_layers = act_a.shape[1]
    collated = np.empty([num_layers, num_layers])
    for i in tqdm.tqdm(range(num_layers), desc="CKA row"):
        # noinspection PyArgumentList
        act_a_tensor = torch.Tensor(act_a[:, i].copy()).float().to(device)
        for j in tqdm.tqdm(range(num_layers)):
            # noinspection PyArgumentList
            act_b_tensor = torch.Tensor(act_b[:, j].copy()).float().to(device)
            collated[i, j] = cka.compute_cka(
                x=act_a_tensor, y=act_b_tensor,
                kernel=cka_kernel,
            ).item()
    return collated


def get_hidden_act(jiant_model: JiantModel, batch, task):
    model_output = wrap_jiant_forward(
        jiant_model=jiant_model,
        batch=batch,
        task=task,
        compute_loss=False,
    )
    raw_hidden = model_output.other[0]
    hidden_act = torch.stack(raw_hidden, dim=2)
    return hidden_act


def compute_activations_from_model(data_obj: DataObj,
                                   task: tasks.Task,
                                   jiant_model: JiantModel,
                                   device):
    num_inputs_per_example = get_num_inputs(task)
    collected_acts = []
    with torch.no_grad():
        jiant_model.eval()
        example_i = 0
        for batch, batch_metadata in tqdm.tqdm(data_obj.dataloader, desc="Computing Activation"):
            batch = batch.to(device)
            hidden_act = get_hidden_act(
                jiant_model=jiant_model,
                batch=batch,
                task=task,
            )

            if task.TASK_TYPE == tasks.TaskTypes.MULTIPLE_CHOICE:
                input_indices = np.repeat(
                    np.arange(len(batch)),
                    [len(data_obj.grouped_position_indices[example_i + i])
                     for i in range(len(batch))],
                )
                # noinspection PyArgumentList
                batch_example_indices = torch.LongTensor(input_indices // num_inputs_per_example).to(device)
                # noinspection PyArgumentList
                batch_choice_indices = torch.LongTensor(input_indices % num_inputs_per_example).to(device)
                # noinspection PyArgumentList
                batch_position_indices = torch.LongTensor(np.concatenate([
                    data_obj.grouped_position_indices[example_i + i] for i in range(len(batch))]
                )).to(device)
                collected_acts.append(hidden_act[
                    batch_example_indices, batch_choice_indices, :, batch_position_indices
                ].cpu().numpy())
                example_i += len(batch)
            else:
                # noinspection PyArgumentList
                batch_example_indices = torch.LongTensor(np.repeat(
                    np.arange(len(batch)),
                    [len(data_obj.grouped_position_indices[example_i + i])
                     for i in range(len(batch))],
                )).to(device)
                # noinspection PyArgumentList
                batch_position_indices = torch.LongTensor(np.concatenate([
                    data_obj.grouped_position_indices[example_i + i] for i in range(len(batch))]
                )).to(device)
                collected_acts.append(hidden_act[batch_example_indices, batch_position_indices].cpu().numpy())
                example_i += len(batch)
    return np.concatenate(collected_acts)


def load_model(jiant_model: JiantModel, model_path: str, device) -> JiantModel:
    state_dict = torch.load(model_path, map_location="cpu")
    first_key = list(state_dict)[0]
    if first_key.startswith("encoder."):
        model_setup.delegate_load(
            jiant_model=jiant_model,
            load_mode="partial",
            weights_dict=state_dict,
        )
    else:
        model_setup.delegate_load(
            jiant_model=jiant_model,
            load_mode="from_transformers",
            weights_dict=state_dict,
        )
    jiant_model.to(device)
    return jiant_model


if __name__ == "__main__":
    main(args=RunConfiguration.run_cli_json_prepend())
