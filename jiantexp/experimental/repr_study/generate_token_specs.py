import numpy as np
import os

import torch

import jiantexp.experimental.repr_study.weights_utils as weights_utils
import jiant.shared.runner as shared_runner
import jiant.shared.caching as caching
import jiant.tasks as tasks
import jiant.utils.zconf as zconf


def get_lengths_path(task_cache_path: str):
    return os.path.join(task_cache_path, "lengths.p")


def compute_lengths(task, task_cache: caching.ChunkedFilesDataCache):
    # shape: (n, ) for single-input
    # shape: (n, k) for multiple-choice
    dataloader = shared_runner.get_eval_dataloader_from_cache(
        eval_cache=task_cache,
        task=task,
        eval_batch_size=10000,
    )
    lengths_ls = []
    for batch, batch_metadata in dataloader:
        lengths_ls.append(batch.input_mask.sum(-1).cpu().numpy())
    lengths = np.concatenate(lengths_ls)
    return lengths


def write_lengths_from_paths(task_config_path: str, task_cache_path: str):
    task = tasks.create_task_from_config_path(task_config_path)
    task_cache = caching.ChunkedFilesDataCache(task_cache_path)
    lengths = compute_lengths(task=task, task_cache=task_cache)
    torch.save(lengths, get_lengths_path(task_cache_path=task_cache_path))


def write_tok_indices(ip_lookup: weights_utils.InputPositionLookup,
                      num_samples, output_path, rng=None):
    """
    TOK and CLS have slightly differently logic. We want e.g. 10k examples in either case

    TOK will sample 10k (example_i, token_i) pairs, and then group them by example_i.
    So len(grouped_input_indices) = len(grouped_position_indices) < 10K

    CLS samples 10K (example_i). But we don't do grouping, so we can still average over 10K
    So len(grouped_input_indices) = len(grouped_position_indices) = 10K
    """
    input_indices, position_indices = ip_lookup.sample(num_samples, rng=rng)
    grouped_input_indices, grouped_position_indices = \
        weights_utils.group_input_position_indices(
            input_indices, position_indices
        )
    torch.save(
        {
            "grouped_input_indices": grouped_input_indices,
            "grouped_position_indices": grouped_position_indices,
        },
        output_path
    )


def write_cls_indices(task, ip_lookup: weights_utils.InputPositionLookup,
                      num_samples, output_path):
    """
    write_tok_indices
    """
    num_examples = len(ip_lookup.lengths)
    if task.TASK_TYPE == tasks.TaskTypes.MULTIPLE_CHOICE:
        num_inputs_per_example = task.NUM_CHOICES
    else:
        num_inputs_per_example = 1
    num_inputs = num_examples * num_inputs_per_example
    torch.save(
        {
            "grouped_input_indices": np.random.choice(
                np.arange(num_inputs),
                size=num_samples,
                replace=num_samples > num_inputs,
            ),
            # assume position-0 for CLS
            "grouped_position_indices": np.zeros(num_samples).reshape(-1, 1),
        },
        output_path,
    )


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    task_config_path = zconf.attr(type=str)
    task_cache_path = zconf.attr(type=str)
    num_samples = zconf.attr(type=int, default=10000)
    rng = zconf.attr(type=int, default=None)
    output_base_path = zconf.attr(type=str)


def main(args: RunConfiguration):
    os.makedirs(args.output_base_path, exist_ok=True)
    task = tasks.create_task_from_config_path(config_path=args.task_config_path)
    lengths_path = get_lengths_path(task_cache_path=args.task_cache_path)
    if not os.path.exists(lengths_path):
        write_lengths_from_paths(
            task_config_path=args.task_config_path,
            task_cache_path=args.task_cache_path,
        )
    ip_lookup = weights_utils.InputPositionLookup.from_path(path=lengths_path)
    write_tok_indices(
        ip_lookup=ip_lookup,
        num_samples=args.num_samples,
        output_path=os.path.join(args.output_base_path, "tok.p"),
        rng=args.rng,
    )
    write_cls_indices(
        task=task,
        ip_lookup=ip_lookup,
        num_samples=args.num_samples,
        output_path=os.path.join(args.output_base_path, "cls.p")
    )


if __name__ == "__main__":
    main(RunConfiguration.default_run_cli())
