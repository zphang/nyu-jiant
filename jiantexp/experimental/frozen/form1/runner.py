import os

import torch
import torch.nn as nn

import jiant.utils.python.io as py_io
import jiant.utils.torch_utils as torch_utils


def save_model_with_metadata(model: nn.Module, metadata: dict, output_dir: str, file_name="model"):
    save_model(
        model=model,
        output_dir=output_dir,
        file_name=file_name,
    )
    py_io.write_json(metadata, os.path.join(output_dir, f"{file_name}.metadata.json"))


def save_model(model: nn.Module, output_dir: str, file_name="model"):
    raw_state_dict = torch_utils.get_model_for_saving(model).state_dict()
    state_dict = {
        n: p
        for n, p in raw_state_dict.items()
        if "frozen_encoder." not in n  # <-- this is a hack. Need better way to exclude encoder
    }
    print("save", len(state_dict))
    torch.save(
        state_dict,
        os.path.join(output_dir, f"{file_name}.p"),
    )
