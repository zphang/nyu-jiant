import torch

import jiant.tasks.evaluate as evaluate
import jiant.proj.main.runner as jiant_runner
from jiant.proj.main.modeling.primary import JiantModel, wrap_jiant_forward
import jiant.utils.transformer_utils as transformer_utils
import jiant.utils.torch_utils as torch_utils
from jiant.utils.display import maybe_tqdm


def get_attn(jiant_model, batch, task):
    with torch.no_grad():
        jiant_model.eval()
        with transformer_utils.output_attention_weights_context(encoder=jiant_model.encoder):
            model_output = wrap_jiant_forward(
                jiant_model=jiant_model, batch=batch, task=task, compute_loss=False,
            )
    return model_output.other[0]


class FrozenAttnRunner(jiant_runner.JiantRunner):
    def __init__(
        self,
        jiant_task_container: jiant_runner.JiantTaskContainer,
        jiant_model: jiant_runner.JiantModel,
        optimizer_scheduler,
        device,
        rparams: jiant_runner.RunnerParameters,
        log_writer,
        frozen_model,
    ):
        super().__init__(
            jiant_task_container=jiant_task_container,
            jiant_model=jiant_model,
            optimizer_scheduler=optimizer_scheduler,
            device=device,
            rparams=rparams,
            log_writer=log_writer
        )
        self.frozen_model = frozen_model

    def run_train_step(self, train_dataloader_dict: dict, train_state: jiant_runner.TrainState):
        self.jiant_model.train()
        task_name, task = self.jiant_task_container.task_sampler.pop()
        task_specific_config = self.jiant_task_container.task_specific_configs[task_name]

        loss_val = 0
        for i in range(task_specific_config.gradient_accumulation_steps):
            batch, batch_metadata = train_dataloader_dict[task_name].pop()
            batch = batch.to(self.device)
            batch.attn = None
            batch.attn = get_attn(jiant_model=self.frozen_model, batch=batch, task=task)
            model_output = wrap_jiant_forward(
                jiant_model=self.jiant_model, batch=batch, task=task, compute_loss=True,
            )
            loss = self.complex_backpropagate(
                loss=model_output.loss,
                gradient_accumulation_steps=task_specific_config.gradient_accumulation_steps,
            )
            loss_val += loss.item()

        self.optimizer_scheduler.step()
        self.optimizer_scheduler.optimizer.zero_grad()

        train_state.step(task_name=task_name)
        self.log_writer.write_entry(
            "loss_train",
            {
                "task": task_name,
                "task_step": train_state.task_steps[task_name],
                "global_step": train_state.global_steps,
                "loss_val": loss_val / task_specific_config.gradient_accumulation_steps,
            },
        )

    def run_val(self, task_name_list, use_subset=None, return_preds=False, verbose=True):
        evaluate_dict = {}
        val_dataloader_dict = self.get_val_dataloader_dict(
            task_name_list=task_name_list, use_subset=use_subset
        )
        val_labels_dict = self.get_val_labels_dict(
            task_name_list=task_name_list, use_subset=use_subset
        )
        for task_name in task_name_list:
            task = self.jiant_task_container.task_dict[task_name]
            evaluate_dict[task_name] = run_val(
                val_dataloader=val_dataloader_dict[task_name],
                val_labels=val_labels_dict[task_name],
                jiant_model=self.jiant_model,
                task=task,
                device=self.device,
                local_rank=self.rparams.local_rank,
                return_preds=return_preds,
                verbose=verbose,
                frozen_model=self.frozen_model,
            )
        return evaluate_dict


def run_val(
    val_dataloader,
    val_labels,
    jiant_model: JiantModel,
    frozen_model: JiantModel,
    task,
    device,
    local_rank,
    return_preds=False,
    verbose=True,
):
    # Reminder:
    #   val_dataloader contains mostly PyTorch-relevant info
    #   val_labels might contain more details information needed for full evaluation
    if not local_rank == -1:
        return
    jiant_model.eval()
    total_eval_loss = 0
    nb_eval_steps, nb_eval_examples = 0, 0
    evaluation_scheme = evaluate.get_evaluation_scheme_for_task(task=task)
    eval_accumulator = evaluation_scheme.get_accumulator()

    for step, (batch, batch_metadata) in enumerate(
        maybe_tqdm(val_dataloader, desc=f"Eval ({task.name}, Val)", verbose=verbose)
    ):
        batch = batch.to(device)

        with torch.no_grad():
            batch.attn = None
            batch.attn = get_attn(jiant_model=frozen_model, batch=batch, task=task)
            model_output = wrap_jiant_forward(
                jiant_model=jiant_model, batch=batch, task=task, compute_loss=True,
            )
        batch_logits = model_output.logits.detach().cpu().numpy()
        batch_loss = model_output.loss.mean().item()
        total_eval_loss += batch_loss
        eval_accumulator.update(
            batch_logits=batch_logits,
            batch_loss=batch_loss,
            batch=batch,
            batch_metadata=batch_metadata,
        )

        nb_eval_examples += len(batch)
        nb_eval_steps += 1
    eval_loss = total_eval_loss / nb_eval_steps
    tokenizer = (
        jiant_model.tokenizer
        if not torch_utils.is_data_parallel(jiant_model)
        else jiant_model.module.tokenizer
    )
    output = {
        "accumulator": eval_accumulator,
        "loss": eval_loss,
        "metrics": evaluation_scheme.compute_metrics_from_accumulator(
            task=task, accumulator=eval_accumulator, labels=val_labels, tokenizer=tokenizer,
        ),
    }
    if return_preds:
        output["preds"] = evaluation_scheme.get_preds_from_accumulator(
            task=task, accumulator=eval_accumulator,
        )
    return output
