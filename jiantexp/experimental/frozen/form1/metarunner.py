from jiantexp.experimental.frozen.form1.runner import save_model_with_metadata
from jiant.proj.main.metarunner import JiantMetarunner
from jiant.utils.zlog import BaseZLogger, PRINT_LOGGER
import jiant.proj.main.runner as jiant_runner


class Form1Metarunner(JiantMetarunner):

    def __init__(
        self,
        runner: jiant_runner.JiantRunner,
        save_every_steps,
        eval_every_steps,
        save_checkpoint_every_steps,
        no_improvements_for_n_evals,
        checkpoint_saver,
        output_dir,
        verbose: bool = True,
        save_best_model: bool = True,
        load_best_model: bool = True,
        save_last_model: bool = True,
        log_writer: BaseZLogger = PRINT_LOGGER,
    ):
        super().__init__(
            runner=runner,
            save_every_steps=save_every_steps,
            eval_every_steps=eval_every_steps,
            save_checkpoint_every_steps=save_checkpoint_every_steps,
            no_improvements_for_n_evals=no_improvements_for_n_evals,
            checkpoint_saver=checkpoint_saver,
            output_dir=output_dir,
            verbose=verbose,
            save_best_model=save_best_model,
            load_best_model=load_best_model,
            save_last_model=save_last_model,
            log_writer=log_writer,
        )

    def save_model(self):
        save_model_with_metadata(
            model=self.model,
            metadata={},
            output_dir=self.output_dir,
            file_name=f"model__{self.train_state.global_steps:09d}",
        )

    def save_best_model_with_metadata(self, val_metrics_dict):
        save_model_with_metadata(
            model=self.model,
            metadata={
                "val_state": self.best_val_state.to_dict(),
                "val_metrics": val_metrics_dict,
            },
            output_dir=self.output_dir,
            file_name="best_model",
        )
