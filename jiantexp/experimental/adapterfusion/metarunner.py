from jiantexp.experimental.adapterfusion.runner import save_model_with_metadata
from jiant.proj.main.metarunner import JiantMetarunner


class AdapterFusionMetarunner(JiantMetarunner):
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
