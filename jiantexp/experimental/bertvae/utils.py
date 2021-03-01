import contextlib
import torch


@contextlib.contextmanager
def temporarily_set_rng(seed=None):
    # Convenience: set None to disable
    if seed is None:
        yield
    else:
        rng_state = torch.random.get_rng_state()
        try:
            torch.random.manual_seed(seed)
            yield
        finally:
            torch.random.set_rng_state(rng_state)


def move_to_device(batch, device):
    return {
        k: v.to(device)
        for k, v in batch.items()
    }


def cycle_dataloader(dataloader, num_steps):
    steps = 0
    while True:
        for batch in dataloader:
            yield batch
            steps += 1
            if steps == num_steps:
                return
