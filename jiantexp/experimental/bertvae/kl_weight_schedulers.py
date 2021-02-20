from abc import ABC, abstractmethod


class KLWeightScheduler(ABC):
    @abstractmethod
    def get_loss(self, step, kl_loss_tensor) -> float:
        raise NotImplementedError()


class ConstantScheduler(KLWeightScheduler):
    def __init__(self, kl_weight):
        self.kl_weight = kl_weight

    def get_loss(self, step, kl_loss_tensor) -> float:
        if self.kl_weight == 1:
            return kl_loss_tensor.mean()
        else:
            return self.kl_weight * kl_loss_tensor.mean()


class BudgetScheduler(KLWeightScheduler):
    def __init__(self, num_steps):
        self.num_steps = num_steps

    def get_loss(self, step, kl_loss_tensor) -> float:
        # from https://github.com/zomux/lanmt/blob/8099f03bb8df54cf22dabc811f3b3b5c63603101/lib_lanmt_model.py#L28
        if step < self.num_steps / 2:
            budget = 1
        else:
            budget = (self.num_steps - step) / (self.num_steps / 2)
        max_mask = ((kl_loss_tensor - budget) > 0.).float()
        kl = kl_loss_tensor * max_mask + (1. - max_mask) * budget
        if step % 100 == 0:
            import pdb; pdb.set_trace()
        return kl.mean()


def create_kl_weight_scheduler(args) -> KLWeightScheduler:
    if args.kl_weight_scheduler_name == "ConstantScheduler":
        return ConstantScheduler(kl_weight=float(args.kl_weight_scheduler_config))
    elif args.kl_weight_scheduler_name == "BudgetScheduler":
        return BudgetScheduler(num_steps=args.num_steps)
