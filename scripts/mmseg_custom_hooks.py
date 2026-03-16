"""Кастомные hooks: вывод метрик валидации, ранняя остановка по mDice/плато."""
from __future__ import annotations

from mmengine.hooks import Hook

from mmseg.registry import HOOKS


def _display_in_notebook(metrics_dict, step_name):
    try:
        from IPython.display import display, Markdown
        numeric = {k: float(v) for k, v in metrics_dict.items() if _is_numeric(v)}
        if not numeric:
            return
        rows = [f"| {k} | {v:.4f} |" for k, v in numeric.items()]
        header = "| Метрика | Значение |"
        sep = "| --- | --- |"
        table = "\n".join([header, sep] + rows)
        display(Markdown(f"**{step_name}**\n\n{table}"))
    except Exception:
        pass


@HOOKS.register_module()
class ValidationMetricsLogHook(Hook):
    priority = "ABOVE_NORMAL"

    def __init__(self, show_in_notebook: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.show_in_notebook = bool(show_in_notebook)

    def after_val_epoch(self, runner, metrics=None):
        if not metrics:
            return
        numeric = {k: v for k, v in metrics.items() if _is_numeric(v)}
        if not numeric:
            return
        parts = [f"{k}: {float(v):.4f}" for k, v in numeric.items()]
        runner.logger.info("Validation metrics: " + "  ".join(parts))
        if self.show_in_notebook:
            step = getattr(runner, "iter", None) or getattr(runner, "epoch", "?")
            _display_in_notebook(numeric, f"Validation @ iter/epoch {step}")


def _is_numeric(value) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


@HOOKS.register_module()
class EarlyExitHook(Hook):
    priority = "LOW"

    def __init__(
        self,
        metric_name: str = "mDice",
        target: float = 0.75,
        patience: int = 4,
        min_delta: float = 1e-4,
    ) -> None:
        self.metric_name = metric_name
        self.target = float(target)
        self.patience = int(patience)
        self.min_delta = float(min_delta)

        self.best = float("-inf")
        self.bad_epochs = 0

    def _read_metric(self, runner):
        mh = runner.message_hub
        candidate_keys = [
            f"val/{self.metric_name}",
            self.metric_name,
            f"{self.metric_name}",
        ]
        for key in candidate_keys:
            try:
                scalar = mh.get_scalar(key)
                if scalar is not None:
                    return float(scalar.current())
            except Exception:
                pass
        try:
            infos = mh.runtime_info
            for key in [f"val/{self.metric_name}", self.metric_name]:
                if key in infos:
                    return float(infos[key])
        except Exception:
            pass
        return None

    def after_val_epoch(self, runner, metrics=None):
        current = None
        if isinstance(metrics, dict):
            if self.metric_name in metrics:
                current = float(metrics[self.metric_name])
            elif f"val/{self.metric_name}" in metrics:
                current = float(metrics[f"val/{self.metric_name}"])

        if current is None:
            current = self._read_metric(runner)

        if current is None:
            runner.logger.warning(
                "EarlyExitHook: метрика %s не найдена, пропускаю проверку.",
                self.metric_name,
            )
            return
        if current >= self.target:
            runner.logger.info(
                "EarlyExitHook: %s=%.6f >= %.6f. Останавливаю обучение.",
                self.metric_name,
                current,
                self.target,
            )
            runner.train_loop.stop_training = True
            return
        if current > self.best + self.min_delta:
            self.best = current
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        if self.bad_epochs >= self.patience:
            runner.logger.info(
                "EarlyExitHook: плато по %s (best=%.6f, current=%.6f, patience=%d). "
                "Останавливаю обучение.",
                self.metric_name,
                self.best,
                current,
                self.patience,
            )
            runner.train_loop.stop_training = True
