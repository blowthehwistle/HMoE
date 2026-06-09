import time
from torch import Tensor
from torchtitan.tools.logging import logger
from torchtitan.distributed.utils import dist_mean
from torchtitan.components.metrics import MetricsProcessor
from typing import Dict


_ENABLE_TTP_METRICS_PROCESSOR = False
_EXTRA_METRICS = []


class MetricsAvgAccumulatorHook:

    def __init__(self, need_accum: bool = True):
        self._accum_steps = 0
        self._accum_metrics = {}
        self._need_accum = need_accum

    def __call__(self, trainer, metrics: Dict[str, Tensor]):
        for k, v in metrics.items():
            if k not in self._accum_metrics:
                self._accum_metrics[k] = .0
            if trainer.parallel_dims.dp_enabled:
                v = dist_mean(v, trainer.parallel_dims.world_mesh['dp'])
            self._accum_metrics[k] += v
        self._accum_steps += 1
        if self._accum_steps % trainer.gradient_accumulation_steps == 0:
            if self._need_accum:
                accum_metrics = {k: v / trainer.gradient_accumulation_steps for k, v in self._accum_metrics.items()}
            else:
                accum_metrics = {k: v for k, v in self._accum_metrics.items()}
            self._accum_steps = 0
            self._accum_metrics = {}
            return accum_metrics
        else:
            return {}


def push_extra_metrics(hook=None, **metrics):
    global _EXTRA_METRICS, _ENABLE_TTP_METRICS_PROCESSOR
    if _ENABLE_TTP_METRICS_PROCESSOR:
        _EXTRA_METRICS.append((hook, metrics))


class TTPMetricsProcessor(MetricsProcessor):

    def __init__(self, job_config, parallel_dims, tag=None):
        super().__init__(job_config, parallel_dims, tag)
        global _ENABLE_TTP_METRICS_PROCESSOR
        _ENABLE_TTP_METRICS_PROCESSOR = True
        self._trainer = None  # set later

    def set_trainer(self, trainer):
        self._trainer = trainer

    def log(self, step, global_avg_loss, global_max_loss, grad_norm, extra_metrics=None):
        global _EXTRA_METRICS
        if extra_metrics is None:
            extra_metrics = {}
        while len(_EXTRA_METRICS) > 0:
            hook, em = _EXTRA_METRICS.pop(0)
            if hook is not None:
                em = hook(self._trainer, em)
            extra_metrics.update(em)

        time_delta = time.perf_counter() - self.time_last_log

        # tokens per second per device, abbreviated as tps
        tps = self.ntokens_since_last_log / (
            time_delta * self.parallel_dims.non_data_parallel_size
        )
        if self.num_flops_per_token > 0:
            tflops = self.num_flops_per_token * tps / 1e12
            mfu = 100 * self.num_flops_per_token * tps / self.gpu_peak_flops
        else:
            tflops = None
            mfu = None

        time_end_to_end = time_delta / self.job_config.metrics.log_freq
        time_data_loading = sum(self.data_loading_times) / len(self.data_loading_times)
        time_data_loading_pct = 100 * sum(self.data_loading_times) / time_delta

        device_mem_stats = self.device_memory_monitor.get_peak_stats()

        metrics = {
            "loss_metrics/global_avg_loss": global_avg_loss,
            "loss_metrics/global_max_loss": global_max_loss,
            "grad_norm": grad_norm,
            "throughput(tps)": tps,
            "time_metrics/end_to_end(s)": time_end_to_end,
            "time_metrics/data_loading(s)": time_data_loading,
            "time_metrics/data_loading(%)": time_data_loading_pct,
            "memory/max_active(GiB)": device_mem_stats.max_active_gib,
            "memory/max_active(%)": device_mem_stats.max_active_pct,
            "memory/max_reserved(GiB)": device_mem_stats.max_reserved_gib,
            "memory/max_reserved(%)": device_mem_stats.max_reserved_pct,
            "memory/num_alloc_retries": device_mem_stats.num_alloc_retries,
            "memory/num_ooms": device_mem_stats.num_ooms,
        }
        if tflops is not None and mfu is not None:
            metrics["estimated_tflops"] = tflops
            metrics["estimated_mfu(%)"] = mfu

        if extra_metrics:
            metrics.update(extra_metrics)

        self.logger.log(metrics, step)

        color = self.color
        logger.info(
            f"{color.red}step: {step:2}  "
            f"{color.green}loss: {global_avg_loss:7.4f}  "
            f"{color.orange}grad_norm: {grad_norm:7.4f}  "
            f"{color.turquoise}memory: {device_mem_stats.max_reserved_gib:5.2f}GiB"
            f"({device_mem_stats.max_reserved_pct:.2f}%)  "
            f"{color.blue}tps: {round(tps):,}  "
            f"{color.cyan}est_tflops: {tflops if tflops is not None else -1:,.2f}  "
            f"{color.magenta}est_mfu: {mfu if mfu is not None else -1:.2f}%{color.reset}"
        )

        self.ntokens_since_last_log = 0
        self.data_loading_times.clear()
        self.time_last_log = time.perf_counter()
        self.device_memory_monitor.reset_peak_stats()


def build_ttp_metrics_processor(job_config, parallel_dims, model_args=None, tag=None) -> TTPMetricsProcessor:
    return TTPMetricsProcessor(job_config, parallel_dims, tag)
