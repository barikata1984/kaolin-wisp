from dataclasses import dataclass, field
from typing import Union, Type, Dict, Any


@dataclass
class MetricsBoard:
    _data: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    """ A module for aggregating losses and metrics during optimization.
    Usage:
        - define_metric() declares the metrics this board expects, call this once when the optimization starts.
        - log_metric() is used to report a metric every iteration or epoch.
        - average_metric() returns the average value logged for the metric so far.
        - finalize_epoch() should be called once at the end of an epoch, to log the aggregated metric in the global
            shared state. This allows all components to access this value through the shared state.
        - clear() should be called to clear the aggregated logs and start a new accumulation (i.e. new epoch starts).
    """

    def __post_init__(self):
        self.num_samples = 0    # Number of samples reported so far, since __init__ or clear have been called.

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def keys(self):
        return self._data.keys()

    def clear(self):
        """ Clears the MetricsBoard, essentially zeroing all accumulated values for defined metrics.
        Defined metrics will remain defined after calling clear().
        clear() should be called to clear the aggregated logs and start a new accumulation (i.e. new epoch starts)."""
        reserved_keys = 'num_samples'
        for k in self.keys():
            if k in reserved_keys:
                continue
            if isinstance(getattr(self, k), list):
                getattr(self, k).clear()
            else:
                setattr(self, k, getattr(self, k) * 0)

        self.num_samples = 0

    def define_metric(self,
                      name,
                      aggregation_type: Union[Type[list], Type[int], Type[float]] = list):
        """ define_metric() declares the metrics this board expects, call this once when the optimization starts.
        Args:
            name (str): The name of the metric, a unique identifier.
            aggregation_type (Union[Type[list], Type[int], Type[float]]):
                type of accumulator to use for aggregating the metric.
        """
        if not hasattr(self, name):
            setattr(self, name, aggregation_type())

    def log_metric(self, key, value):
        """ log_metric is used to report a metric every iteration or epoch.
        Args:
            key (str): A unique identifier for the metric, assumed to be defined with define_metric.
            value: The accumulated value for the metric, a numeric value.
        """
        if not hasattr(self, key):
            self.define_metric(key)
        getattr(self, key).append(value)

    def average_metric(self, metric):
        """ Returns the average value logged for the metric so far, i.e. for output logging purposes.
        Args:
            metric (str): A unique identifier for the metric, assumed to be defined with define_metric.
        """
        if not hasattr(self, metric):
            raise ValueError(f'metric {metric} is not defined in MetricsBoard.')
        metric_value = getattr(self, metric)
        if isinstance(metric_value, list):
            metric_value = sum(metric_value)
        if self.num_samples == 0:
            return metric_value
        else:
            return 1.0 * metric_value / self.num_samples

    def finalize_epoch(self, wisp_state):
        """ finalize_epoch() should be called once at the end of an epoch, to log the aggregated metric in the global
        shared state. This allows all components to access this value through the shared state.
        Args:
            wisp_state (WispState): Shared state which allows other wisp components the access the final metrics
            (i.e. gui).
        """
        reserved_keys = 'num_samples'
        for k in self.keys():
            if k in reserved_keys:
                continue
            running_k = self.average_metric(k)
            if 'loss' in k:
                wisp_state.optimization.losses[k].append(running_k)
            else:
                wisp_state.optimization.metrics[k].append(running_k)

    @property
    def active_metrics(self):
        """ Returns all currently defined metrics. """
        return [k for k in self.keys() if k not in ['num_samples']] # num_samples is a field, not a metric
