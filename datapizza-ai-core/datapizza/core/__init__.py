import logging

from datapizza.core.utils import _basic_config

__all__ = ["clients", "type", "utils", "memory", "tools"]

# Setup base logging

# Create and configure the main package logger
log = logging.getLogger(__name__)

_basic_config(log)

log.setLevel(logging.DEBUG)
