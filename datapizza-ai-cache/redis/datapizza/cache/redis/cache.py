import logging
import pickle

from datapizza.core.cache import Cache

import redis

log = logging.getLogger(__name__)


class RedisCache(Cache):
    """
    A Redis-based cache implementation.
    """

    def __init__(
        self, host="localhost", port=6379, db=0, expiration_time=3600
    ):  # 1 hour default
        self.redis = redis.Redis(host=host, port=port, db=db)
        self.expiration_time = expiration_time

    def get(self, key: str) -> str | None:
        """Retrieve and deserialize object"""
        pickled_obj = self.redis.get(key)
        if pickled_obj is None:
            return None
        return pickle.loads(pickled_obj)  # type: ignore

    def set(self, key: str, obj):
        """Serialize and store object"""
        pickled_obj = pickle.dumps(obj)
        self.redis.set(key, pickled_obj, ex=self.expiration_time)
