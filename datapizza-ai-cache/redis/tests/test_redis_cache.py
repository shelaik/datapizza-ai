from datapizza.cache.redis import RedisCache


def test_redis_cache():
    cache = RedisCache(host="localhost", port=6379, db=0)
    assert cache is not None
