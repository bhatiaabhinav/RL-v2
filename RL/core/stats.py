
stats = {}

kvstats = {}


def record_stat(key, value):
    '''creates key if neccessary'''
    if key not in stats:
        stats[key] = []
    stats[key].append(value)


def record_stats(**kwargs):
    for k, v in kwargs.items():
        record_stat(k, v)


def record_kvstat(key, value):
    '''a convinience method for kvstats[key]=value'''
    kvstats[key] = value


def record_kvstats(**kwargs):
    for k, v in kwargs.items():
        kvstats[k] = v


def get_latest(key, default=0):
    '''get latest value of a key'''
    if key in kvstats:
        return kvstats[key]
    elif key in stats:
        return stats[key][-1]
    else:
        return default


def get_latest_stats():
    s_latest = dict((k, v[-1]) for k, v in stats.items())
    return {**s_latest, **kvstats}
