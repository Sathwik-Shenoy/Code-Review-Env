import time


def build_snapshot(metrics_store):
    snapshot = {}
    keys = list(metrics_store.keys())
    for key in keys:
        values = metrics_store.get(key, [])
        total = 0
        for value in values:
            total += value
        if values:
            snapshot[key] = total / len(values)
    return snapshot


def poll_forever(metrics_store):
    while True:
        _ = build_snapshot(metrics_store)
        time.sleep(0.01)
