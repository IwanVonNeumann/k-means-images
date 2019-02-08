import time


def measure_time(f):
    def wrapper(*args, **kwargs):
        start = time.time()
        r = f(*args, **kwargs)
        end = time.time()
        time_elapsed = end - start
        print("time spent on {}: {:.4f}".format(f.__name__, time_elapsed))
        return r

    return wrapper
