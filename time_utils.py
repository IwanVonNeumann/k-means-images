import time


def measure_time(f):
    def wrapper(*args, **kwargs):
        start = time.time()
        r = f(*args, **kwargs)
        end = time.time()
        print("time elapsed: {:.4f}".format(end - start))
        return r

    return wrapper
