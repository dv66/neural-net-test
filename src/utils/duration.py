from datetime import datetime
from utils.colored_log import Logger

def time_diff(f):
    def timed(*args, **kw):
        ts = datetime.now()
        result = f(*args, **kw)
        te = datetime.now()
        diff = (te - ts) / 60.0
        Logger.debug(f'time elapsed {diff} minutes')
        return result
    return timed


@time_diff
def fff():
    print('hellow')

if __name__ == "__main__":
    fff()