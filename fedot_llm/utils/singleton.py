from threading import Lock


class SingletonMeta(type):

    __instances = {}
    __lock:Lock = Lock()
    def __call__(cls, *args, **kwargs):
        with cls.__lock:
            if cls not in cls.__instances:
                cls.__instances[cls] = super().__call__(*args, **kwargs)
        return cls.__instances[cls]
