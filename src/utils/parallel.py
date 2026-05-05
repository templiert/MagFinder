import threading

from java.lang import Runtime

from src.utils.log import dlog, log_block


def start_threads(function, fraction_cores=1, arguments=(), n_threads=None):
    """Runs `function` in parallel across multiple threads.

    Worker functions must accept an AtomicInteger as their first argument,
    which they use to atomically claim work items (work-stealing pattern):

        def my_worker(atom, items, ...):
            while atom.get() < len(items):
                k = atom.getAndIncrement()
                if k < len(items):
                    process(items[k])

    The AtomicInteger must be created by the caller
    and passed as the first element of `arguments`,
    e.g.: arguments=(AtomicInteger(0), items, ...)
    """
    if n_threads is None:
        n_threads = max(
            int(Runtime.getRuntime().availableProcessors() * fraction_cores), 1
        )
    threads = []
    with log_block("Running {} threads".format(n_threads)):
        for id_thread in range(n_threads):
            thread = threading.Thread(target=function, args=arguments)
            threads.append(thread)
            thread.start()
            dlog("Thread {} started".format(id_thread))
        for id_thread, thread in enumerate(threads):
            thread.join()
            dlog("Thread {} ended".format(id_thread))
