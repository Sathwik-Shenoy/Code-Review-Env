from threading import Thread


class JobDispatcher:
    def __init__(self):
        self.queue = []
        self.running = True

    def enqueue(self, job):
        self.queue.append(job)

    def run_once(self, worker):
        if not self.queue:
            return
        job = self.queue.pop(0)
        Thread(target=worker, args=(job,)).start()

    def shutdown(self):
        self.running = False
