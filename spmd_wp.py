import torch.multiprocessing as mp
from typing import Callable

class SPMDWP():
    """
    Single program multiple data worker pool.\\
    Currently used as a base class for the ensemble of particles to inherit from.
    """
    def __init__(self, N: int, n_workers: int, worker_loop : Callable, global_params : dict):
        """
        Initialise the queues and workers that use the queues

        :param int N: number of replicate instances to be distributed among workers
        :param int n_workers: number of worker processes that should be spawned
        :param Callable worker_loop: the function implementing the various commands that workers should respond to
        :param dict global_params: parameters passed to all workers
        """
        self.N = N

        assert 1 <= n_workers <= mp.cpu_count(), "Number of workers is limited to the number of hardware threads"
        assert n_workers <= N, "Number of workers is maximally the number of replicate instances"
        self.n_workers = n_workers

        # Set up the multiprocessing starter method
        ctx = mp.get_context(method='spawn') # we currently hardcode 'spawn' as 'fork' risks sharing memory
        
        # Set up a command queue for each worker
        self.cmd_qs = [ctx.Queue() for _ in range(self.n_workers)]
        
        # Set up a global results queue
        self.res_q = ctx.Queue()

        # Partition global replicate indices into shards for each prospective worker
        shards = []
        base = N // self.n_workers
        rem = N % self.n_workers
        start = 0
        for w in range(self.n_workers):
            n = base + (1 if w < rem else 0)
            inds = list(range(start, start + n))
            shards.append(inds)
            start += n
        self.shards = shards
        
        # Initialise a worker dedicated to processing replicates within each shard
        self.workers = []
        for worker_i, inds in enumerate(shards):
            p = ctx.Process(
                target=worker_loop,
                args=(worker_i, inds, global_params, self.cmd_qs[worker_i], self.res_q),
                daemon=True,
            )
            p.start()
            self.workers.append(p)

    def _broadcast(self, msg):
        """
        Broadcast a command message to the command queue of every worker
        """
        for q in self.cmd_qs:
            q.put(msg)

    def _gather_n(self, expected_type: str, n: int):
        """
        Gather results of the specified type from the results queue
        """
        out = []
        # Wait until n workers have deposited results
        while len(out) < n:
            m = self.res_q.get()
            if m[0] == expected_type:
                out.append(m)
        return out

    def deallocate(self):
        """
        Exit out of the worker loops
        """
        # Tell every worker to exit its loop
        for q in self.cmd_qs:
            q.put(None)  # sentinel your worker_loop treats as "break"

        # Wait for processes to actually terminate
        for p in self.workers:
            p.join()