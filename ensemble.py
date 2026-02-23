import os

from spmd_wp import SPMDWP

import torch
import torch.multiprocessing as mp
import torch.distributions as D

from particle import Particle



def pf_worker_loop(
    worker_id: int,
    particle_indices: list[int],
    hyp_params: dict,
    cmd_q: mp.Queue,
    res_q: mp.Queue,
):  
    """
    The worker loop defines the various 'methods' (commands) that can be executed by workers. \\
    Each worker is expected to own a slice of the particle list, and is responsible for operations on that slice. 
    """

    # Disable within worker multithreading to prevent oversubscription
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    # Create local particles for this worker
    local_particles = [Particle(**hyp_params) for _ in particle_indices]
    local_logw = torch.zeros(len(local_particles), dtype=torch.float64)

    while True:
        # Read message from the command queue
        msg = cmd_q.get()
        if msg is None:
            break
        
        # Command is the first element of the message tuple
        cmd = msg[0]

        # ------------------------------------------------------------------
        #   Define commands ('methods') that can be executed by the worker
        # ------------------------------------------------------------------
        if cmd == "before":
            """
            Execute before_action() on all particles within the worker to process the observaiton o_t
            """
            # Observation is passed in as the second element of the message
            (o_t,) = msg[1:] # msg: ("before", o_t)

            # Each particle processes the observation
            for particle in local_particles:
                particle.before_action(o_t)
            
            # If all particles finished processing, return worker id
            res_q.put(("before_done", worker_id))
            

        elif cmd == "act":
            """
            Execute sample_action() on all particles within the worker to yield candidates for a_t
            """
            # Each particle samples an action
            actions = torch.empty(len(local_particles), dtype=torch.long)
            
            for i, particle in enumerate(local_particles):
                actions[i] = particle.sample_action()

            # Drop selected a_t candidates into results queue
            res_q.put(("actions", worker_id, actions))

        
        elif cmd == "after":
            """
            Execute after_action() on all particles within the worker to process the observaiton o_t
            """
            # Observation, action, and reward is passed in as the second message element
            o_t, a_t, r_t = msg[1:] # msg: ("after", o_t, a_t, r_t)

            # Each particle conducts inference on the latent state and contexts, and updates internal parameters
            vec_s   = torch.empty(len(local_particles), dtype=torch.long)
            vec_c_o = torch.empty(len(local_particles), dtype=torch.long)
            vec_c_r = torch.empty(len(local_particles), dtype=torch.long)
            vec_j   = torch.empty(len(local_particles), dtype=torch.long)

            for i, particle in enumerate(local_particles):
                s_t, c_o_t, c_r_t, j_t = particle.after_action(o_t, int(a_t), float(r_t))
                vec_s[i]        = s_t
                vec_c_o[i]      = c_o_t
                vec_c_r[i]      = c_r_t
                vec_j[i]        = j_t
                local_logw[i]   = float(particle.log_weight)

            # Deposit inferred state, contexts, jump, and log weight into the results queue
            res_q.put(("after_done", worker_id, local_logw.clone(), vec_s, vec_c_o, vec_c_r, vec_j))


        elif cmd == "get_states":
            """
            Get the minimal state vector needed to reconstitute each particle
            """
            # Read in the requested indices
            (req_global_i,) = msg[1:] # msg: ("get_states", requested_global_indices_list)
            
            # Gather states for requested indices
            global_to_local = {g: li for li, g in enumerate(particle_indices)} # particle_indices maps local->global
            
            out = {}
            for g in req_global_i:
                if g in global_to_local:
                    li = global_to_local[g]
                    out[g] = local_particles[li].to_state()

            # Deposit minimal states in the results queue
            res_q.put(("states", worker_id, out))


        elif cmd == "set_particles_from_states":
            """
            Reconstitute each particle from the minimal state
            """
            # Read in the state dicts
            (new_states,) = msg[1:] # msg: ("set_particles_from_states", new_states_list_in_worker_order)
            assert len(new_states) == len(local_particles)
            
            # Reconstitute particles from minimal states
            local_particles = [Particle.from_state(hyp_params, st) for st in new_states]
            # reset weights inside particles is done in your Ensemble.resample (you set log_weights=0)
            for p in local_particles:
                p.log_weight = 0.0
            local_logw.zero_()
            
            # If all particles have been reconstituted, return worker id
            res_q.put(("set_done", worker_id))

        
        elif cmd == "get_particles":
            """
            Return a list of particle objects within the worker
            """
            # Read in the indices of requrested particles
            (req_global_i,) = msg[1:] # msg: ("get_particles", requested_global_indices_list)
            
            # Collect copies of the particles into the output list
            global_to_local = {g: li for li, g in enumerate(particle_indices)} 

            out = {}
            for g in req_global_i:
                if g in global_to_local:
                    li = global_to_local[g]
                    out[g] = local_particles[li]  # full Particle object
            
            # Deposit each object in the results queue
            res_q.put(("particles", worker_id, out))


        else:
            raise ValueError(f"Unknown cmd {cmd}")


class MPEnsemble(SPMDWP):
    def __init__(self, N: int, n_workers: int, hyp_params: dict):
        # Init the base spmdwp class instance
        super().__init__(N, n_workers, pf_worker_loop, hyp_params)
        
        # Initalise extra data structures that are tracked during particle filtering
        self.log_weights    = torch.zeros(N, dtype=torch.float64)
        self.vec_s_t        = torch.zeros(N, dtype=torch.long)
        self.vec_c_o_t      = torch.zeros(N, dtype=torch.long)
        self.vec_c_r_t      = torch.zeros(N, dtype=torch.long)
        self.vec_j_t        = torch.zeros(N, dtype=torch.long)

    @property
    def weights(self):
        return torch.exp(self.log_weights - torch.logsumexp(self.log_weights, dim=0))

    def before_action(self, o_t: torch.Tensor):
        """
        Process the incoming observation o_t
        """
        # Send command to run .before_action(o_t) on each particle
        self._broadcast(("before", o_t))

        # Check that all workers have finished
        self._gather_n("before_done", self.n_workers)

    def select_action(self):
        """
        Select an action a_t
        """
        # Send command to run .sample_action() on each particle
        self._broadcast(("act",))

        # Collect results from all workers, and stitch back into global order
        msgs = self._gather_n("actions", self.n_workers)

        actions = torch.empty(self.N, dtype=torch.long)
        for (_, worker_id, acts_local) in msgs:
            shard = self.shards[worker_id]
            actions[shard] = acts_local

        # Pick a particle from which the action is emitted
        selected_particle = D.Categorical(probs=self.weights).sample().item()
        selected_action = int(actions[selected_particle].item())

        return selected_action, actions

    def after_action(self, a_t: int, r_t: float, o_t: torch.Tensor):
        """
        Process the resulting o_t, a_t, r_t emissions at time t
        """
        # Broadcast the command to run .after_action(o_t, a_t, r_t) on each particle
        self._broadcast(("after", o_t, int(a_t), float(r_t)))
        
        # Collect results from all workers, and stitch back into global order
        msgs = self._gather_n("after_done", self.n_workers)

        for (_, worker_id, logw_local, vec_s, vec_c_o, vec_c_r, vec_j) in msgs:
            shard = self.shards[worker_id]
            self.log_weights[shard] = logw_local
            self.vec_s_t[shard] = vec_s
            self.vec_c_o_t[shard] = vec_c_o
            self.vec_c_r_t[shard] = vec_c_r
            self.vec_j_t[shard] = vec_j

        #self.resample()

    def resample(self):
        """
        Resample particles according to weights
        """
        # sample N ancestor indices using current log weights
        ancestor = D.Categorical(logits=self.log_weights).sample((self.N,))
        ancestor = ancestor.to(torch.long)

        # 1) Ask each worker for states of the ancestors it owns (batched)
        needed = ancestor.unique().tolist() # avoid duplicate requests
        self._broadcast(("get_states", needed)) # broadcast state request
        state_msgs = self._gather_n("states", self.n_workers) # collect results

        ancestor_state = {}
        for (_, worker_id, partial) in state_msgs:
            ancestor_state.update(partial)

        # Check that states have been recieved from all required ancestors
        missing = [i for i in needed if i not in ancestor_state]
        if missing:
            raise RuntimeError(f"Missing ancestor states for indices: {missing[:10]} ...")

        # 2) build per-worker new state lists (in that worker's local particle order)
        for worker_id, shard in enumerate(self.shards):
            new_states = []
            for global_pos in shard:
                anc_idx = int(ancestor[global_pos].item())
                new_states.append(ancestor_state[anc_idx])
            self.cmd_qs[worker_id].put(("set_particles_from_states", new_states))

        self._gather_n("set_done", self.n_workers)

        # 3) reset log_weights after resampling
        self.log_weights.zero_()
    

    @property
    def particles(self) -> list:
        """
        Returns a *snapshot* list of Particle objects in global index order.

        NOTE:
        - These are deserialized copies of the particles in the main process (not live worker objects).
        - This is expensive (pickles all particles) and should only be used when extracting unique states is required.
        """
        all_idx = list(range(self.N))
        self._broadcast(("get_particles", all_idx))
        msgs = self._gather_n("particles", self.n_workers)

        # Merge {global_idx: Particle} maps
        merged = {}
        for (_, worker_id, partial) in msgs:
            merged.update(partial)

        missing = [i for i in all_idx if i not in merged]
        if missing:
            raise RuntimeError(f"Missing particles for indices: {missing[:10]} ...")

        # Stitch into a single ordered list
        return [merged[i] for i in range(self.N)]