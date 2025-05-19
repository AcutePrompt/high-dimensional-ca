import cupy as cp
import numpy as np
import random
import argparse
import os

# Exploration Protocol: GPU-accelerated CA with parameter sweeps

def make_hypergraph(N, M):
    # CPU-side hypergraph neighbor sets
    adj = cp.zeros((N, M), dtype=cp.int32)
    for i in range(N):
        nbrs = set()
        while len(nbrs) < M:
            j = random.randrange(N)
            if j != i:
                nbrs.add(j)
        adj[i, :] = cp.array(list(nbrs), dtype=cp.int32)
    return adj

# Full set of activation functions
ACTS_FULL = {
    'sigmoid': lambda x: 1/(1+cp.exp(-x)),
    'relu':    lambda x: cp.maximum(x,0),
    'sine':    lambda x: cp.sin(x),
    'tanh':    lambda x: cp.tanh(x),
}

# Main GPU simulation
def simulate(args):
    # seeds
    random.seed(args.seed)
    cp.random.seed(args.seed)

    # parameters
    N = args.grid_size**2
    D = args.state_dim
    M = args.max_neighbors

    # select activations
    ACTS = {name: ACTS_FULL[name] for name in args.activations}
    acts_list = list(ACTS.keys())

    # initialize state, rules, types
    x = cp.random.randn(N, D, dtype=cp.float32)
    R = cp.random.randn(N, D, 4, dtype=cp.float32) * args.init_sigma
    act_types = cp.random.randint(len(acts_list), size=N, dtype=cp.int32)

    # global field
    F = cp.zeros(D, dtype=cp.float32)
    # neighbors
    nbrs = make_hypergraph(N, M)
    # clocks
    tau = cp.random.uniform(args.min_tau, args.max_tau, size=N).astype(cp.float32)

    snapshots = []
    for t in range(1, args.steps+1):
        # pulse
        if random.random() < args.pulse_prob:
            F += args.pulse_strength * cp.random.randn(D, dtype=cp.float32)
        # diffuse global field
        F = args.diff_alpha * x.mean(axis=0) + (1-args.diff_alpha)*F

        x_flat = x.reshape(N, D)
        nm = x_flat[nbrs].mean(axis=1)

        rn = cp.random.rand(N, dtype=cp.float32)
        pm = 1.0 / tau
        upd = cp.where(rn < pm)[0]

        for i in upd.get():
            old = x_flat[i]
            params = R[i]
            act = ACTS[acts_list[int(act_types[i])]]
            pre = params[:,0] + params[:,1]*old + params[:,2]*nm[i] + params[:,3]*F
            new = act(pre)
            x_flat[i] = new
            # rule mutation
            if random.random() < args.mu_param:
                R[i] += cp.random.randn(D,4).astype(cp.float32)*args.param_sigma
            # structural mutation
            if random.random() < args.mu_struct:
                act_types[i] = random.randrange(len(acts_list))
            # rewiring
            if not args.disable_rewire:
                delta = float(cp.linalg.norm(new-old))
                if delta > args.curiosity_thresh:
                    slot = random.randrange(M)
                    cand = random.randrange(N)
                    nbrs[i, slot] = int(cand)
        # snapshot
        if t % args.snapshot_interval == 0:
            snapshots.append((t, x.get(), F.get()))
            print(f"[Progress] t={t}")

    os.makedirs(args.output_dir, exist_ok=True)
    for t, xs, Fs in snapshots:
        np.save(os.path.join(args.output_dir, f"state_{t}.npy"), xs)
        np.save(os.path.join(args.output_dir, f"field_{t}.npy"), Fs)
        print(f"[Saved] t={t}")
    print("Done.")

if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--grid_size',        type=int,   default=50)
    p.add_argument('--state_dim',        type=int,   default=16)
    p.add_argument('--max_neighbors',    type=int,   default=10)
    p.add_argument('--steps',            type=int,   default=10000)
    p.add_argument('--min_tau',          type=float, default=10)
    p.add_argument('--max_tau',          type=float, default=100)
    p.add_argument('--diff_alpha',       type=float, default=0.5,
                       help='mix ratio for global field')
    p.add_argument('--init_sigma',       type=float, default=0.1)
    p.add_argument('--mu_param',         type=float, default=0.01)
    p.add_argument('--param_sigma',      type=float, default=0.05)
    p.add_argument('--mu_struct',        type=float, default=0.005)
    p.add_argument('--pulse_prob',       type=float, default=0.001)
    p.add_argument('--pulse_strength',   type=float, default=1.0)
    p.add_argument('--curiosity_thresh', type=float, default=1.0)
    p.add_argument('--snapshot_interval',type=int,   default=100)
    p.add_argument('--output_dir',       type=str,   default='output')
    p.add_argument('--seed',             type=int,   default=42)
    p.add_argument('--disable_rewire',   action='store_true',
                       help='turn off curiosity-driven rewiring')
    p.add_argument('--activations',      nargs='+', choices=list(ACTS_FULL.keys()),
                       default=list(ACTS_FULL.keys()),
                       help='subset of activation functions to use')
    args = p.parse_args()
    simulate(args)
