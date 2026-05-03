import os
import random
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib


matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.func import stack_module_state, vmap, functional_call, grad



def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)   # 输出 (batch,)



def generate_data(n, d, dist_type='normal', device='cpu'):
    if dist_type == 'normal':
        X = torch.randn(n, d, device=device)
    elif dist_type == 'uniform':
        X = torch.rand(n, d, device=device) * 2 - 1
    elif dist_type == 'binomial':
        ones = torch.ones(n, d, device=device)
        probs = torch.tensor([0.5], device=device)
        X = torch.binomial(ones, probs)
    else:
        raise ValueError("Unsupported distribution type")

    if d > 1:
        h_X = X[:, 0] ** 2 + torch.sin(X[:, 1])
    else:
        h_X = X[:, 0] ** 2

    epsilon = torch.randn(n, device=device) * 0.5
    Y_raw = h_X + epsilon
    return X, Y_raw



def run_parallel_experiment(
    X_train, Y_train, X_L, Y_L, X_U,
    d, batch_size, T, eta, M_runs, device
):
    N = X_train.size(0)


    models = [MLP(d).to(device) for _ in range(M_runs)]
    base_model = MLP(d).to(device)


    params, buffers = stack_module_state(models)

    def compute_loss(params, buffers, x, y):
        pred = functional_call(base_model, (params, buffers), x)
        return torch.nn.functional.mse_loss(pred, y)


    ft_compute_grad = vmap(grad(compute_loss), in_dims=(0, 0, 0, 0))


    for step in range(T):
 
        perms = torch.rand((M_runs, N), device=device).argsort(dim=1)

        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)

            batch_indices = perms[:, start_idx:end_idx]      # (M_runs, bs)
            batch_X = X_train[batch_indices]                 # (M_runs, bs, d)
            batch_y = Y_train[batch_indices]                 # (M_runs, bs)

            grads = ft_compute_grad(params, buffers, batch_X, batch_y)


            with torch.no_grad():
                for name in params.keys():
                    params[name].sub_(eta * grads[name])


    def predict(params, buffers, x):
        return functional_call(base_model, (params, buffers), x)

    all_preds_L = vmap(predict, in_dims=(0, 0, None))(params, buffers, X_L)  # (M_runs, n_l)
    all_preds_U = vmap(predict, in_dims=(0, 0, None))(params, buffers, X_U)  # (M_runs, n_u)


    Delta = Y_L.unsqueeze(0) - all_preds_L
    theta_PPI_hat = torch.mean(Delta, dim=1) + torch.mean(all_preds_U, dim=1)
    sigma_sq_Delta_hat = torch.var(Delta, dim=1, unbiased=True)

    n_l = X_L.shape[0]
    t_stat = theta_PPI_hat / torch.sqrt(sigma_sq_Delta_hat / n_l + 1e-12)

    reject = torch.abs(t_stat) > 1.96
    return reject.float().mean().item()



def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current acceleration device: {device}")

    os.makedirs("results", exist_ok=True)

    n_l = 15
    N_u = 1000
    n_t = 2000
    d = 10
    T = 30
    eta = 0.5
    M_runs = 200

    distributions = ['normal', 'uniform', 'binomial']
    batch_sizes = [16, 32, 64, 128, 256, 512, n_t]

    criterion_results = []
    print("Start the synthetic PPI + SGD experiment...")

    for dist in distributions:
        print(f"\n---> Current data distribution: {dist.upper()}")

        X_train, Y_train_raw = generate_data(n_t, d, dist, device)
        X_L, Y_L_raw = generate_data(n_l, d, dist, device)
        X_U, _ = generate_data(N_u, d, dist, device)

        Y_L_mean = Y_L_raw.mean()
        Y_train = Y_train_raw - Y_L_mean
        Y_L = Y_L_raw - Y_L_mean

        for B in tqdm(batch_sizes, desc=f"Batch Size Sweep [{dist}]"):
            rejection_rate = run_parallel_experiment(
                X_train=X_train,
                Y_train=Y_train,
                X_L=X_L,
                Y_L=Y_L,
                X_U=X_U,
                d=d,
                batch_size=B,
                T=T,
                eta=eta,
                M_runs=M_runs,
                device=device
            )

            criterion_results.append({
                'Distribution': dist,
                'Batch Size': B,
                'Rejection Rate': rejection_rate
            })

            print(f"Batch Size: {B:4d} | Rejection Rate: {rejection_rate:.4f}")


    df_results = pd.DataFrame(criterion_results)
    csv_path = "results/ppi_sgd_parallel_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\nThe experimental data has been saved to: {csv_path}")


    plt.figure(figsize=(10, 6))

    for dist in distributions:
        sub = df_results[df_results['Distribution'] == dist].sort_values('Batch Size')
        plt.plot(
            sub['Batch Size'],
            sub['Rejection Rate'],
            marker='s',
            linewidth=2,
            label=dist
        )

    plt.axhline(y=0.05, color='r', linestyle='--', label=r'Target $\alpha$ (0.05)')
    plt.xscale('log', base=2)
    plt.xticks(batch_sizes, labels=[str(b) for b in batch_sizes])

    plt.title('Evaluation of PPI Validity: Rejection Rate vs SGD Batch Size', fontsize=14)
    plt.xlabel('Batch Size (Log Scale)', fontsize=12)
    plt.ylabel(r'Rejection Rate ($\hat{\alpha}$)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)

    plot_path = "results/rejection_rate_vs_batchsize_parallel.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f"The chart has been saved to: {plot_path}")

    plt.show()


if __name__ == "__main__":
    main()
