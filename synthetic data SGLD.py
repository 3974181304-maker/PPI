import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib
# 强制使用 TkAgg 后端，这样会弹出独立窗口
matplotlib.use('TkAgg')

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


def generate_data(n, d, dist_type='normal'):
    if dist_type == 'normal':
        X = torch.randn(n, d)
    elif dist_type == 'uniform':
        X = torch.rand(n, d) * 2 - 1  # [-1, 1] 
    elif dist_type == 'binomial':
        X = torch.binomial(torch.ones(n, d), torch.tensor([0.5])) # 0 or 1
    else:
        raise ValueError("Unsupported distribution type")

    h_X = (X[:, 0]**2 + torch.sin(X[:, 1])).view(-1, 1) if d > 1 else (X[:, 0]**2).view(-1, 1)

    epsilon = torch.randn(n, 1) * 0.5
    Y_raw = h_X + epsilon
    
    return X, Y_raw


def run_single_experiment(X_train, Y_train, X_L, Y_L, X_U, d, B, T, sigma, eta):
    
    model = MLP(input_dim=d)
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=B, shuffle=True)
    criterion = nn.MSELoss()

    model.train()
    for step in range(T):
        for X_b, Y_b in train_loader:

            Y_hat = model(X_b)

            loss = criterion(Y_hat, Y_b)
            
            model.zero_grad()
            loss.backward()
            
            with torch.no_grad():
                for param in model.parameters():
                    if param.grad is not None:
                        noise = torch.randn_like(param) * sigma
                        g_tilde = param.grad + noise
                        param.data.sub_(eta * g_tilde)

    model.eval()
    with torch.no_grad():
        f_X_L = model(X_L)
        f_X_U = model(X_U)
        
        Delta = Y_L - f_X_L # Rectifier
        
        theta_PPI_hat = torch.mean(Delta) + torch.mean(f_X_U)
        
        sigma_sq_Delta_hat = torch.var(Delta, unbiased=True)
        
        n_l = X_L.shape[0] 
        t_stat = theta_PPI_hat / torch.sqrt(sigma_sq_Delta_hat / n_l)
        
        reject = torch.abs(t_stat).item() > 1.96
        
    return 1 if reject else 0



def main():
    set_seed(42)
    
    os.makedirs("results", exist_ok=True)
#参数修改过
    n_l = 20          # labeled data size
    N_u = 200         # unlabeled data size
    n_t = 1000         # train data size
    d = 10             # feature dimension
    B = 64             # Batch size
    T = 10             # train epochs
    eta = 0.01         # learning rate
    M = 200            # number of experiments
    

    distributions = ['normal', 'uniform', 'binomial']
    sigma_values = [0, 0.2,0.4, 0.6,0.8, 1.0,1.2,1.4,1.6,1.8,2.0]
    
    results = []

    print("Start the PPI + SGLD experiment...")
    for dist in distributions:
        print(f"\n---> Current data distribution: {dist.upper()}")

        X_train, Y_train_raw = generate_data(n_t, d, dist)
        X_L, Y_L_raw = generate_data(n_l, d, dist)
        X_U, _ = generate_data(N_u, d, dist) 
        Y_train = Y_train_raw - Y_L_raw.mean()
        Y_L = Y_L_raw - Y_L_raw.mean() 

        for sigma in tqdm(sigma_values, desc="Traverse Sigma"):
            
            R = 0 

            for m in range(M):
                rejected = run_single_experiment(
                    X_train, Y_train, X_L, Y_L, X_U, 
                    d, B, T, sigma, eta
                )
                if rejected:
                    R += 1

            rejection_rate = R / M
            
            results.append({
                'Distribution': dist,
                'Sigma': sigma,
                'Rejection Rate': rejection_rate
            })

    df_results = pd.DataFrame(results)
    csv_path = "results/ppi_sgld_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\nThe experimental data has been saved to:{csv_path}")

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df_results, 
        x='Sigma', 
        y='Rejection Rate', 
        hue='Distribution', 
        marker='o',
        linewidth=2,
        markersize=8
    )
    
    plt.axhline(y=0, color='r', linestyle='--')
    
    plt.title(r'Evaluation of PPI Validity: Rejection Rate vs SGLD Noise Scale ($\sigma$)', fontsize=14)
    plt.xlabel(r'SGLD Noise Scale ($\sigma$)', fontsize=12)
    plt.ylabel(r'Rejection Rate ($\hat{\alpha}$)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    
    plot_path = "results/rejection_rate_vs_sigma.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f"The chart has been saved to:{plot_path}")
    
    plt.show()

if __name__ == "__main__":
    main()
