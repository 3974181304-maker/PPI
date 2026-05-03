import torch
from torch.func import stack_module_state, vmap, functional_call, grad
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib
# 强制使用 TkAgg 后端，这样会弹出独立窗口
matplotlib.use('TkAgg')


# --- 模型定义 ---
class AirQualityMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# --- 数据准备 (切换到 C6H6 指标) ---
def load_and_prepare_data(csv_path='AirQualityUCI.csv', test_size=0.5, random_state=54):
    try:
        # 注意：CSV 使用分号分割，且小数点用逗号表示
        df = pd.read_csv(csv_path, sep=';', decimal=',')
    except FileNotFoundError:
        print(f"错误：找不到文件 {csv_path}。")
        raise

    # 指标改进：
    # 输入特征：5个传感器读数 + 温度 + 相对湿度 + 绝对湿度
    feature_cols = ["PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S4(NO2)", "PT08.S5(O3)", "T", "RH", "AH"]
    # 目标：苯浓度 (通常比 CO 回归更稳定)
    target_col = "C6H6(GT)"

    # 数据清理
    data = df[feature_cols + [target_col]].apply(pd.to_numeric, errors="coerce")
    data = data.replace(-200, np.nan).dropna()

    X_all, y_all = data[feature_cols].values, data[target_col].values
    X_train, X_test_pool, y_train, y_test_pool = train_test_split(
        X_all, y_all, test_size=test_size, random_state=random_state
    )

    true_mu = np.mean(y_test_pool)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test_pool = scaler.transform(X_test_pool)

    return X_train, y_train, X_test_pool, y_test_pool, true_mu


# --- 核心实验函数 (独立Batch并行) ---
def run_parallel_experiment(sigma_list, K_splits, M_runs, epochs, batch_size, lr,
                            X_train, y_train, X_test_pool, y_test_pool, n_labeled, n_unlabeled, true_mu):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train.shape[1]

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)

    N = len(X_train_t)
    num_batches = N // batch_size
    results = []

    for sigma in sigma_list:
        print(f"\n--- Testing Sigma: {sigma} (Target: C6H6, M={M_runs}) ---")
        total_rejections = 0

        for k in range(K_splits):
            np.random.seed(k)
            all_indices = np.random.choice(len(X_test_pool), size=n_labeled + n_unlabeled, replace=False)
            X_L = torch.tensor(X_test_pool[all_indices[:n_labeled]], dtype=torch.float32, device=device)
            y_L = torch.tensor(y_test_pool[all_indices[:n_labeled]], dtype=torch.float32, device=device)
            X_U = torch.tensor(X_test_pool[all_indices[n_labeled:]], dtype=torch.float32, device=device)

            base_model = AirQualityMLP(input_dim).to(device)
            models = [AirQualityMLP(input_dim).to(device) for _ in range(M_runs)]
            params, buffers = stack_module_state(models)

            def compute_loss(params, buffers, x, y):
                pred = functional_call(base_model, (params, buffers), x)
                return torch.nn.functional.mse_loss(pred, y)

            ft_compute_grad = vmap(grad(compute_loss), in_dims=(0, 0, 0, 0))

            for epoch in range(epochs):
                perms = torch.rand((M_runs, N), device=device).argsort(dim=1)
                for i in range(num_batches):
                    batch_indices = perms[:, i * batch_size: (i + 1) * batch_size]
                    batch_X = X_train_t[batch_indices]
                    batch_y = y_train_t[batch_indices]

                    grads = ft_compute_grad(params, buffers, batch_X, batch_y)

                    with torch.no_grad():
                        for name, param in params.items():
                            g = grads[name]
                            noise = torch.randn_like(g) * sigma
                            param.sub_((g + noise) * lr)

            def predict(params, buffers, x):
                return functional_call(base_model, (params, buffers), x)

            all_preds_L = vmap(predict, in_dims=(0, 0, None))(params, buffers, X_L)
            all_preds_U = vmap(predict, in_dims=(0, 0, None))(params, buffers, X_U)

            y_L_expanded = y_L.unsqueeze(0)
            deltas = y_L_expanded - all_preds_L

            ppi_point_est = torch.mean(all_preds_U, dim=1) + torch.mean(deltas, dim=1)

            var_U = torch.var(all_preds_U, dim=1, unbiased=True)
            var_delta = torch.var(deltas, dim=1, unbiased=True)
            se = torch.sqrt(var_U / n_unlabeled + var_delta / n_labeled)

            t_stats = (ppi_point_est - true_mu) / (se + 1e-12)
            rejections = torch.abs(t_stats) > 1.96
            total_rejections += torch.sum(rejections).item()

        final_rate = total_rejections / (K_splits * M_runs)
        results.append((sigma, final_rate))
        print(f"Sigma: {sigma:.1f} | Rejection Rate: {final_rate:.4f}")

    return results


def main():
    # 数据加载
    X_train, y_train, X_test_pool, y_test_pool, true_mu = load_and_prepare_data()

    sigma_list = [0.1, 0.2, 0.5, 1.0, 2.0,3.0,4.0, 5.0]
    results = run_parallel_experiment(
        sigma_list=sigma_list,
        K_splits=1,
        M_runs=100,
        epochs=30,  # 苯浓度回归较快，增加 epoch 使其更充分
        batch_size=64,
        lr=0.01,  # 苯的量级较小，适当调小学习率
        X_train=X_train,
        y_train=y_train,
        X_test_pool=X_test_pool,
        y_test_pool=y_test_pool,
        n_labeled=20,
        n_unlabeled=150,
        true_mu=true_mu
    )

    # 绘图逻辑
    sigmas = [res[0] for res in results]
    rates = [res[1] for res in results]

    plt.figure(figsize=(10, 6))
    plt.plot(sigmas, rates, marker='s', linestyle='--', color='g', linewidth=2,
             label='C6H6 PPI Rejection Rate')
    plt.title('Rejection Rate vs. Sigma (Predicting Benzene)', fontsize=14)
    plt.xlabel('Gradient Noise Sigma', fontsize=12)
    plt.ylabel('Rejection Rate', fontsize=12)
    plt.ylim(-0.1, 0.3)
    plt.grid(True, which='both', linestyle=':', alpha=0.5)
    plt.legend()

    plt.savefig('benzene_rejection_plot.png')
    print("\n=> 苯浓度 PPI 实验图表已保存为: benzene_rejection_plot.png")
    plt.show()


if __name__ == "__main__":
    main()