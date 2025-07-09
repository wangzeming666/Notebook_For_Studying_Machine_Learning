def add_knn_price_features_radius_v2(
    df, base_df, lat_col='latitude', lon_col='longitude',
    target_col='sale_price', radii=[0.005, 0.01, 0.02], max_neighbors=300
):
    df = df.copy()
    coords_query = df[[lat_col, lon_col]].astype(np.float64).values
    coords_base = base_df[[lat_col, lon_col]].astype(np.float64).values
    targets_base = base_df[target_col].astype(np.float64).values

    tree = KDTree(coords_base, leaf_size=40, metric='euclidean')

    for r in radii:
        neighbor_indices = tree.query_radius(coords_query, r=r)
        
        stats = {
            'mean': [], 'std': [], 'min': [], 'max': [],
            'range': [], 'iqr': [], 'median': [], 'count': []
        }

        for i, inds in enumerate(neighbor_indices):
            if base_df is df:
                inds = inds[inds != i]

            if len(inds) == 0:
                for key in stats:
                    stats[key].append(np.nan if key != 'count' else 0)
            else:
                if len(inds) > max_neighbors:
                    inds = inds[:max_neighbors]
                vals = targets_base[inds]
                stats['mean'].append(np.mean(vals))
                stats['std'].append(np.std(vals))
                stats['min'].append(np.min(vals))
                stats['max'].append(np.max(vals))
                stats['range'].append(np.max(vals) - np.min(vals))
                stats['iqr'].append(np.percentile(vals, 75) - np.percentile(vals, 25))
                stats['median'].append(np.median(vals))
                stats['count'].append(len(inds))

        for key in stats:
            df[f'knn_radius_{key}_r{r}'] = stats[key]

    return df






import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np

# ========== 1. 定义网络结构 ==========
class QuantileRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # 输出两个分位数：lower 和 upper
        )

    def forward(self, x):
        return self.net(x)


# ========== 2. Quantile Loss (Pinball Loss) ==========
def pinball_interval_loss(preds, target, alpha=0.8, penalty_weight=10.0):
    y_low = preds[:, 0]
    y_high = preds[:, 1]
    y = target.squeeze()

    # 覆盖惩罚
    under = (y < y_low).float()
    over = (y > y_high).float()
    miss_penalty = (2.0 / (1 - alpha)) * ((y_low - y) * under + (y - y_high) * over)

    # 区间宽度惩罚
    width_penalty = (y_high - y_low)

    # 结构惩罚项：low > high 是不合法的
    structure_penalty = torch.mean(torch.relu(y_low - y_high)) * penalty_weight

    return torch.mean(width_penalty + miss_penalty) + structure_penalty



# ========== 3. 训练函数 ==========
def train_model(X, y, alpha=0.8, epochs=100, batch_size=1024, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据准备
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size)

    # 模型定义
    model = QuantileRegressor(input_dim=X.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    best_model = None
    patience, patience_counter = 10, 0

    # ========== 4. 训练循环 ==========
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = pinball_interval_loss(preds, yb, alpha)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 验证
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = pinball_interval_loss(preds, yb, alpha)
                val_loss += loss.item()

        print(f"Epoch {epoch+1}, Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}")

        with torch.no_grad():
            preds = model(X_val_tensor.to(device)).cpu().numpy()
            y_true = y_val_tensor.cpu().numpy()
            coverage = np.mean((y_true >= preds[:, 0]) & (y_true <= preds[:, 1]))
            avg_width = np.mean(preds[:, 1] - preds[:, 0])
            print(f"Val Coverage: {coverage:.4f}, Avg Width: {avg_width:.2f}")


        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break

    model.load_state_dict(best_model)
    torch.save(best_model, 'best_interval_model.pth')
    return model


# ========== 5. 用法示例 ==========
# X_final: numpy 数组 (N, D)
# y_train_lower, y_train_upper: numpy 数组 (N,)
# 合并成 target 矩阵：
# y_train = np.stack([y_train_lower, y_train_upper], axis=1)

# model = train_model(X_final, y_train)

# 输出预测
# model(torch.tensor(X_test).to(device)).cpu().detach().numpy()






