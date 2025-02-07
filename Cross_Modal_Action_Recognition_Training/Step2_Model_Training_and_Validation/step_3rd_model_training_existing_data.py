from torch.utils.data import Dataset, DataLoader


class HandDataset(Dataset):
    def __init__(self, npz_file):
        data = np.load(npz_file)
        self.emg = torch.FloatTensor(data['emg'])
        self.keypoints = torch.FloatTensor(data['keypoints']).view(-1, 21, 2)

    def __len__(self):
        return len(self.emg)

    def __getitem__(self, idx):
        return self.emg[idx], self.keypoints[idx]


# 数据加载
dataset = HandDataset('backup_database_and_model_repository/handpose_dataset.npz')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 训练配置
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# 训练循环
for epoch in range(50):
    total_loss = 0
    for emg, kp in dataloader:
        optimizer.zero_grad()
        pred = model(emg)
        loss = criterion(pred, kp)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}")