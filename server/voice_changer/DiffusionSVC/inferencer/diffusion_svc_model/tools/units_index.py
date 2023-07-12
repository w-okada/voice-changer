import os
import numpy as np
from tqdm import tqdm
import pickle
import torch
from pathlib import Path

def train_index(path):
    import faiss
    # from: RVC https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
    # 获取文件列表
    listdir_res = []
    for file in os.listdir(path):
        listdir_res.append(os.path.join(path, file))
    npys = []
    # 读取文件
    print(" [INFO] Loading the Units files...")
    for name in tqdm(sorted(listdir_res)):
        phone = np.load(name)
        npys.append(phone)
    # 正式内容
    big_npy = np.concatenate(npys, 0)
    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    index = faiss.index_factory(big_npy.shape[1], "IVF%s,Flat" % n_ivf)
    index_ivf = faiss.extract_index_ivf(index)  #
    index_ivf.nprobe = 1
    index.train(big_npy)
    batch_size_add = 8192
    print(" [INFO] Training the Units indexes...")
    for i in tqdm(range(0, big_npy.shape[0], batch_size_add)):
        index.add(big_npy[i: i + batch_size_add])
    return index


class UnitsIndexer:
    def __init__(self, exp_path):
        exp_path = Path(exp_path)
        self.model = None
        self.exp_path = exp_path
        self.spk_id = -1
        self.active = False
        self.big_all_npy = None

    def load(self, spk_id=1, exp_path=None):
        if (exp_path is not None) and os.path.samefile(self.exp_path, Path(exp_path)):
            exp_path = Path(exp_path)
            self.exp_path = exp_path
        index_pkl_path = os.path.join(self.exp_path, 'units_index', f'spk{spk_id}.pkl')
        if not os.path.isfile(index_pkl_path):
            self.active = False
            print(f" [WARNING]  No such file as {index_pkl_path}, Disable Units Indexer.")
        else:
            import faiss
            self.spk_id = spk_id
            self.active = True
            with open(index_pkl_path, "rb") as f:
                self.model = pickle.load(f)[str(spk_id)]
            self.big_all_npy = self.model.reconstruct_n(0, self.model.ntotal)
            print(f" [INFO]  Successfully load Units Indexer from {index_pkl_path}.")

    def __call__(self, units_t, spk_id=1, ratio=1):
        if self.spk_id != spk_id:
            self.load(spk_id=spk_id)
        if self.active:
            units = units_t.squeeze().to('cpu').numpy()
            # print(" [INFO] Starting feature retrieval...")
            score, ix = self.model.search(units, k=8)
            weight = np.square(1 / score)
            weight /= weight.sum(axis=1, keepdims=True)
            npy = np.sum(self.big_all_npy[ix] * np.expand_dims(weight, axis=2), axis=1)
            units = ratio * npy + (1 - ratio) * units
            units_t = torch.from_numpy(units).unsqueeze(0).float().to(units_t.device)
            # print(f" [INFO] End feature retrieval...Ratio is {ratio}.")
            return units_t
        else:
            print(f" [WARNING] Units Indexer is not active, disable units index.")
            return units_t
