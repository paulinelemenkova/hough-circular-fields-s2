import numpy as np, pandas as pd, rasterio, rasterio.windows, torch
import albumentations as A
from torch.utils.data import Dataset


class PivotChips(Dataset):
    def __init__(self, index_parquet, split, stats_npz, train=True):
        df = (
            pd.read_parquet(index_parquet)
            if str(index_parquet).endswith(".parquet")
            else pd.read_csv(index_parquet)
        )
        self.df = df[df.split == split].reset_index(drop=True)
        s = np.load(stats_npz)
        self.mu, self.sd = s["mean"][:, None, None], s["std"][:, None, None]
        self.aug = (
            A.Compose([A.D4(p=1.0), A.RandomBrightnessContrast(0.05, 0.05, p=0.3)])
            if train
            else None
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        win = rasterio.windows.Window(r.col_off, r.row_off, 256, 256)
        with rasterio.open(r.stack) as src:
            x = src.read(window=win).astype(np.float32)
        with rasterio.open(r.instances) as src:
            inst = src.read(1).astype(np.int32)
        x = np.nan_to_num((x - self.mu) / self.sd)
        if self.aug is not None:
            out = self.aug(image=x.transpose(1, 2, 0), mask=inst)
            x, inst = out["image"].transpose(2, 0, 1), out["mask"]
        ids = np.unique(inst)[1:]
        masks = (
            np.stack([inst == k for k in ids])
            if len(ids)
            else np.zeros((0, *inst.shape), bool)
        )
        boxes = [
            [*np.argwhere(m).min(0)[::-1], *np.argwhere(m).max(0)[::-1]] for m in masks
        ]
        target = {
            "semantic": torch.from_numpy((inst > 0).astype(np.float32)),
            "masks": torch.from_numpy(masks.astype(np.uint8)),
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.ones(len(ids), dtype=torch.int64),
        }
        return torch.from_numpy(np.ascontiguousarray(x)), target
