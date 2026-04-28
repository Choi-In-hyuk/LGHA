"""
CachedSceneDataset

LiberoDataset을 확장해 사전 계산된 V-JEPA z_scene을 로드.
학습 중 V-JEPA forward 없이 캐시에서 z_scene을 바로 읽음.
"""

import os
import torch
import numpy as np
from hierarchical_vla.libero_bench.dataloader import LiberoDataset


class CachedSceneDataset(LiberoDataset):
    """
    LiberoDataset + z_scene cache 로드.
    __getitem__에서 obs["z_scene_cache"] 반환.
    """

    def __init__(self, cache_dir: str, **kwargs):
        super().__init__(**kwargs)
        self.cache_dir = cache_dir
        self._load_cache()

    def _load_cache(self):
        """각 hdf5 파일에 대응하는 cache .pt 파일 로드."""
        import h5py

        data_dir = self.data_dir
        file_list = sorted(f for f in os.listdir(data_dir) if f.endswith(".hdf5"))

        self.z_cache = []   # 각 demo의 z_scene 텐서 (T, scene_emb_dim)
        self.demo_file_map = []  # 각 demo가 어떤 파일의 몇 번째 demo인지

        for file in file_list:
            task_name = file.replace("_demo.hdf5", "")
            cache_path = os.path.join(self.cache_dir, f"{task_name}.pt")

            if not os.path.exists(cache_path):
                raise FileNotFoundError(
                    f"Cache not found: {cache_path}\n"
                    f"Run: python -m models.precompute_vjepa --data_dir ... --cache_dir {self.cache_dir}"
                )

            demo_cache = torch.load(cache_path, map_location="cpu")  # dict: demo_name → (T, D)

            f = h5py.File(os.path.join(data_dir, file), "r")
            demo_keys = list(f["data"].keys())
            indices = np.argsort([int(k[5:]) for k in demo_keys])
            f.close()

            for i in indices[:self.demos_per_task]:
                demo_name = demo_keys[i]
                self.z_cache.append(demo_cache[demo_name])  # (T, D)

    def __getitem__(self, idx):
        obs, act, mask = super().__getitem__(idx)

        i, start, end = self.slices[idx]
        cache_len = len(self.z_cache[i])
        z_scene = self.z_cache[i][min(start, cache_len - 1)]  # (scene_emb_dim,)

        obs["z_scene_cache"] = z_scene.float()
        return obs, act, mask
