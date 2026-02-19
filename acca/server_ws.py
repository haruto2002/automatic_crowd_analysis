import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import websockets
from hydra.utils import instantiate
from omegaconf import OmegaConf


def build_detector(conf_path: Path):
    cfg = OmegaConf.load(conf_path)
    detector = instantiate(cfg)
    return detector


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8765
    max_size: int | None = None


class InferenceServer:
    def __init__(self, cfg_path: Path, server_cfg: Optional[ServerConfig] = None):
        self.cfg_path = cfg_path
        self.server_cfg = server_cfg or ServerConfig()

        # Detectorはここで1回だけ初期化
        self.detector = build_detector(cfg_path)

        # 必要ならロック（GPU推論がスレッドセーフでない場合に備える）
        self._lock = asyncio.Lock()

    def _run_inference(self, images: List[np.ndarray]) -> List[np.ndarray]:
        if not images:
            return []
        return self.detector(images)

    async def handler(self, ws):
        async for msg in ws:
            try:
                arr = np.frombuffer(msg, np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

                # 同時に複数クライアントが来るとGPU競合することがあるので、
                # まずはロックで直列化（動作確認優先）
                async with self._lock:
                    results = self._run_inference([frame])

                if not results:
                    detections = []
                else:
                    detections = results[0]  # batch=1を想定

                print(detections.shape)
                resp: bytes = detections.tobytes()
                await ws.send(resp)

            except Exception as e:
                err = str(e)
                await ws.send(err.tobytes())

    async def serve(self):
        async with websockets.serve(
            self.handler,
            self.server_cfg.host,
            self.server_cfg.port,
            max_size=self.server_cfg.max_size,
        ):
            print(
                f"[server] detector={type(self.detector).__name__} cfg={self.cfg_path}"
            )
            print(
                f"[server] listening on ws://{self.server_cfg.host}:{self.server_cfg.port}"
            )
            await asyncio.Future()


if __name__ == "__main__":
    cfg_path = Path("acca/conf/rt_conf/p2pnet.yaml")
    server = InferenceServer(cfg_path)
    asyncio.run(server.serve())
