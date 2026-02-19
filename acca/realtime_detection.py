import asyncio
import signal
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf


class MJPEGFrameExtractor:
    def __init__(self):
        self.buf = bytearray()

    def feed(self, data: bytes):
        self.buf.extend(data)

    def pop_frames(self):
        frames = []
        buf = self.buf

        while True:
            soi = buf.find(b"\xff\xd8")  # Start Of Image
            if soi < 0:
                # SOIがないならバッファを捨てすぎないように末尾だけ残す
                if len(buf) > 1024 * 1024:
                    del buf[:-1024]
                break

            eoi = buf.find(b"\xff\xd9", soi + 2)  # End Of Image
            if eoi < 0:
                # まだ終端が来てない
                # SOIより前のゴミは捨てる
                if soi > 0:
                    del buf[:soi]
                break

            jpg = bytes(buf[soi : eoi + 2])
            frames.append(jpg)
            del buf[: eoi + 2]

        return frames


async def ffmpeg_mjpeg_pipe(rtsp_url: str):
    """
    ffmpegでRTSP(MJPEG)を 'デコードせず' JPEGフレームの連結(mjpeg)としてstdoutへ吐くジェネレータ
    """
    # 低遅延寄りのオプション（重要）
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-rtsp_transport",
        "tcp",
        "-fflags",
        "nobuffer",
        "-flags",
        "low_delay",
        "-max_delay",
        "0",
        "-i",
        rtsp_url,
        "-an",
        "-c:v",
        "copy",  # ← MJPEGなら“そのまま”コピーできる
        "-f",
        "mjpeg",
        "pipe:1",
    ]

    print("[ffmpeg] start:", " ".join(cmd))

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    async def _log_stderr(p):
        assert p.stderr is not None
        while True:
            line = await p.stderr.readline()
            if not line:
                break
            print("[ffmpeg stderr]", line.decode("utf-8", "ignore").rstrip())

    # stderr ログをバックグラウンドで読み続ける
    asyncio.create_task(_log_stderr(proc))

    extractor = MJPEGFrameExtractor()

    try:
        while True:
            if proc.stdout is None:
                break
            chunk = await proc.stdout.read(64 * 1024)
            if not chunk:
                break
            extractor.feed(chunk)
            for jpg in extractor.pop_frames():
                yield jpg
    finally:
        # プロセス終了
        if proc.returncode is None:
            proc.send_signal(signal.SIGTERM)
            try:
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
        if proc.returncode not in (0, None):
            print(f"[ffmpeg] exited with code {proc.returncode}")


class RealTimeDetector:
    def __init__(self, cfg_path: Path, rtsp_url: Optional[str] = None):
        self.cfg_path = cfg_path
        self.rtsp_url = rtsp_url
        self.detector = self.build_detector(cfg_path)

    def build_detector(self, conf_path: Path):
        cfg = OmegaConf.load(conf_path)
        detector = instantiate(cfg)
        return detector

    def _run_inference(self, images: List[np.ndarray]) -> List[np.ndarray]:
        if not images:
            return []
        return self.detector(images)

    async def run(self) -> None:
        if not self.rtsp_url:
            raise ValueError("rtsp_url is required for RealTimeDetector.run()")
        frame_count = 0
        t_prev = time.perf_counter()
        async for jpg in ffmpeg_mjpeg_pipe(self.rtsp_url):
            t_recv = time.perf_counter()
            frame = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
            t_after_decode = time.perf_counter()
            if frame is None:
                continue
            results = self._run_inference([frame])
            t_after_inference = time.perf_counter()

            frame_count += 1
            interval_ms = (t_recv - t_prev) * 1000
            decode_ms = (t_after_decode - t_recv) * 1000
            inference_ms = (t_after_inference - t_after_decode) * 1000
            total_ms = (t_after_inference - t_recv) * 1000
            t_prev = t_recv

            n_people = len(results[0]) if results else 0
            print(
                f"[frame {frame_count}] count={n_people} | "
                f"interval={interval_ms:.0f}ms decode={decode_ms:.1f}ms inference={inference_ms:.0f}ms total={total_ms:.0f}ms"
            )


def main():

    rtsp_url = "rtsp://192.168.1.100:8554/stream"
    cfg_path = Path("acca/conf/rt_conf/p2pnet.yaml")

    detector = RealTimeDetector(cfg_path, rtsp_url=rtsp_url)
    try:
        asyncio.run(detector.run())
    except KeyboardInterrupt:
        print("\n[exit] interrupted by user")


if __name__ == "__main__":
    main()
