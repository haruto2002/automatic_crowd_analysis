import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

HOST = "member"
IP = "192.168.0.10"
PORT = 554
PW = "AIST-rwdc"
RTSP_URL = f"rtsp://{HOST}:{PW}@{IP}:{PORT}/ONVIF/MediaInput?profile=def_profile1"
W, H = 1920, 1080  # ★ストリームの解像度に合わせる
FPS = 15  # 任意（表示用/目安）


class RealTimeDetector:
    def __init__(self, cfg_path: Path):
        self.cfg_path = cfg_path
        self.detector = self.build_detector(cfg_path)

    def build_detector(self, conf_path: Path):
        cfg = OmegaConf.load(conf_path)
        detector = instantiate(cfg)
        return detector

    def run(self, images: List[np.ndarray]) -> List[np.ndarray]:
        results = self.detector(images)
        return results

    def display_results(
        self,
        img: np.ndarray,
        detection: np.ndarray,
        point_size: int = 5,
        color: tuple[int, int, int] = (0, 0, 255),
    ) -> np.ndarray:
        for x, y, _ in detection:
            cv2.circle(img, (int(x), int(y)), point_size, color, -1)
        return img


@dataclass
class ReaderStats:
    # reader側（FFmpegから読めたフレーム）
    read_frames: int = 0
    read_fail: int = 0  # フレームが読めない/不足
    restarts: int = 0

    # 表示側（latest上書きで飛ばした分）
    shown_frames: int = 0
    skipped_frames: int = 0

    # タイムスタンプ
    last_read_ts: float = 0.0
    last_shown_ts: float = 0.0


class FFmpegRTSPReader:
    """
    FFmpegでRTSPを受信し、rawvideo(bgr24)をstdoutに流してPythonで読む。
    - latest(最新フレーム1枚)のみ保持 → 遅延を溜めにくい
    - 自動再接続つき
    """

    def __init__(
        self,
        rtsp_url: str,
        size: Tuple[int, int],
        transport: str = "udp",  # "udp" or "tcp"
        ffmpeg_path: str = "ffmpeg",
        log_every_sec: float = 1.0,  # ログ間隔
        reconnect_backoff_sec: float = 0.5,  # 再接続の待ち（短すぎると暴れる）
    ):
        self.rtsp_url = rtsp_url
        self.w, self.h = size
        self.transport = transport
        self.ffmpeg_path = ffmpeg_path

        self.frame_bytes = self.w * self.h * 3  # bgr24

        self.log_every_sec = log_every_sec
        self.reconnect_backoff_sec = reconnect_backoff_sec

        self._lock = threading.Lock()
        self._stop = threading.Event()

        self._proc: Optional[subprocess.Popen] = None

        # latest共有（最新フレームのみ）
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_ts: float = 0.0
        self._latest_seq: int = 0  # readerが読んだ順番号

        self.stats = ReaderStats()

        self._thread: Optional[threading.Thread] = None

    # ---------- public API ----------

    def start(self) -> None:
        """readerスレッド開始"""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """停止（スレッド停止＋FFmpeg終了）"""
        self._stop.set()
        self._terminate_ffmpeg()

    def get_latest(self) -> Tuple[Optional[np.ndarray], int, float]:
        """
        最新フレームを取得（copyして返す）
        returns: (frame or None, seq, ts)
        """
        with self._lock:
            if self._latest_frame is None:
                return None, self._latest_seq, self._latest_ts
            return self._latest_frame.copy(), self._latest_seq, self._latest_ts

    # ---------- internal ----------

    def _build_ffmpeg_cmd(self) -> list:
        # 低遅延寄りの定番セット（環境で効き方は変わります）
        return [
            self.ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-rtsp_transport",
            self.transport,
            # 解析/バッファ抑制（攻め設定。安定しないならprobesizeを増やす）
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-probesize",
            "32",
            "-analyzeduration",
            "0",
            "-i",
            self.rtsp_url,
            "-an",  # 音声不要
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            # rawvideoはヘッダ無しなので、解像度固定が安全
            "-vf",
            f"scale={self.w}:{self.h}",
            "pipe:1",
        ]

    def _spawn_ffmpeg(self) -> None:
        self._terminate_ffmpeg()

        cmd = self._build_ffmpeg_cmd()
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,  # エラー確認用（loglevel errorで詰まりにくくする）
            bufsize=10**8,
        )
        self.stats.restarts += 1

    def _terminate_ffmpeg(self) -> None:
        proc = self._proc
        self._proc = None
        if not proc:
            return
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _reader_loop(self) -> None:
        next_log = time.time() + self.log_every_sec

        while not self._stop.is_set():
            # FFmpeg起動（未起動なら起動）
            if self._proc is None:
                self._spawn_ffmpeg()
                time.sleep(0.05)

            proc = self._proc
            if proc is None:
                continue

            # FFmpegが落ちたら再起動
            if proc.poll() is not None:
                self._print_ffmpeg_error(proc)
                self._spawn_ffmpeg()
                time.sleep(self.reconnect_backoff_sec)
                continue

            # 1フレーム分読む（ブロッキング）
            try:
                raw = proc.stdout.read(self.frame_bytes) if proc.stdout else b""
            except Exception:
                raw = b""

            # 読めない/不足 → 再起動（切断やパイプ破断など）
            if not raw or len(raw) < self.frame_bytes:
                self.stats.read_fail += 1
                # 少し待ってから再起動
                self._spawn_ffmpeg()
                time.sleep(self.reconnect_backoff_sec)
                continue

            # bytes → numpy画像
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.h, self.w, 3))
            now = time.perf_counter()

            # latest更新（上書き）
            with self._lock:
                self._latest_frame = frame
                self._latest_ts = now
                self._latest_seq += 1

                self.stats.read_frames += 1
                self.stats.last_read_ts = now

            # 定期ログ
            if self.log_every_sec > 0 and now >= next_log:
                next_log = now + self.log_every_sec
                self._log_status()

        # stop時
        self._terminate_ffmpeg()

    def _print_ffmpeg_error(self, proc: subprocess.Popen) -> None:
        try:
            if proc.stderr:
                err = proc.stderr.read().decode("utf-8", errors="ignore").strip()
                if err:
                    print("[FFMPEG-ERR]", err)
        except Exception:
            pass

    def _log_status(self) -> None:
        s = self.stats
        # メイン側が更新している skipped/shown も含めたいので、そのまま表示
        print(
            f"[STAT] read={s.read_frames} shown={s.shown_frames} "
            f"skipped={s.skipped_frames} read_fail={s.read_fail} restarts={s.restarts} "
            f"last_read_age={(time.perf_counter() - s.last_read_ts) * 1000:.1f}ms"
        )


def main():
    cfg_path = Path("acca/conf/rt_conf/p2pnet.yaml")
    detector = RealTimeDetector(cfg_path)
    print("Detector setting DONE")

    reader = FFmpegRTSPReader(
        rtsp_url=RTSP_URL,
        size=(W, H),
        transport="udp",  # udpがダメなら "tcp"
        log_every_sec=1.0,
        reconnect_backoff_sec=0.5,
    )
    reader.start()

    # GUIウォームアップ
    cv2.imshow("frame", np.zeros((H, W, 3), np.uint8))
    cv2.waitKey(1)

    last_shown_seq = 0
    age_ms_txt = "age= --- "

    try:
        while True:
            frame, seq, ts = reader.get_latest()
            if frame is None:
                time.sleep(0.01)
                continue

            # 表示側スキップ検出（latest上書きで飛んだ分）
            if last_shown_seq != 0:
                skipped = seq - last_shown_seq - 1
                if skipped > 0:
                    reader.stats.skipped_frames += skipped
                    age_ms = (time.perf_counter() - ts) * 1000.0
                    print(
                        f"[SKIP] skipped={skipped} total={reader.stats.skipped_frames} "
                        f"shown_seq={seq} prev_seq={last_shown_seq} age={age_ms:.1f}ms"
                    )
            last_shown_seq = seq

            # 表示統計
            reader.stats.shown_frames += 1
            reader.stats.last_shown_ts = time.perf_counter()

            # フレーム鮮度（FFmpegが返した時刻から表示まで）
            if seq % 10 == 0:
                age_ms = (time.perf_counter() - ts) * 1000.0
                age_ms_txt = f"age={age_ms:.1f}ms"
            cv2.putText(
                frame,
                age_ms_txt,
                (1500, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 255, 0),
                2,
            )

            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        reader.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
