from pathlib import Path
import re
import subprocess
from typing import List, Optional


AUDIO_PATTERN = re.compile(r"_([0-9]+)\.mp4$")
VIDEO_RES_PATTERN = re.compile(r"video_(\d+)x(\d+)_.*\.mp4$")
BV_PATTERN = re.compile(r"\[(BV[0-9A-Za-z]{10})\]")

def extract_bv_id_from_dir(dir_path: str) -> str | None:
    """
    从目录名中提取 BV 号，例如：
    '[BV1ZB421B7aA]纪录片xxx' -> 'BV1ZB421B7aA'
    """
    m = BV_PATTERN.search(dir_path)
    if not m:
        return None
    return m.group(1)


def parse_audio_bitrate(p: Path) -> int:
    """
    从 audio 文件名中解析码率
    audio_mp4a.40.2_67141.mp4 -> 67141
    """
    m = AUDIO_PATTERN.search(p.name)
    if not m:
        return -1
    return int(m.group(1))


def parse_video_resolution(p: Path) -> int:
    """
    从 video 文件名解析分辨率，用 像素总数 作为排序依据
    """
    m = VIDEO_RES_PATTERN.match(p.name)
    if not m:
        return -1
    w, h = map(int, m.groups())
    return w * h

def video_has_audio(video_path: Path) -> bool:
    """
    判断 video 是否包含音频流
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=index",
        "-of", "csv=p=0",
        str(video_path)
    ]
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return bool(result.stdout.strip())

def extract_audio_from_video(video_path: Path) -> Path | None:
    """
    从 video 提取音频
    - 若无音频流，直接返回 None
    """
    if not video_has_audio(video_path):
        print(f"[WARN] No audio stream in: {video_path.name}")
        return None

    out_audio = video_path.with_name("audio_extracted.m4a")

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "copy",
        str(out_audio)
    ]

    subprocess.run(cmd, check=True)
    return out_audio


def get_best_audio_in_dir(dir_path: Path) -> Optional[Path]:
    """
    返回该目录下最优音频路径：
    - 有 audio：返回码率最大的
    - 无 audio：从 video 提取
    """
    audio_files = list(dir_path.glob("audio*.mp4"))

    if audio_files:
        best_audio = max(audio_files, key=parse_audio_bitrate)
        return best_audio

    video_files = list(dir_path.glob("video*.mp4"))
    if not video_files:
        return None

    best_video = max(video_files, key=parse_video_resolution)
    return extract_audio_from_video(best_video)


def collect_all_best_audios(root_dir: Path) -> List[Path]:
    """
    遍历 root_dir 下的所有子目录，收集最优音频
    """
    results = []

    for d in root_dir.iterdir():
        if not d.is_dir():
            continue

        audio = get_best_audio_in_dir(d)
        if audio:
            results.append(audio)

    return results

if __name__ == "__main__":
    root = Path("/mnt/nvme/workspaces/liuyutong/bilibili-downloader/downloads")

    best_audios = collect_all_best_audios(root)

    for a in best_audios:
        print(a)