import json
from pathlib import Path


def extract_utterances(jsonl_file):
    """
    从你的jsonl格式提取utterance

    返回:
    [
        {
            "speaker": "A",
            "audio": "...",
            "start": 3.848,
            "end": 7.280,
            "text": "こんばんは"
        },
        ...
    ]
    """

    utterances = []

    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            sample = json.loads(line)

            audio_info = sample[0]["content"][0]
            text_info = sample[1]["content"][0]

            audio_path = audio_info["audio"]

            if "A_gd" in audio_path:
                speaker = "A"
            elif "B_gd" in audio_path:
                speaker = "B"
            else:
                speaker = "UNK"

            utterances.append(
                {
                    "speaker": speaker,
                    "audio": audio_path,
                    "start": float(audio_info["start"]),
                    "end": float(audio_info["end"]),
                    "text": text_info["text"],
                }
            )

    utterances.sort(key=lambda x: x["start"])

    return utterances


def overlap(seg1, seg2):
    """
    判断两个区间是否重叠
    """

    return min(seg1["end"], seg2["end"]) > max(
        seg1["start"], seg2["start"]
    )


def build_overlap_clusters(utterances):
    """
    提取连续 overlap cluster

    返回:
    [
        [
            utt1,
            utt2,
            ...
        ],
        [
            utt10,
            utt11,
            ...
        ]
    ]
    """

    if len(utterances) == 0:
        return []

    clusters = []

    current_cluster = [utterances[0]]

    # 当前cluster覆盖到的最远时间
    cluster_end = utterances[0]["end"]

    for utt in utterances[1:]:

        # 与当前cluster有重叠
        if utt["start"] < cluster_end:

            current_cluster.append(utt)

            cluster_end = max(cluster_end, utt["end"])

        else:

            if len(current_cluster) > 1:
                clusters.append(current_cluster)

            current_cluster = [utt]
            cluster_end = utt["end"]

    if len(current_cluster) > 1:
        clusters.append(current_cluster)

    return clusters


def cluster_info(cluster):
    """
    统计cluster信息
    """

    start = min(x["start"] for x in cluster)
    end = max(x["end"] for x in cluster)

    return {
        "start": start,
        "end": end,
        "duration": end - start,
        "num_utts": len(cluster),
        "speakers": sorted(set(x["speaker"] for x in cluster)),
    }


def save_clusters(clusters, output_file):

    with open(output_file, "w", encoding="utf-8") as f:

        for cluster in clusters:

            obj = [{
                "cluster_info": cluster_info(cluster),
                "utterances": cluster,
            }]

            f.write(
                json.dumps(
                    obj,
                    ensure_ascii=False,
                )
                + "\n"
            )

def main():
    conv_jsonl_dir = "/n/work6/yizhang/Moris/zoom2025/psuedo_conv_labels"
    output_dir = "/n/work6/yizhang/Moris/zoom2025/psuedo_overlap_labels"

    # Process all JSONL files in the conversation directory
    for jsonl_file in Path(conv_jsonl_dir).glob("*.jsonl"):
        utterances = extract_utterances(jsonl_file)

        print(f"total utterances: {len(utterances)}")

        clusters = build_overlap_clusters(utterances)

        print(f"overlap clusters: {len(clusters)}")

        save_clusters(
            clusters,
            Path(output_dir) / f"{Path(jsonl_file).stem}_overlap.jsonl",
        )

if __name__ == "__main__":
    main()