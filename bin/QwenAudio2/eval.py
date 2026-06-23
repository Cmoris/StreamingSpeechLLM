import re
import json
from typing import List, Dict

def compute_cer(ref, hyp):
    import numpy as np
    ref, hyp = list(ref), list(hyp)
    dp = np.zeros((len(ref)+1, len(hyp)+1), dtype=int)
    for i in range(len(ref)+1): dp[i][0] = i
    for j in range(len(hyp)+1): dp[0][j] = j
    for i in range(1, len(ref)+1):
        for j in range(1, len(hyp)+1):
            dp[i][j] = dp[i-1][j-1] if ref[i-1]==hyp[j-1] else min(
                dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+1)
    return dp[len(ref)][len(hyp)] / max(1, len(ref))


# import re

def clean_special_tokens(text: str) -> str:
    """
    只保留:
    - 文本
    - <speaker_A> </speaker_A>
    - <speaker_B> </speaker_B>

    删除:
    - 其它所有特殊 token
    - system/user/assistant/time/audio 等 prompt
    """

    # 删除 prompt 内容
    text = re.sub(
        r'(system|user|assistant).*?(?=<speaker_[AB]>|$)',
        '',
        text,
        flags=re.DOTALL
    )

    # 删除除 speaker 外的所有 tag
    text = re.sub(
        r'</?(?!speaker_A\b|speaker_B\b)[^>]+>',
        '',
        text
    )

    # 删除 ...
    text = text.replace("...", "")

    # 清理空白
    text = re.sub(r'\s+', '', text)

    return text.strip()



def split_by_time(text: str) -> List[Dict]:
    """
    按照 Time=x.x-y.ys 切分数据。

    返回格式:
    [
        {
            "time": "0.0-1.0",
            "start": 0.0,
            "end": 1.0,
            "text": "<speaker_A>こんにちは</speaker_A>"
        },
        ...
    ]
    """

    pattern = re.compile(
        r'Time=(\d+\.\d+)-(\d+\.\d+)s.*?assistant\s*(.*?)(?=\nuser\nTime=|\Z)',
        re.DOTALL
    )

    results = []

    for match in pattern.finditer(text):
        start = float(match.group(1))
        end = float(match.group(2))
        content = match.group(3)

        # # 保留 speaker token，去掉其它特殊 token
        # content = re.sub(
        #     r'</?(?!speaker_A\b|speaker_B\b)[^>]+>',
        #     '',
        #     content
        # )

        # # 去掉 ...
        # content = content.replace("...", "")

        # 清理空白
        content = re.sub(r'\s+', '', content)

        results.append({
            "time": f"{start}-{end}",
            "start": start,
            "end": end,
            "text": content
        })

    return results

def asr_sub(text):
    mapping = {
    "<te>": "<speaker_B>",
    "<ts>": "</speaker_B>"
}

    text = re.sub(
        r'<te>|<ts>',
        lambda m: mapping[m.group()],
        text
    )
    
    return text

def read_jsonl(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        yield json.loads(line)
        
def main():
    gd_file = "/home/yizhang/Moris/StreamingSpeechLLM/bin/QwenAudio2/data/l3_conv_test_with_backchannel_groundtruth_0.5s.jsonl"
    inf_file = "/home/yizhang/Moris/StreamingSpeechLLM/bin/QwenAudio2/result_asr/qwen2audio_asrl2_chunk2s/Chunk2.0_0_results.jsonl"
    all_cer = []
    gd_text = []
    inf_text = []
    for gd in read_jsonl(gd_file):
        text = gd["refs"][0]
        gd_text.append([clean_special_tokens(g["text"]) for g in split_by_time(text)])
        
    for result in read_jsonl(inf_file):
        text = [clean_special_tokens(asr_sub(t[2])) for t in result] 
        inf_text.append(text)
        
    for inf, gd in zip(inf_text, gd_text):
        inf = "".join(inf)
        gd = "".join(gd)
        cer = compute_cer(ref=gd, hyp=inf)
        all_cer.append(cer)
    print(sum(all_cer)/len(all_cer))

if __name__ == "__main__":
    main()