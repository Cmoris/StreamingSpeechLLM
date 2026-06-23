import os, torch, json, tqdm

root = '/n/work6/yizhang/Moris/zoom2025/finetune_labels/train'
files = os.listdir(root)

for file in tqdm.tqdm(files):
    path = os.path.join(root, file)
    if not path.endswith('.jsonl'):
        continue
    lines = open(path).readlines()
    with open(path, 'a') as f:
        seeks = [0] + torch.tensor([len(l) for l in lines]).cumsum(dim=-1)[:-1].tolist()
        f.write(json.dumps(seeks))
