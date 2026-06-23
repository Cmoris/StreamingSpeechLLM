from huggingface_hub import HfApi, create_repo

repo_id = "cloudmoris/zoom"
local_folder = "/n/work6/yizhang/Moris/zoom2025/audios.tar.gz"

# 1. 创建 repo，如果已存在不会报错
create_repo(
    repo_id=repo_id,
    repo_type="dataset",   # 如果是模型就改成 "model"，Space 则是 "space"
    private=False,         # True 表示私有仓库
    exist_ok=True,
)

# 2. 上传整个文件夹
api = HfApi()
api.upload_large_folder(
    folder_path=local_folder,
    repo_id=repo_id,
    repo_type="dataset",
)
