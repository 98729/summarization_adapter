from huggingface_hub import HfApi

def upload(name: str):
    repo_name = f"cui54/{name}"
    local_path = f"weights/{name}"
    
    api = HfApi()
    api.create_repo(repo_id=repo_name, private=True)  # Set to False if you want it public

    api.upload_folder(
        folder_path=local_path,
        path_in_repo="",
        repo_id=repo_name
    )

    print(f"Model uploaded to https://huggingface.co/{repo_name}")

upload("web_sim_traj_shoppingmapmix2")
