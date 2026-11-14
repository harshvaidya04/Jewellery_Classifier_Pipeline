import yaml
from pathlib import Path


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_ecr_uri(config: dict) -> str:
    aws = config["aws"]
    ecr = config["ecr"]
    return f"{aws['account_id']}.dkr.ecr.{aws['region']}.amazonaws.com/{ecr['repo_name']}:{ecr['image_tag']}"


def get_s3_paths(config: dict) -> dict:
    bucket = config["s3"]["bucket"]
    prefix = config["s3"]["prefix"]
    return {
        "train":  f"s3://{bucket}/{prefix}/data/train",
        "val":    f"s3://{bucket}/{prefix}/data/val",
        "test":   f"s3://{bucket}/{prefix}/data/test",
        "output": f"s3://{bucket}/{prefix}/output",
    }
