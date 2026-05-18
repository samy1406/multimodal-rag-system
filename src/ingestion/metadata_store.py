import json

def save_metadata(metadata: dict, path: str) -> None:
    with open(path, 'w') as f:
        json.dump(metadata, f)
    return None


def load_metadata(path: str) -> dict:
    with open(path, 'r') as file:
        data = json.load(file)
    
    data = {int(k): v for k,v in data.items()}
    return data