# VLMEmbed
## Set up env
```bash
uv sync
```

## Download dataset

```bash
source .venv/bin/activate
bash download.sh
bash prepare.sh
```

## Run 
```bash
CUDA_VISIBLE_DEVICES=0 bash project_commands_0.sh
```
And
```
CUDA_VISIBLE_DEVICES=1 bash project_commands_1.sh
```
