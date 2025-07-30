Run DDP version by order:
```
time torchrun --nproc_per_node=4 ./VGG_DDP.py 2>&1 | tee output.log
```