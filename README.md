## Stable Diffusion optimization
```shell
conda create -n sd-1.5 python=3.8
conda activate sd-1.5
pip install -r requirement.txt
numactl -C 0-7 -l python example.py
```