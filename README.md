# eval

You can build eval dependencies by running the following commands:
```bash
conda create -n eval python=3.10 -y
conda activate eval
cd verl
pip install -e .
pip install math_verify
pip install /root/cth/cth/models/flash_attn/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install vllm==0.8.2
```
