# census_run.py — 인구조사용. rescue 설정(dynamo config)과 절대 섞지 말 것.
import os, ctypes
import torch, torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard

import nvtx_cov
nvtx_cov.install()

import sys
print(f"ENV: py{sys.version.split()[0]} torch{torch.__version__} cuda{torch.version.cuda}", flush=True)

class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, 4, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, x):
        h, _ = self.attn(self.n1(x), self.n1(x), self.n1(x))
        x = x + h
        return x + self.mlp(self.n2(x))

def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    d = 512
    model = nn.Sequential(*[Block(d) for _ in range(4)]).cuda()
    for blk in model:
        fully_shard(blk)          
    fully_shard(model)

    compiled = torch.compile(model)   
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)

    for step in range(20):
        torch.cuda.nvtx.range_push(f"step_{step}")
        x = torch.randn(8, 64, d, device="cuda")
        loss = compiled(x).pow(2).mean()    
        loss.backward()
        opt.step(); opt.zero_grad()
        torch.cuda.nvtx.range_pop()
        if rank == 0:
            print(f"step {step} loss={loss.item():.6f}", flush=True)

    torch.cuda.synchronize()
    dist.barrier()                
    ctypes.CDLL(os.environ["CUPTI_TRACER_SO"]).cupti_trace_stop()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()