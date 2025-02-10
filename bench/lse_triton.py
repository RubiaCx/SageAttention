import torch
from flash_attn.utils.benchmark import benchmark_forward
import argparse

# 分别导入两种 lse 累加精度的前向函数实现
from sageattention.triton.attn_qk_int8_per_block_lse_fp16 import forward as forward_fp16_lse
from sageattention.triton.attn_qk_int8_per_block_lsp_fp32 import forward as forward_fp32_lse
from sageattention.triton.attn_qk_int8_per_block_lsp_cast import forward as forward_lse_cast

parser = argparse.ArgumentParser(description="Benchmark LSE accumulation precision differences")
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
parser.add_argument('--num_heads', type=int, default=32, help='Number of heads')
parser.add_argument('--head_dim', type=int, default=128, help='Head dimension')
args = parser.parse_args()

batch_size = args.batch_size
num_heads = args.num_heads
head_dim = args.head_dim

print(f"Benchmarking LSE precision differences: FP16 vs FP32 accumulation")
print(f"batch_size: {batch_size}, num_heads: {num_heads}, head_dim: {head_dim}")

seq_lens = [16]
# seq_lens = [1024, 2048, 4096, 8192, 16384]

for seq_len in seq_lens:
    #! 计算理论 FLOPs： 4 * num_heads * batch_size * head_dim * seq_len^2
    flops = 4 * num_heads * batch_size * head_dim * seq_len * seq_len
    print(f"\nSequence Length: {seq_len}, Estimated FLOPs: {flops:.0f}")
    
    # 构造输入张量（Q, K 为 INT8，V 为 FP16），以及量化 scale 信息
    q = torch.randint(-100, 100, (batch_size, num_heads, seq_len, head_dim), dtype=torch.int8, device='cuda')
    k = torch.randint(-100, 100, (batch_size, num_heads, seq_len, head_dim), dtype=torch.int8, device='cuda')
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
    
    q_scale = torch.randn(batch_size, num_heads, (seq_len + 127) // 128, 1, dtype=torch.float16, device='cuda')
    k_scale = torch.randn(batch_size, num_heads, (seq_len + 63) // 64, 1, dtype=torch.float16, device='cuda')
    
    for i in range(5):
        forward_fp16_lse(q, k, v, q_scale, k_scale, output_dtype=torch.float16, return_lse=True)
        forward_fp32_lse(q, k, v, q_scale, k_scale, output_dtype=torch.float16, return_lse=True)
        forward_lse_cast(q, k, v, q_scale, k_scale, output_dtype=torch.float16, return_lse=True)
    torch.cuda.synchronize()

    # Benchmark FP16 累加 lse 版本
    _, time_fp16 = benchmark_forward(forward_fp16_lse, q, k, v, q_scale, k_scale,
                                     output_dtype=torch.float16, return_lse=True,
                                     repeats=100, verbose=False, desc='FP16 LSE')
    # Benchmark FP32 累加 lse 版本
    _, time_fp32 = benchmark_forward(forward_fp32_lse, q, k, v, q_scale, k_scale,
                                     output_dtype=torch.float16, return_lse=True,
                                     repeats=100, verbose=False, desc='FP32 LSE')
    # Benchmark LSE CAST 版本
    _, time_lse_cast = benchmark_forward(forward_lse_cast, q, k, v, q_scale, k_scale,
                                     output_dtype=torch.float16, return_lse=True,
                                     repeats=100, verbose=False, desc='LSE CAST')
    
    # 计算吞吐量（TFLOPs）
    throughput_fp16 = flops / time_fp16.mean * 1e-12
    throughput_fp32 = flops / time_fp32.mean * 1e-12
    throughput_lse_cast = flops / time_lse_cast.mean * 1e-12
    print(f"Seq len {seq_len}: FP16 LSE throughput: {throughput_fp16:.3f} TFLOPs, FP32 LSE throughput: {throughput_fp32:.3f} TFLOPs, LSE CAST throughput: {throughput_lse_cast:.3f} TFLOPs")
    
    _, lse_fp16 = forward_fp16_lse(q, k, v, q_scale, k_scale, output_dtype=torch.float16, return_lse=True)
    _, lse_fp32 = forward_fp32_lse(q, k, v, q_scale, k_scale, output_dtype=torch.float16, return_lse=True)
    _, lse_cast = forward_lse_cast(q, k, v, q_scale, k_scale, output_dtype=torch.float16, return_lse=True)
    # 打印一部分 lse 结果
    print(lse_fp16.shape)

    print(lse_fp16)
    print(lse_fp32)
    print(lse_cast)
    # 将两者转换为 float 进行比较
    diff = (lse_fp16.float() - lse_fp32.float()).abs().mean().item()
    print(f"Average absolute difference between LSE outputs: {diff:.6f}") 
    diff = (lse_fp32.float() - lse_cast.float()).abs().mean().item()
    print(f"Average absolute difference between LSE outputs: {diff:.6f}") 