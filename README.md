# sglang-play
Playing with SGLANG on H100 NVL GPUs

## Installation
1. `uv venv`
2. `uv pip install --upgrade pip`
3. `uv pip install --prerelease=allow sglang`
4. `uv pip install --force-reinstall  torch==2.11.0 torchaudio==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu129`
5. `uv pip install --force-reinstall sglang-kernel --index-url https://docs.sglang.ai/whl/cu129/`
6. `uv pip install --force-reinstall sgl-deep-gemm --index-url https://docs.sglang.ai/whl/cu129/ --no-deps`


## JIT host compiler fix (no sudo)

The server crashes on the first request with
`fatal error: version: No such file or directory` while ninja/nvcc build
`sgl_kernel_jit_fused_rope_*`.

**Root cause:** sglang >= 0.5 compiles some kernels (e.g. fused RoPE) at
runtime with `-std=c++20` via tvm-ffi, which invokes `nvcc` without `-ccbin`.
nvcc therefore uses the system `gcc` as host compiler — GCC 8.5 on this
RHEL 8 machine — which supports neither C++20 (nvcc needs host GCC >= 10)
nor the `<version>` header (libstdc++ >= GCC 9). The `Clang 21.1.4` printed
by `check_environment.py` is only the compiler that built uv's CPython
binary, not a compiler on this machine. No prebuilt wheel can bypass this:
the RoPE kernel is JIT-only on CUDA (verified in sglang 0.5.10.post1 and
0.5.14) and the tvm-ffi cache is populated only by compiling locally.

**Fix** — install a user-space GCC 13 (conda-forge, what sglang tests
against) and point the JIT at it via `CXX` and nvcc's `NVCC_PREPEND_FLAGS`:

1. `bash scripts/setup_toolchain.sh` — one time; installs micromamba + GCC 13
   into `./.toolchain` (~1.5 GB) and self-tests a C++20 `<version>` compile.
2. `rm -rf ~/.cache/tvm-ffi` — one time; clears the failed build cache.
3. `source scripts/toolchain_env.sh` — in every shell that runs sglang
   (`scripts/demo_run.sh` now does this automatically). Sets `CC`, `CXX`,
   `NVCC_PREPEND_FLAGS="-ccbin .../g++"` and `LD_LIBRARY_PATH` (the built
   `.so` needs the matching newer `libstdc++.so.6` at runtime).
4. `uv run python3 src/jit_smoke_test.py` — compiles the exact kernel that
   crashed the server; once it prints OK, `bash scripts/demo_run.sh` works
   and the result is cached in `~/.cache/tvm-ffi`.

Also note: `src/offline_demo.py` additionally needed an
`if __name__ == "__main__":` guard — sglang.Engine spawns subprocesses with
the multiprocessing `spawn` method, which re-imports the main module.


### check_environment.py output

```
(sglang-play) bash-4.4$ uv run python3 src/check_environment.py 
Python: 3.11.14 (main, Dec 17 2025, 21:07:37) [Clang 21.1.4 ]
CUDA available: True
GPU 0,1,2,3,4,5,6,7: NVIDIA H100 NVL
GPU 0,1,2,3,4,5,6,7 Compute Capability: 9.0
CUDA_HOME: /home/whan/cuda-12.8
NVCC: Cuda compilation tools, release 12.8, V12.8.61
CUDA Driver Version: 570.158.01
PyTorch: 2.11.0+cu129
sglang: 0.5.10.post1
sglang-kernel: 0.4.4+cu129
flashinfer_python: 0.6.7.post3
flashinfer_cubin: 0.6.7.post3
flashinfer_jit_cache: Module Not Found
triton: 3.6.0
transformers: 5.3.0
torchao: 0.9.0
numpy: 2.4.4
aiohttp: 3.14.1
fastapi: 0.139.0
huggingface_hub: 1.22.0
interegular: 0.3.3
modelscope: 1.38.1
orjson: 3.11.9
outlines: 0.1.11
packaging: 26.2
psutil: 7.2.2
pydantic: 2.14.0a1
python-multipart: 0.0.32
pyzmq: 27.1.0
uvicorn: 0.51.0
uvloop: 0.22.1
vllm: Module Not Found
xgrammar: 0.1.32
openai: 2.6.1
tiktoken: 0.13.0
anthropic: 0.116.0
litellm: Module Not Found
torchcodec: 0.9.1
NVIDIA Topology: 
        GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7    NIC0    NIC1    CPU Affinity     NUMA Affinity   GPU NUMA ID
GPU0     X      NV12    NODE    NODE    SYS     SYS     SYS     SYS     NODE    NODE    0-63,128-191     0               N/A
GPU1    NV12     X      NODE    NODE    SYS     SYS     SYS     SYS     PHB     PHB     0-63,128-191     0               N/A
GPU2    NODE    NODE     X      NV12    SYS     SYS     SYS     SYS     NODE    NODE    0-63,128-191     0               N/A
GPU3    NODE    NODE    NV12     X      SYS     SYS     SYS     SYS     NODE    NODE    0-63,128-191     0               N/A
GPU4    SYS     SYS     SYS     SYS      X      NV12    NODE    NODE    SYS     SYS     64-127,192-255   1               N/A
GPU5    SYS     SYS     SYS     SYS     NV12     X      NODE    NODE    SYS     SYS     64-127,192-255   1               N/A
GPU6    SYS     SYS     SYS     SYS     NODE    NODE     X      NV12    SYS     SYS     64-127,192-255   1               N/A
GPU7    SYS     SYS     SYS     SYS     NODE    NODE    NV12     X      SYS     SYS     64-127,192-255   1               N/A
NIC0    NODE    PHB     NODE    NODE    SYS     SYS     SYS     SYS      X      PIX
NIC1    NODE    PHB     NODE    NODE    SYS     SYS     SYS     SYS     PIX      X 

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges withina NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks

NIC Legend:

  NIC0: mlx5_0
  NIC1: mlx5_1


ulimit soft: 8192
(sglang-play) bash-4.4$
```

### cli checks

```
(sglang-play) bash-4.4$ python3 -c "import sysconfig; print('CC:', sysconfig.get_config_var('CC')); print('CXX:', sysconfig.get_config_var('CXX'))"
CC: cc -pthread
CXX: c++ -pthread
(sglang-play) bash-4.4$ echo "CXX Override: $CXX"
CXX Override: 
(sglang-play) bash-4.4$ echo "CC Override: $CC"
CC Override: 
```


### Current status

1. After running `bash scripts/demo_run.sh`

```
(sglang-play) bash-4.4$ bash scripts/demo_run.sh 
/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/srt/entrypoints/http_server.py:172: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
  from sglang.srt.utils.json_response import (
[2026-07-08 16:06:36] Fail to set RLIMIT_NOFILE: current limit exceeds maximum limit
[2026-07-08 16:06:36] server_args=ServerArgs(model_path='qwen/qwen2.5-0.5b-instruct', tokenizer_path='qwen/qwen2.5-0.5b-instruct', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='0.0.0.0', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, ssl_keyfile_password=None, enable_ssl_refresh=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.86, max_running_requests=None, max_queued_requests=None, max_total_tokens=None, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, disable_priority_preemption=False, default_priority_value=None, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, stream_response_default_include_usage=False, incremental_streaming_output=False, enable_streaming_session=False, random_seed=726810559, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=None, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, use_ray=False, custom_sigquit_handler=None, log_level='info', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_mfu_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='qwen/qwen2.5-0.5b-instruct', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser=None, tool_call_parser=None, tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, experts_shared_outer_loras=None, attention_backend='triton', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='pytorch', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='auto', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, mamba_backend='triton', speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_max_trie_depth=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, enforce_disable_flashinfer_allreduce_fusion=False, enable_aiter_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, enable_elastic_expert_backup=False, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, linear_attn_backend='triton', linear_attn_decode_backend=None, linear_attn_prefill_backend=None, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, enable_hisparse=False, hisparse_config=None, enable_lmcache=False, kt_weight_path=None, kt_method='AMXINT4', kt_cpuinfer=None, kt_threadpool_count=2, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=256, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], disable_cuda_graph=True, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, pre_warm_nccl=False, disable_overlap_schedule=True, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, disable_piecewise_cuda_graph=True, enforce_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, gc_threshold=None, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_fused_moe_sum_all_reduce=False, enable_prefill_context_parallel=False, prefill_cp_mode='in-seq-split', enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], enable_adaptive_dispatch_to_encoder=False, custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, engine_info_bootstrap_port=6789, modelexpress_config=None, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, enable_mm_global_cache=False, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)
[2026-07-08 16:06:37] Using default HuggingFace chat template with detected content format: string
[2026-07-08 16:06:47] Init torch distributed begin.
[2026-07-08 16:06:48] Init torch distributed ends. elapsed=0.84 s, mem usage=0.09 GB
[2026-07-08 16:06:48] Load weight begin. avail mem=36.54 GB
[2026-07-08 16:06:48] Found local HF snapshot for qwen/qwen2.5-0.5b-instruct at /p01/whan/.huggingface/models--qwen--qwen2.5-0.5b-instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775; skipping download.
[2026-07-08 16:06:48] No model.safetensors.index.json found in remote.
Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.73it/s]
[2026-07-08 16:06:49] Load weight end. elapsed=0.41 s, type=Qwen2ForCausalLM, avail mem=35.57 GB, mem usage=0.98 GB.
[2026-07-08 16:06:49] Using KV cache dtype: torch.bfloat16
[2026-07-08 16:06:49] KV Cache is allocated. #tokens: 2658165, K size: 15.21 GB, V size: 15.21 GB
[2026-07-08 16:06:49] Memory pool end. avail mem=4.55 GB
[2026-07-08 16:06:49] Disable piecewise CUDA graph because --disable-piecewise-cuda-graph is set
[2026-07-08 16:06:50] max_total_num_tokens=2658165, chunked_prefill_size=8192, max_prefill_tokens=16384, max_running_requests=4096, context_len=32768, available_gpu_mem=4.46 GB
[2026-07-08 16:06:50] INFO:     Started server process [3636759]
[2026-07-08 16:06:50] INFO:     Waiting for application startup.
[2026-07-08 16:06:50] Using default chat sampling params from model generation config: {'repetition_penalty': 1.1, 'temperature': 0.7, 'top_k': 20, 'top_p': 0.8}
[2026-07-08 16:06:50] INFO:     Application startup complete.
[2026-07-08 16:06:50] INFO:     Uvicorn running on http://0.0.0.0:30000 (Press CTRL+C to quit)
```

Then when I do `uv run python3 src/online_demo.py` I get 

```
(sglang-play) bash-4.4$ bash scripts/demo_run.sh 
/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/srt/entrypoints/http_server.py:172: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directlyto JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
  from sglang.srt.utils.json_response import (
[2026-07-08 16:06:36] Fail to set RLIMIT_NOFILE: current limit exceeds maximum limit
[2026-07-08 16:06:36] server_args=ServerArgs(model_path='qwen/qwen2.5-0.5b-instruct', tokenizer_path='qwen/qwen2.5-0.5b-instruct', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='0.0.0.0', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, ssl_keyfile_password=None, enable_ssl_refresh=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None,modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.86, max_running_requests=None, max_queued_requests=None, max_total_tokens=None, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, disable_priority_preemption=False, default_priority_value=None, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None,device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, stream_response_default_include_usage=False, incremental_streaming_output=False, enable_streaming_session=False, random_seed=726810559, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=None, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, use_ray=False, custom_sigquit_handler=None, log_level='info', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_mfu_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='qwen/qwen2.5-0.5b-instruct', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser=None, tool_call_parser=None, tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, experts_shared_outer_loras=None, attention_backend='triton', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='pytorch', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='auto', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, mamba_backend='triton', speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_max_trie_depth=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, enforce_disable_flashinfer_allreduce_fusion=False, enable_aiter_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, enable_elastic_expert_backup=False, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, linear_attn_backend='triton', linear_attn_decode_backend=None, linear_attn_prefill_backend=None, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, enable_hisparse=False, hisparse_config=None, enable_lmcache=False, kt_weight_path=None, kt_method='AMXINT4', kt_cpuinfer=None, kt_threadpool_count=2, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=256, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], disable_cuda_graph=True, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False,disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, pre_warm_nccl=False, disable_overlap_schedule=True, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, disable_piecewise_cuda_graph=True, enforce_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, gc_threshold=None,enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_fused_moe_sum_all_reduce=False, enable_prefill_context_parallel=False, prefill_cp_mode='in-seq-split', enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], enable_adaptive_dispatch_to_encoder=False, custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, engine_info_bootstrap_port=6789, modelexpress_config=None, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, enable_mm_global_cache=False, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)
[2026-07-08 16:06:37] Using default HuggingFace chat template with detected content format: string
[2026-07-08 16:06:47] Init torch distributed begin.
[2026-07-08 16:06:48] Init torch distributed ends. elapsed=0.84 s, mem usage=0.09 GB
[2026-07-08 16:06:48] Load weight begin. avail mem=36.54 GB
[2026-07-08 16:06:48] Found local HF snapshot for qwen/qwen2.5-0.5b-instruct at /p01/whan/.huggingface/models--qwen--qwen2.5-0.5b-instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775; skipping download.
[2026-07-08 16:06:48] No model.safetensors.index.json found in remote.
Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.73it/s]
[2026-07-08 16:06:49] Load weight end. elapsed=0.41 s, type=Qwen2ForCausalLM, avail mem=35.57 GB, mem usage=0.98 GB.
[2026-07-08 16:06:49] Using KV cache dtype: torch.bfloat16
[2026-07-08 16:06:49] KV Cache is allocated. #tokens: 2658165, K size: 15.21 GB, V size: 15.21 GB
[2026-07-08 16:06:49] Memory pool end. avail mem=4.55 GB
[2026-07-08 16:06:49] Disable piecewise CUDA graph because --disable-piecewise-cuda-graph is set
[2026-07-08 16:06:50] max_total_num_tokens=2658165, chunked_prefill_size=8192, max_prefill_tokens=16384, max_running_requests=4096, context_len=32768, available_gpu_mem=4.46 GB
[2026-07-08 16:06:50] INFO:     Started server process [3636759]
[2026-07-08 16:06:50] INFO:     Waiting for application startup.
[2026-07-08 16:06:50] Using default chat sampling params from model generation config: {'repetition_penalty': 1.1, 'temperature': 0.7, 'top_k': 20, 'top_p': 0.8}
[2026-07-08 16:06:50] INFO:     Application startup complete.
[2026-07-08 16:06:50] INFO:     Uvicorn running on http://0.0.0.0:30000 (Press CTRL+C to quit)
[2026-07-08 16:07:38] Scheduler hit an exception: Traceback (most recent call last):
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/srt/managers/scheduler.py", line 3616, in run_scheduler_process
    scheduler.run_event_loop()
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/srt/managers/scheduler.py", line 1300, in run_event_loop
    dispatch_event_loop(self)
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/srt/managers/scheduler.py", line 3499, in dispatch_event_loop
    scheduler.event_loop_normal()
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/torch/utils/_contextlib.py", line 124, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/srt/managers/scheduler.py", line 1319, in event_loop_normal
    result = self.run_batch(batch)
             ^^^^^^^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/srt/managers/scheduler.py", line 2724, in run_batch
    batch_result = self.model_worker.forward_batch_generation(
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/srt/managers/tp_worker.py", line 469, in forward_batch_generation
    out = self.model_runner.forward(
          ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/srt/model_executor/model_runner.py", line 2739, in forward
    output = self._forward_raw(
             ^^^^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/srt/model_executor/model_runner.py", line 2849, in _forward_raw
    ret, can_run_graph = self.forward_extend(
                         ^^^^^^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/srt/model_executor/model_runner.py", line 2676, in forward_extend
    self.model.forward(
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/torch/utils/_contextlib.py", line 124, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/srt/models/qwen2.py", line 482, in forward
    hidden_states = self.model(
                    ^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1790, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/srt/models/qwen2.py", line 363, in forward
    hidden_states, residual = layer(
                              ^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1790, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/srt/models/qwen2.py", line 249, in forward
    hidden_states = self.self_attn(
                    ^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1790, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/srt/models/qwen2.py", line 188, in forward
    q, k = self.rotary_emb(positions, q, k)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1790, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/srt/layers/utils/multi_platform.py", line 73, in forward
    return self._forward_method(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/srt/layers/rotary_embedding/base.py", line 340, in forward_cuda
    apply_rope_with_cos_sin_cache_inplace(
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/jit_kernel/rope.py", line 219, in apply_rope_with_cos_sin_cache_inplace
    apply_rope_inplace(
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/torch/_ops.py", line 1269, in __call__
    return self._op(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/jit_kernel/rope.py", line 136, in apply_rope_inplace
    module = _jit_fused_rope_module(is_neox, rope_dim, q.dtype)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/jit_kernel/utils.py", line 57, in wrapper
    result_map[key] = fn(*args, **kwargs)
                      ^^^^^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/jit_kernel/rope.py", line 32, in _jit_fused_rope_module
    return load_jit(
           ^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/jit_kernel/utils.py", line 208, in load_jit
    return load_inline(
           ^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/tvm_ffi/cpp/extension.py", line1132, in load_inline
    build_inline(
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/tvm_ffi/cpp/extension.py", line974, in build_inline
    return _build_impl(
           ^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/tvm_ffi/cpp/extension.py", line738, in _build_impl
    build_ninja(str(build_dir))
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/tvm_ffi/cpp/extension.py", line604, in build_ninja
    raise RuntimeError("\n".join(msg))
RuntimeError: ninja exited with status 1
stdout:
[1/2] /home/whan/cuda-12.8/bin/nvcc  --generate-dependencies-with-compile --dependency-output cuda_0.o.d -Xcompiler -fPIC -std=c++17 -O2 -gencode=arch=compute_90,code=sm_90 -DSGL_CUDA_ARCH=900 -std=c++20 -O3 --expt-relaxed-constexpr -I/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/tvm_ffi/include -I/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/tvm_ffi/include -I/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/jit_kernel/include -c /home/whan/.cache/tvm-ffi/sgl_kernel_jit_fused_rope_true_64_true_bf16_t_7836bb3521763c89/cuda.cu -o cuda_0.o
FAILED: [code=1] cuda_0.o 
/home/whan/cuda-12.8/bin/nvcc  --generate-dependencies-with-compile --dependency-output cuda_0.o.d -Xcompiler -fPIC -std=c++17 -O2 -gencode=arch=compute_90,code=sm_90 -DSGL_CUDA_ARCH=900 -std=c++20 -O3 --expt-relaxed-constexpr -I/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/tvm_ffi/include -I/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/tvm_ffi/include -I/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/jit_kernel/include -c /home/whan/.cache/tvm-ffi/sgl_kernel_jit_fused_rope_true_64_true_bf16_t_7836bb3521763c89/cuda.cu -o cuda_0.o
nvcc warning : incompatible redefinition for option 'std', the last value of this option was used
nvcc warning : incompatible redefinition for option 'optimize', the last value of this option was used
nvcc warning : The -std=c++20 flag is not supported with the configured host compiler. Flag willbe ignored.
In file included from /p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/jit_kernel/include/sgl_kernel/utils.h:42,
                 from /p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/jit_kernel/include/sgl_kernel/tensor.h:13,
                 from /p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/jit_kernel/csrc/elementwise/rope.cuh:1,
                 from /home/whan/.cache/tvm-ffi/sgl_kernel_jit_fused_rope_true_64_true_bf16_t_7836bb3521763c89/cuda.cu:7:
/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/jit_kernel/include/sgl_kernel/source_location.h:8:10: fatal error: version: No such file or directory
 #include <version>
          ^~~~~~~~~
compilation terminated.
ninja: build stopped: subcommand failed.


[2026-07-08 16:07:38] SIGQUIT received. signum=None, frame=None. It usually means one child failed.
scripts/demo_run.sh: line 1: 3636759 Killed                  sglang serve --model-path qwen/qwen2.5-0.5b-instruct --host 0.0.0.0 --port 30000 --disable-cuda-graph --disable-piecewise-cuda-graph--sampling-backend pytorch --attention-backend triton --disable-overlap-schedule
(sglang-play) bash-4.4$ 
```

And in the other terminal where I am hosting the server i get 

```
(sglang-play) bash-4.4$ bash scripts/demo_run.sh 
/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/srt/entrypoints/http_server.py:172: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
  from sglang.srt.utils.json_response import (
[2026-07-08 16:06:36] Fail to set RLIMIT_NOFILE: current limit exceeds maximum limit
[2026-07-08 16:06:36] server_args=ServerArgs(model_path='qwen/qwen2.5-0.5b-instruct', tokenizer_path='qwen/qwen2.5-0.5b-instruct', tokenizer_mode='auto', tokenizer_worker_num=1, skip_tokenizer_init=False, load_format='auto', model_loader_extra_config='{}', trust_remote_code=False, context_length=None, is_embedding=False, enable_multimodal=None, revision=None, model_impl='auto', host='0.0.0.0', port=30000, fastapi_root_path='', grpc_mode=False, skip_server_warmup=False, warmups=None, nccl_port=None, checkpoint_engine_wait_weights_before_ready=False, ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, ssl_keyfile_password=None, enable_ssl_refresh=False, dtype='auto', quantization=None, quantization_param_path=None, kv_cache_dtype='auto', enable_fp32_lm_head=False, modelopt_quant=None, modelopt_checkpoint_restore_path=None, modelopt_checkpoint_save_path=None, modelopt_export_path=None, quantize_and_serve=False, rl_quant_profile=None, mem_fraction_static=0.86, max_running_requests=None, max_queued_requests=None, max_total_tokens=None, chunked_prefill_size=8192, enable_dynamic_chunking=False, max_prefill_tokens=16384, prefill_max_requests=None, schedule_policy='fcfs', enable_priority_scheduling=False, disable_priority_preemption=False, default_priority_value=None, abort_on_priority_when_disabled=False, schedule_low_priority_values_first=False, priority_scheduling_preemption_threshold=10, schedule_conservativeness=1.0, page_size=1, swa_full_tokens_ratio=0.8, disable_hybrid_swa_memory=False, radix_eviction_policy='lru', enable_prefill_delayer=False, prefill_delayer_max_delay_passes=30, prefill_delayer_token_usage_low_watermark=None, prefill_delayer_forward_passes_buckets=None, prefill_delayer_wait_seconds_buckets=None, device='cuda', tp_size=1, pp_size=1, pp_max_micro_batch_size=None, pp_async_batch_depth=0, stream_interval=1, stream_response_default_include_usage=False, incremental_streaming_output=False, enable_streaming_session=False, random_seed=726810559, constrained_json_whitespace_pattern=None, constrained_json_disable_any_whitespace=False, watchdog_timeout=300, soft_watchdog_timeout=None, dist_timeout=None, download_dir=None, model_checksum=None, base_gpu_id=0, gpu_id_step=1, sleep_on_idle=False, use_ray=False, custom_sigquit_handler=None, log_level='info', log_level_http=None, log_requests=False, log_requests_level=2, log_requests_format='text', log_requests_target=None, uvicorn_access_log_exclude_prefixes=[], crash_dump_folder=None, show_time_cost=False, enable_metrics=False, enable_mfu_metrics=False, enable_metrics_for_all_schedulers=False, tokenizer_metrics_custom_labels_header='x-custom-labels', tokenizer_metrics_allowed_custom_labels=None, extra_metric_labels=None, bucket_time_to_first_token=None, bucket_inter_token_latency=None, bucket_e2e_request_latency=None, collect_tokens_histogram=False, prompt_tokens_buckets=None, generation_tokens_buckets=None, gc_warning_threshold_secs=0.0, decode_log_interval=40, enable_request_time_stats_logging=False, kv_events_config=None, enable_trace=False, otlp_traces_endpoint='localhost:4317', export_metrics_to_file=False, export_metrics_to_file_dir=None, api_key=None, admin_api_key=None, served_model_name='qwen/qwen2.5-0.5b-instruct', weight_version='default', chat_template=None, hf_chat_template_name=None, completion_template=None, file_storage_path='sglang_storage', enable_cache_report=False, reasoning_parser=None, tool_call_parser=None, tool_server=None, sampling_defaults='model', dp_size=1, load_balance_method='round_robin', attn_cp_size=1, moe_dp_size=1, dist_init_addr=None, nnodes=1, node_rank=0, json_model_override_args='{}', preferred_sampling_params=None, enable_lora=None, enable_lora_overlap_loading=None, max_lora_rank=None, lora_target_modules=None, lora_paths=None, max_loaded_loras=None, max_loras_per_batch=8, lora_eviction_policy='lru', lora_backend='csgmv', max_lora_chunk_size=16, experts_shared_outer_loras=None, attention_backend='triton', decode_attention_backend=None, prefill_attention_backend=None, sampling_backend='pytorch', grammar_backend='xgrammar', mm_attention_backend=None, fp8_gemm_runner_backend='auto', fp4_gemm_runner_backend='auto', nsa_prefill_backend=None, nsa_decode_backend=None, disable_flashinfer_autotune=False, mamba_backend='triton', speculative_algorithm=None, speculative_draft_model_path=None, speculative_draft_model_revision=None, speculative_draft_load_format=None, speculative_num_steps=None, speculative_eagle_topk=None, speculative_num_draft_tokens=None, speculative_accept_threshold_single=1.0, speculative_accept_threshold_acc=1.0, speculative_token_map=None, speculative_attention_mode='prefill', speculative_draft_attention_backend=None, speculative_moe_runner_backend='auto', speculative_moe_a2a_backend=None, speculative_draft_model_quantization=None, speculative_ngram_min_bfs_breadth=1, speculative_ngram_max_bfs_breadth=10, speculative_ngram_match_type='BFS', speculative_ngram_max_trie_depth=18, speculative_ngram_capacity=10000000, enable_multi_layer_eagle=False, ep_size=1, moe_a2a_backend='none', moe_runner_backend='auto', flashinfer_mxfp4_moe_precision='default', enable_flashinfer_allreduce_fusion=False, enforce_disable_flashinfer_allreduce_fusion=False, enable_aiter_allreduce_fusion=False, deepep_mode='auto', ep_num_redundant_experts=0, ep_dispatch_algorithm=None, init_expert_location='trivial', enable_eplb=False, eplb_algorithm='auto', eplb_rebalance_num_iterations=1000, eplb_rebalance_layers_per_chunk=None, eplb_min_rebalancing_utilization_threshold=1.0, expert_distribution_recorder_mode=None, expert_distribution_recorder_buffer_size=1000, enable_expert_distribution_metrics=False, deepep_config=None, moe_dense_tp_size=None, elastic_ep_backend=None, enable_elastic_expert_backup=False, mooncake_ib_device=None, max_mamba_cache_size=None, mamba_ssm_dtype=None, mamba_full_memory_ratio=0.9, mamba_scheduler_strategy='no_buffer', mamba_track_interval=256, linear_attn_backend='triton', linear_attn_decode_backend=None, linear_attn_prefill_backend=None, enable_hierarchical_cache=False, hicache_ratio=2.0, hicache_size=0, hicache_write_policy='write_through', hicache_io_backend='kernel', hicache_mem_layout='layer_first', hicache_storage_backend=None, hicache_storage_prefetch_policy='best_effort', hicache_storage_backend_extra_config=None, enable_hisparse=False, hisparse_config=None, enable_lmcache=False, kt_weight_path=None, kt_method='AMXINT4', kt_cpuinfer=None, kt_threadpool_count=2, kt_num_gpu_experts=None, kt_max_deferred_experts_per_token=None, dllm_algorithm=None, dllm_algorithm_config=None, enable_double_sparsity=False, ds_channel_config_path=None, ds_heavy_channel_num=32, ds_heavy_token_num=256, ds_heavy_channel_type='qk', ds_sparse_decode_threshold=4096, cpu_offload_gb=0, offload_group_size=-1, offload_num_in_group=1, offload_prefetch_step=1, offload_mode='cpu', multi_item_scoring_delimiter=None, disable_radix_cache=False, cuda_graph_max_bs=256, cuda_graph_bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256], disable_cuda_graph=True, disable_cuda_graph_padding=False, enable_profile_cuda_graph=False, enable_cudagraph_gc=False, enable_layerwise_nvtx_marker=False, enable_nccl_nvls=False, enable_symm_mem=False, disable_flashinfer_cutlass_moe_fp4_allgather=False, enable_tokenizer_batch_encode=False, disable_tokenizer_batch_decode=False, disable_outlines_disk_cache=False, disable_custom_all_reduce=False, enable_mscclpp=False, enable_torch_symm_mem=False, pre_warm_nccl=False, disable_overlap_schedule=True, enable_mixed_chunk=False, enable_dp_attention=False, enable_dp_lm_head=False, enable_two_batch_overlap=False, enable_single_batch_overlap=False, tbo_token_distribution_threshold=0.48, enable_torch_compile=False, disable_piecewise_cuda_graph=True, enforce_piecewise_cuda_graph=False, enable_torch_compile_debug_mode=False, torch_compile_max_bs=32, piecewise_cuda_graph_max_tokens=8192, piecewise_cuda_graph_tokens=[4, 8, 12, 16, 20, 24, 28, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416, 448, 480, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192], piecewise_cuda_graph_compiler='eager', torchao_config='', enable_nan_detection=False, enable_p2p_check=False, triton_attention_reduce_in_fp32=False, triton_attention_num_kv_splits=8, triton_attention_split_tile_size=None, num_continuous_decode_steps=1, delete_ckpt_after_loading=False, enable_memory_saver=False, enable_weights_cpu_backup=False, enable_draft_weights_cpu_backup=False, allow_auto_truncate=False, enable_custom_logit_processor=False, flashinfer_mla_disable_ragged=False, disable_shared_experts_fusion=False, disable_chunked_prefix_cache=False, disable_fast_image_processor=False, keep_mm_feature_on_device=False, enable_return_hidden_states=False, enable_return_routed_experts=False, scheduler_recv_interval=1, numa_node=None, enable_deterministic_inference=False, rl_on_policy_target=None, enable_attn_tp_input_scattered=False, gc_threshold=None, enable_nsa_prefill_context_parallel=False, nsa_prefill_cp_mode='round-robin-split', enable_fused_qk_norm_rope=False, enable_precise_embedding_interpolation=False, enable_fused_moe_sum_all_reduce=False, enable_prefill_context_parallel=False, prefill_cp_mode='in-seq-split', enable_dynamic_batch_tokenizer=False, dynamic_batch_tokenizer_batch_size=32, dynamic_batch_tokenizer_batch_timeout=0.002, debug_tensor_dump_output_folder=None, debug_tensor_dump_layers=None, debug_tensor_dump_input_file=None, debug_tensor_dump_inject=False, disaggregation_mode='null', disaggregation_transfer_backend='mooncake', disaggregation_bootstrap_port=8998, disaggregation_ib_device=None, disaggregation_decode_enable_offload_kvcache=False, num_reserved_decode_tokens=512, disaggregation_decode_polling_interval=1, encoder_only=False, language_only=False, encoder_transfer_backend='zmq_to_scheduler', encoder_urls=[], enable_adaptive_dispatch_to_encoder=False, custom_weight_loader=[], weight_loader_disable_mmap=False, remote_instance_weight_loader_seed_instance_ip=None, remote_instance_weight_loader_seed_instance_service_port=None, remote_instance_weight_loader_send_weights_group_ports=None, remote_instance_weight_loader_backend='nccl', remote_instance_weight_loader_start_seed_via_transfer_engine=False, engine_info_bootstrap_port=6789, modelexpress_config=None, enable_pdmux=False, pdmux_config_path=None, sm_group_num=8, enable_broadcast_mm_inputs_process=False, enable_prefix_mm_cache=False, mm_enable_dp_encoder=False, mm_process_config={}, limit_mm_data_per_request=None, enable_mm_global_cache=False, decrypted_config_file=None, decrypted_draft_config_file=None, forward_hooks=None)
[2026-07-08 16:06:37] Using default HuggingFace chat template with detected content format: string
[2026-07-08 16:06:47] Init torch distributed begin.
[2026-07-08 16:06:48] Init torch distributed ends. elapsed=0.84 s, mem usage=0.09 GB
[2026-07-08 16:06:48] Load weight begin. avail mem=36.54 GB
[2026-07-08 16:06:48] Found local HF snapshot for qwen/qwen2.5-0.5b-instruct at /p01/whan/.huggingface/models--qwen--qwen2.5-0.5b-instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775; skipping download.
[2026-07-08 16:06:48] No model.safetensors.index.json found in remote.
Multi-thread loading shards: 100% Completed | 1/1 [00:00<00:00,  6.73it/s]
[2026-07-08 16:06:49] Load weight end. elapsed=0.41 s, type=Qwen2ForCausalLM, avail mem=35.57 GB, mem usage=0.98 GB.
[2026-07-08 16:06:49] Using KV cache dtype: torch.bfloat16
[2026-07-08 16:06:49] KV Cache is allocated. #tokens: 2658165, K size: 15.21 GB, V size: 15.21 GB
[2026-07-08 16:06:49] Memory pool end. avail mem=4.55 GB
[2026-07-08 16:06:49] Disable piecewise CUDA graph because --disable-piecewise-cuda-graph is set
[2026-07-08 16:06:50] max_total_num_tokens=2658165, chunked_prefill_size=8192, max_prefill_tokens=16384, max_running_requests=4096, context_len=32768, available_gpu_mem=4.46 GB
[2026-07-08 16:06:50] INFO:     Started server process [3636759]
[2026-07-08 16:06:50] INFO:     Waiting for application startup.
[2026-07-08 16:06:50] Using default chat sampling params from model generation config: {'repetition_penalty': 1.1, 'temperature': 0.7, 'top_k': 20, 'top_p': 0.8}
[2026-07-08 16:06:50] INFO:     Application startup complete.
[2026-07-08 16:06:50] INFO:     Uvicorn running on http://0.0.0.0:30000 (Press CTRL+C to quit)
[2026-07-08 16:07:38] Scheduler hit an exception: Traceback (most recent call last):
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/srt/managers/scheduler.py", line 3616, in run_scheduler_process
    scheduler.run_event_loop()
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/srt/managers/scheduler.py", line 1300, in run_event_loop
    dispatch_event_loop(self)
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/srt/managers/scheduler.py", line 3499, in dispatch_event_loop
    scheduler.event_loop_normal()
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/torch/utils/_contextlib.py", line 124, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/srt/managers/scheduler.py", line 1319, in event_loop_normal
    result = self.run_batch(batch)
             ^^^^^^^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/srt/managers/scheduler.py", line 2724, in run_batch
    batch_result = self.model_worker.forward_batch_generation(
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/srt/managers/tp_worker.py", line 469, in forward_batch_generation
    out = self.model_runner.forward(
          ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/srt/model_executor/model_runner.py", line 2739, in forward
    output = self._forward_raw(
             ^^^^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/srt/model_executor/model_runner.py", line 2849, in _forward_raw
    ret, can_run_graph = self.forward_extend(
                         ^^^^^^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/srt/model_executor/model_runner.py", line 2676, in forward_extend
    self.model.forward(
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/torch/utils/_contextlib.py", line 124, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/srt/models/qwen2.py", line 482, in forward
    hidden_states = self.model(
                    ^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1790, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/srt/models/qwen2.py", line 363, in forward
    hidden_states, residual = layer(
                              ^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1790, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/srt/models/qwen2.py", line 249, in forward
    hidden_states = self.self_attn(
                    ^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1790, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/srt/models/qwen2.py", line 188, in forward
    q, k = self.rotary_emb(positions, q, k)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1790, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/srt/layers/utils/multi_platform.py", line 73, in forward
    return self._forward_method(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/srt/layers/rotary_embedding/base.py", line 340, in forward_cuda
    apply_rope_with_cos_sin_cache_inplace(
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/jit_kernel/rope.py", line 219, in apply_rope_with_cos_sin_cache_inplace
    apply_rope_inplace(
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/torch/_ops.py", line 1269, in __call__
    return self._op(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/jit_kernel/rope.py", line 136, in apply_rope_inplace
    module = _jit_fused_rope_module(is_neox, rope_dim, q.dtype)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/jit_kernel/utils.py", line 57, in wrapper
    result_map[key] = fn(*args, **kwargs)
                      ^^^^^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/jit_kernel/rope.py", line 32, in _jit_fused_rope_module
    return load_jit(
           ^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/jit_kernel/utils.py", line 208, in load_jit
    return load_inline(
           ^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/tvm_ffi/cpp/extension.py", line 1132, in load_inline
    build_inline(
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/tvm_ffi/cpp/extension.py", line 974, in build_inline
    return _build_impl(
           ^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/tvm_ffi/cpp/extension.py", line 738, in _build_impl
    build_ninja(str(build_dir))
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/tvm_ffi/cpp/extension.py", line 604, in build_ninja
    raise RuntimeError("\n".join(msg))
RuntimeError: ninja exited with status 1
stdout:
[1/2] /home/whan/cuda-12.8/bin/nvcc  --generate-dependencies-with-compile --dependency-output cuda_0.o.d -Xcompiler -fPIC -std=c++17 -O2 -gencode=arch=compute_90,code=sm_90 -DSGL_CUDA_ARCH=900 -std=c++20 -O3 --expt-relaxed-constexpr -I/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/tvm_ffi/include -I/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/tvm_ffi/include -I/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/jit_kernel/include -c /home/whan/.cache/tvm-ffi/sgl_kernel_jit_fused_rope_true_64_true_bf16_t_7836bb3521763c89/cuda.cu -o cuda_0.o
FAILED: [code=1] cuda_0.o 
/home/whan/cuda-12.8/bin/nvcc  --generate-dependencies-with-compile --dependency-output cuda_0.o.d -Xcompiler -fPIC -std=c++17 -O2 -gencode=arch=compute_90,code=sm_90 -DSGL_CUDA_ARCH=900 -std=c++20 -O3 --expt-relaxed-constexpr -I/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/tvm_ffi/include -I/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/tvm_ffi/include -I/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/jit_kernel/include -c /home/whan/.cache/tvm-ffi/sgl_kernel_jit_fused_rope_true_64_true_bf16_t_7836bb3521763c89/cuda.cu -o cuda_0.o
nvcc warning : incompatible redefinition for option 'std', the last value of this option was used
nvcc warning : incompatible redefinition for option 'optimize', the last value of this option was used
nvcc warning : The -std=c++20 flag is not supported with the configured host compiler. Flag will be ignored.
In file included from /p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/jit_kernel/include/sgl_kernel/utils.h:42,
                 from /p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/jit_kernel/include/sgl_kernel/tensor.h:13,
                 from /p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/jit_kernel/csrc/elementwise/rope.cuh:1,
                 from /home/whan/.cache/tvm-ffi/sgl_kernel_jit_fused_rope_true_64_true_bf16_t_7836bb3521763c89/cuda.cu:7:
/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/jit_kernel/include/sgl_kernel/source_location.h:8:10: fatal error: version: No such file or directory
 #include <version>
          ^~~~~~~~~
compilation terminated.
ninja: build stopped: subcommand failed.


[2026-07-08 16:07:38] SIGQUIT received. signum=None, frame=None. It usually means one child failed.
scripts/demo_run.sh: line 1: 3636759 Killed                  sglang serve --model-path qwen/qwen2.5-0.5b-instruct --host 0.0.0.0 --port 30000 --disable-cuda-graph --disable-piecewise-cuda-graph --sampling-backend pytorch --attention-backend triton --disable-overlap-schedule
(sglang-play) bash-4.4$ 
```


For the src/offline_demo.py

I get 

```
(sglang-play) bash-4.4$ uv run python3 src/offline_demo.py 
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/home/whan/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib/python3.11/multiprocessing/spawn.py", line 122, in spawn_main
    exitcode = _main(fd, parent_sentinel)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/whan/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib/python3.11/multiprocessing/spawn.py", line 131, in _main
    prepare(preparation_data)
  File "/home/whan/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib/python3.11/multiprocessing/spawn.py", line 246, in prepare
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/home/whan/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib/python3.11/multiprocessing/spawn.py", line 122, in spawn_main
    _fixup_main_from_path(data['init_main_from_path'])
  File "/home/whan/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib/python3.11/multiprocessing/spawn.py", line 297, in _fixup_main_from_path
    main_content = runpy.run_path(main_path,
                   ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<frozen runpy>", line 291, in run_path
  File "<frozen runpy>", line 98, in _run_module_code
  File "<frozen runpy>", line 88, in _run_code
  File "/p01/whan/sglang-play/src/offline_demo.py", line 3, in <module>
    exitcode = _main(fd, parent_sentinel)
    llm = sglang.Engine(model_path="qwen/qwen2.5-0.5b-instruct")
               ^^^^^^^^^^^^^^^^^^^^^^^^^^
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/utils.py", line 406, in __call__
  File "/home/whan/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib/python3.11/multiprocessing/spawn.py", line 131, in _main
    return module(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/srt/entrypoints/engine.py", line 197, in __init__
    ) = self._launch_subprocesses(
        ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/srt/entrypoints/engine.py", line 668, in _launch_subprocesses
    scheduler_init_result, scheduler_procs = cls._launch_scheduler_processes(
                                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/srt/entrypoints/engine.py", line 576, in _launch_scheduler_processes
    proc.start()
  File "/home/whan/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib/python3.11/multiprocessing/process.py", line 121, in start
    prepare(preparation_data)
  File "/home/whan/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib/python3.11/multiprocessing/spawn.py", line 246, in prepare
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "/home/whan/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib/python3.11/multiprocessing/context.py", line 224, in _Popen
    _fixup_main_from_path(data['init_main_from_path'])
  File "/home/whan/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib/python3.11/multiprocessing/spawn.py", line 297, in _fixup_main_from_path
    return _default_context.get_context().Process._Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/whan/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib/python3.11/multiprocessing/context.py", line 288, in _Popen
    main_content = runpy.run_path(main_path,
                   ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<frozen runpy>", line 291, in run_path
  File "<frozen runpy>", line 98, in _run_module_code
  File "<frozen runpy>", line 88, in _run_code
  File "/p01/whan/sglang-play/src/offline_demo.py", line 3, in <module>
    llm = sglang.Engine(model_path="qwen/qwen2.5-0.5b-instruct")
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/utils.py", line 406, in __call__
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "/home/whan/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib/python3.11/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    return module(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/srt/entrypoints/engine.py", line 197, in __init__
    ) = self._launch_subprocesses(
        ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/srt/entrypoints/engine.py", line 668, in _launch_subprocesses
    scheduler_init_result, scheduler_procs = cls._launch_scheduler_processes(
    super().__init__(process_obj)
                                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/srt/entrypoints/engine.py", line 576, in _launch_scheduler_processes
  File "/home/whan/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib/python3.11/multiprocessing/popen_fork.py", line 19, in __init__
    proc.start()
  File "/home/whan/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib/python3.11/multiprocessing/process.py", line 121, in start
    self._launch(process_obj)
  File "/home/whan/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib/python3.11/multiprocessing/popen_spawn_posix.py", line 42, in _launch
    self._popen = self._Popen(self)
                  ^^^^^^^^^^^^^^^^^
  File "/home/whan/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib/python3.11/multiprocessing/context.py", line 224, in _Popen
    prep_data = spawn.get_preparation_data(process_obj._name)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/whan/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib/python3.11/multiprocessing/spawn.py", line 164, in get_preparation_data
    return _default_context.get_context().Process._Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/whan/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib/python3.11/multiprocessing/context.py", line 288, in _Popen
    _check_not_importing_main()
  File "/home/whan/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib/python3.11/multiprocessing/spawn.py", line 140, in _check_not_importing_main
    return Popen(process_obj)
           ^^^^^^^^^^^^^^^^^^
  File "/home/whan/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib/python3.11/multiprocessing/popen_spawn_posix.py", line 32, in __init__
    raise RuntimeError('''
RuntimeError: 
        An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.

        This probably means that you are not using fork to start your
        child processes and you have forgotten to use the proper idiom
        in the main module:

            if __name__ == '__main__':
                freeze_support()
                ...

        The "freeze_support()" line can be omitted if the program
        is not going to be frozen to produce an executable.

        To fix this issue, refer to the "Safe importing of main module"
        section in https://docs.python.org/3/library/multiprocessing.html
        
    super().__init__(process_obj)
  File "/home/whan/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib/python3.11/multiprocessing/popen_fork.py", line 19, in __init__
    self._launch(process_obj)
  File "/home/whan/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib/python3.11/multiprocessing/popen_spawn_posix.py", line 42, in _launch
    prep_data = spawn.get_preparation_data(process_obj._name)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/whan/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib/python3.11/multiprocessing/spawn.py", line 164, in get_preparation_data
    _check_not_importing_main()
  File "/home/whan/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib/python3.11/multiprocessing/spawn.py", line 140, in _check_not_importing_main
    raise RuntimeError('''
RuntimeError: 
        An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.

        This probably means that you are not using fork to start your
        child processes and you have forgotten to use the proper idiom
        in the main module:

            if __name__ == '__main__':
                freeze_support()
                ...

        The "freeze_support()" line can be omitted if the program
        is not going to be frozen to produce an executable.

        To fix this issue, refer to the "Safe importing of main module"
        section in https://docs.python.org/3/library/multiprocessing.html
        
Traceback (most recent call last):
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/srt/entrypoints/engine.py", line 1207, in _wait_for_scheduler_ready
    data = scheduler_pipe_readers[i].recv()
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/whan/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib/python3.11/multiprocessing/connection.py", line 250, in recv
    buf = self._recv_bytes()
          ^^^^^^^^^^^^^^^^^^
  File "/home/whan/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib/python3.11/multiprocessing/connection.py", line 430, in _recv_bytes
    buf = self._recv(4)
          ^^^^^^^^^^^^^
  File "/home/whan/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/lib/python3.11/multiprocessing/connection.py", line 399, in _recv
    raise EOFError
EOFError

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/p01/whan/sglang-play/src/offline_demo.py", line 3, in <module>
    llm = sglang.Engine(model_path="qwen/qwen2.5-0.5b-instruct")
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/utils.py", line 406, in __call__
    return module(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/srt/entrypoints/engine.py", line 197, in __init__
    ) = self._launch_subprocesses(
        ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/srt/entrypoints/engine.py", line 730, in _launch_subprocesses
    scheduler_init_result.wait_for_ready()
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/srt/entrypoints/engine.py", line 599, in wait_for_ready
    infos = _wait_for_scheduler_ready(scheduler_pipe_readers, scheduler_procs)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/p01/whan/sglang-play/.venv/lib/python3.11/site-packages/sglang/srt/entrypoints/engine.py", line 1209, in _wait_for_scheduler_ready
    raise _scheduler_died_error(i, scheduler_procs[i])
RuntimeError: Rank 0 scheduler died during initialization (exit code: 1). If exit code is -9 (SIGKILL), a common cause is the OS OOM killer. Run `dmesg -T | grep -i oom` to check.
```