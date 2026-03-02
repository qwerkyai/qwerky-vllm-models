# Qwerky vLLM Models

A vLLM plugin for serving Qwerky AI's MambaInLlama hybrid models without the `--trust-remote-code` flag.

## Installation

```bash
pip install vllm qwerky-vllm-models
```

## Usage

After installing, serve Qwerky models with vLLM:

```bash
vllm serve QwerkyAI/Qwerky-Llama3.2-Mamba-3B-Llama3.3-70B-base-distill --max-model-len 4096
```

The plugin automatically registers the model architecture with vLLM on import.

## Supported Models

- `QwerkyAI/Qwerky-Llama3.2-Mamba-3B-Llama3.3-70B-base-distill` (Mamba 1 hybrid)
- `QwerkyAI/HybridMamba2-Exp6umrbg19-full` (Mamba 2 hybrid)

## How It Works

This package uses vLLM's plugin system (`vllm.general_plugins` entry point) to register the MambaInLlama model architecture. This means:

- No fork of vLLM required
- No `--trust-remote-code` flag needed
- Works with standard vLLM installation
- CUDA graph support for optimized decode latency
- Uses vLLM's native Triton-accelerated Mamba kernels
- Mamba 2 support via vLLM's built-in `MambaMixer2` (chunked semi-parallel SSD scan)

## Requirements

- Python >= 3.10
- vLLM >= 0.14.0
- PyTorch >= 2.0.0

## Changelog

### 0.2.70
- **Mamba 2 hybrid model support** via vLLM's built-in `MambaMixer2` layer
- New `Mamba2DecoderLayer` wrapping `MambaMixer2` — gets CUDA graphs, prefix caching, TP, and FP8 for free
- Backbone dispatches `Mamba2DecoderLayer` vs `MambaDecoderLayer` based on `mamba_version` config field
- Dynamic `mamba_in_proj_sizes` for Mamba 2 weight loading (`[gate, x, B, C, dt]` split)
- State shape/dtype delegates to `mamba2_state_shape()` / `mamba2_state_dtype()` for Mamba 2 models
- New architecture: `QwerkyMamba2HybridForCausalLM` registered in plugin
- 7 new Mamba 2 config fields: `mamba_version`, `mamba_n_groups`, `mamba_num_heads`, `mamba_head_dim`, `ssm_state_size`, `d_conv`, `mamba_intermediate_size`
- All Mamba 1 code unchanged — fully backward compatible

### 0.2.69
- **FP8 quantization support**: Wire `quant_config` through all Mamba layers
- `quant_config` passed to `MambaInLlamaMambaMixer` (in_proj, dt_proj, out_proj), `MLP` (gate_up_proj, down_proj), and `MambaDecoderLayer`
- conv1d intentionally excluded (tiny depthwise conv, no benefit from quantization)
- With `--quantization fp8` on H100: enables FP8 for all major GEMMs in all layers
- No regression without quantization (`quant_config=None` is a no-op)

### 0.2.68
- **Multi-file restructure**: Split `modeling.py` (1462 lines) into `modeling/` package
- `modeling/mixer.py`: `MambaInLlamaMambaMixer` + custom op registration (~600 lines)
- `modeling/layers.py`: `RMSNorm` fallback, `MLP`, `MambaDecoderLayer` (~150 lines)
- `modeling/model.py`: Backbone, CausalLM, weight loading, aliases (~500 lines)
- `modeling/__init__.py`: Re-exports all public classes for backward compatibility
- Zero functional change — all import paths preserved

### 0.2.67
- **Replace custom MHADecoderLayer with vLLM's native `LlamaDecoderLayer`**
- Gets quant_config support, proper RoPE (via vLLM's `patch_rope_parameters`), and future vLLM patches for free
- Weight mapping: `mha.in_proj` → `self_attn.qkv_proj` (split q/k/v), `mha.out_proj` → `self_attn.o_proj`
- Unified backbone forward: `layer(positions, hidden_states, residual)` for all layer types
- Removes ~100 lines of custom attention code

### 0.2.66
- **MHA tensor parallelism**: `QKVParallelLinear` + `RowParallelLinear` for attention layers
- **RoPE config fix**: `max_position_embeddings` passed correctly (was capped at 2048 with torch.compile)
- **HF accuracy parity achieved**: mmlu 0.35→0.44, hellaswag_norm 0.58→0.63, gsm8k 0.09→0.47
- Register `QwerkyLlamaMambaHybridConfig` with `AutoConfig`
- **Metadata optimization**: Pass `metadata=layer_metadata` to `causal_conv1d_fn`, eliminating 28 CPU syncs/prefill (+6% throughput)

### 0.2.65
- **Speed optimizations**: Precomputed static tensors (conv_weight, D, dt_bias, multi-head views) — avoids per-forward recomputation
- **Early slicing**: Slice to actual tokens after `in_proj` before expensive split/dt_proj/expand operations, skipping CUDA graph padding
- **expand instead of repeat_interleave**: Zero-copy stride-0 view + single reshape for x and B expansion
- **permute instead of rearrange**: Direct `permute()` for prefill B/C transpose, avoids einops overhead on hot path
- **Fused MLP**: `MergedColumnParallelLinear` + `SiluAndMul` + `RowParallelLinear` (TP-ready, replaces separate gate/up projections)
- **Fused RMSNorm residual**: Thread residual through all layers — fuses elementwise add + normalization into one CUDA kernel
- **`@support_torch_compile`**: Applied to model backbone, registers custom op as splitting op for torch.compile exclusion
- **Architecture alias**: `QwerkyLlamaMambaHybridForCausalLM` registered in plugin

### 0.2.64
- **FIX**: Guard `layer_metadata.block_idx_last_computed_token` with `is not None` before splitting
- Fields are `None` during CUDA graph decode capture even when prefix caching is enabled
- Fixes `AttributeError: 'NoneType' object has no attribute 'split'` crash with `--enable-prefix-caching`

### 0.2.63
- **LoRA support**: `is_lora_enabled` parameter, contiguity enforcement for LoRA kernel
- **ROCm platform check**: `current_platform.is_rocm()` contiguity for non-contiguous GEMM correctness
- **LoRA-aware output projection**: Separate contiguous path for LoRA kernel compatibility
- **Speculative decoding ready**: `selective_state_update` and `causal_conv1d_update` accept `num_accepted_tokens` (kernel-level plumbing in place)

### 0.2.62
- **Tensor parallelism support**: Replace `nn.Linear` with vLLM parallel layers (`MergedColumnParallelLinear`, `RowParallelLinear`, `ColumnParallelLinear`)
- **TP-aware weight loading**: `set_weight_attrs` for A and D parameters with custom weight_loaders for TP sharding and A_log→A conversion
- **Rewrite `load_weights`** to use vLLM `weight_loader` pattern for all parameters
- **Conv1d as ColumnParallelLinear**: Matches vLLM MambaMixer pattern, enables TP sharding of conv weights
- **Per-partition dimension tracking**: `d_inner_local`, `d_xb_local`, etc. for correct TP operation
- **TP-aware state shapes**: `get_state_shape()` uses `MambaStateShapeCalculator` for TP-correct cache allocation

### 0.2.61
- **Prefix caching support**: Pass `block_idx_first_scheduled_token`, `block_idx_last_scheduled_token`, `initial_state_idx`, `num_computed_tokens`, and `block_size_to_align` to `causal_conv1d_fn` and `selective_scan_fn`
- **Separate read/write state indices for decode**: Compute `state_indices_d_input` and `state_indices_d_output` via `.gather()` when prefix caching is enabled
- **`dst_state_batch_indices` support**: Pass to `selective_state_update` for correct state write-back with prefix caching
- **Prefix caching params for decode conv**: Pass `block_idx_last_scheduled_token` and `initial_state_idx` to `causal_conv1d_update`
- Store `mamba_block_size` from cache config

### 0.2.60
- **MAJOR FIX**: Rewrite decode path to match vLLM kernel conventions
- `causal_conv1d_update` expects batch-first `(num_decode, d_inner)`, not dim-first
- `selective_state_update` needs multi-head format: reshape state/x/dt/z to `(*, nheads, head_dim, ...)` so kernel's `nheads % ngroups == 0` assertion passes with grouped B/C
- Preallocate `out` tensor for `selective_state_update` (required, returns None)

### 0.2.59
- **FIX**: Conv state shape must be `(d_conv-1, conv_dim)` not `(conv_dim, d_conv-1)`
- vLLM's `causal_conv1d_fn` asserts `stride_istate_dim == 1` (conv_dim must be contiguous)
- Matches vLLM's `mamba_utils.py:mamba1_state_shape()` which swaps the axes

### 0.2.58
- **FIX**: Register `mambainllama_mixer` custom op via `direct_register_custom_op`
- `@CustomOp.register()` only adds to vLLM's internal registry, does not create a `torch.ops.vllm.*` callable
- Now properly creates the torch op that `forward()` dispatches through

### 0.2.57
- **FIX**: Custom op name mismatch — `forward()` called `torch.ops.vllm.mamba_mixer` but op was registered as `mambainllama_mixer`

### 0.2.56
- **MAJOR**: CUDA graph support via custom op pattern
- Adopt vLLM's `MambaBase + CustomOp` pattern for CUDA graph compatibility
- `torch.ops.vllm.mambainllama_mixer` dispatch acts as compiler breakpoint
- Fix state shapes: conv `(conv_dim, d_conv-1)`, ssm `(d_inner, d_state)` — no transpose needed
- Output tensor pattern for custom op compatibility
- `VocabParallelEmbedding`, `load_weights` returns `set[str]`
- Remove factory pattern, fallback state management, `is_attention_free`

### 0.2.55
- **FIX**: Compute SSM scan in float32 to match original `selective_scan_fn` precision
- bfloat16 at dA~0.98 causes ~55% cumulative error over 100 steps

### 0.2.54
- **MAJOR**: Use vLLM's `Attention` class for MHA layers
- Replaced manual attention with vLLM's native Attention — model now produces coherent output
- `ParallelLMHead`, `cache_config` passthrough, `get_rope()`

### 0.2.53
- Self-managed KV cache for MHA layers (superseded by v0.2.54)

### 0.2.52
- Environment version logging on plugin startup

### 0.2.51
- Cleanup of debug logging from earlier versions

### 0.2.50
- Remove excessive checkpoint weight logging

### 0.2.49
- Fix weight loading edge cases for attention layer projections

### 0.2.48
- Improve A_log -> A conversion logging

### 0.2.47
- Fix repeat_kv expansion for grouped-head Mamba

### 0.2.46
- Cleanup debug prints from v0.2.39-0.2.40

### 0.2.45
- Fix conv1d weight shape handling for vLLM ops path

### 0.2.44
- **CRITICAL FIX**: Proper state persistence in PyTorch fallback path
- Previously, SSM state was reset to zero every forward call, causing output degeneration
- Now properly initializes SSM state from `ssm_state` parameter if provided
- Updates `ssm_state` with final state after scan for next token generation
- Handles `conv_state` for proper causal convolution context
- This should fix the "Paris...garbage" issue where first token was correct but rest was gibberish

### 0.2.43
- **FIX**: Fix dtype mismatch in PyTorch fallback path
- A and D parameters were initialized as float32, causing mismatch with bfloat16 inputs
- Cast A and D to input dtype before use in SSM computation
- Fixes: `RuntimeError: expected scalar type BFloat16 but found Float`

### 0.2.42
- **FIX**: Fix shape mismatch in PyTorch fallback SSM computation
- Line 843: `A.unsqueeze(0).unsqueeze(-1)` → `A.unsqueeze(0).unsqueeze(2)`
- dt shape (batch, d_inner, seqlen) now correctly broadcasts with A shape (d_inner, d_state)

### 0.2.41
- **CRITICAL FIX**: Remove early return when attn_metadata is None
- The early return (added in v0.2.33) was triggering during actual inference, not just warmup
- This caused the model to skip all SSM computation and output gibberish
- Now the model always performs actual Mamba SSM computation
- Internal caches are used when vLLM doesn't provide state

### 0.2.40
- **DEBUG**: Added print statement at forward entry to confirm Mixer is called
- Print shows layer index and whether attn_metadata is present
- This will reveal if forward is being called at all

### 0.2.39
- **DEBUG**: Added split statistics logging to diagnose gibberish output
- Logs z/x/B/C/dt shapes and mean/std after in_proj split
- Logs which forward path is taken (vLLM ops vs PyTorch fallback)
- This will help identify if the in_proj split order is correct

### 0.2.38
- **CRITICAL FIX**: Restore double bias in dt_proj for vLLM ops path
- Model was trained with bias applied twice: once in dt_proj, once in softplus
- Changed `dt_proj.weight @ dt` to `dt_proj(dt)` to include first bias application
- SSM kernel applies second bias via `delta_bias` parameter
- This matches the fix in v0.2.24 but was missing in the vLLM ops code path

### 0.2.37
- **CRITICAL FIX**: Handle `A_log` -> `A` weight conversion for Mamba layers
- Checkpoint stores `A_log` but model uses `A = -exp(A_log)` per Mamba paper
- This was causing 22 Mamba layer weights to not load, resulting in gibberish output
- Now all 343/343 parameters should load correctly

### 0.2.36
- **MAJOR**: Use `get_forward_context()` to retrieve state in vLLM V1 mode
- In V1, `attn_metadata` is a dict keyed by layer `prefix` - now indexed correctly
- Retrieve `state_indices_tensor` and `query_start_loc` from layer-specific metadata
- Get `conv_state`/`ssm_state` from `self.kv_cache[virtual_engine]`
- Added V1-specific debug logging to diagnose state retrieval
- This matches how vLLM's native MambaMixer retrieves state in V1 architecture

### 0.2.33
- **FIX**: Early return during warmup (matches vLLM native MambaMixer)
- When attn_metadata is None, skip SSM computation entirely
- Just do in_proj -> out_proj for shape/memory profiling
- No performance impact on actual inference (only affects warmup)

### 0.2.32
- **FIX**: Handle None state_indices during warmup/profiling
- When state_indices is None, pass None for conv_state/ssm_state to kernels
- vLLM kernels expect both indices and state together, or neither
- This fixes Triton compilation error: `'NoneType' object has no attribute 'type'`

### 0.2.31
- **FIX**: Fix `stride_istate_dim == 1` assertion in causal_conv1d_fn
- vLLM's causal_conv1d expects conv_state with stride_dim == 1 (dim axis contiguous)
- Changed state storage format: (batch, d_conv-1, conv_dim) with transpose before use
- Similarly fixed ssm_state: (batch, d_state, d_inner) with transpose before use
- Updated `get_state_shape()`, `allocate_inference_cache()`, and `_ensure_cache()` to match

### 0.2.30
- **FIX**: Adapt to vLLM 0.14+ API changes for `causal_conv1d_fn` and `selective_scan_fn`
- vLLM 0.14 requires `query_start_loc` parameter for varlen batching support
- Construct `query_start_loc` from attn_metadata or input shape
- Updated tensor shapes for prefill path: (dim, total_tokens) format
- Pass `query_start_loc` to both conv and SSM scan functions

### 0.2.29
- **FIX**: Use plain nn.Module instead of MambaBase to fix parameter registration
- MambaBase inherits from AttentionLayerBase which breaks nn.Module initialization
- This was causing only 187/395 parameters to load (Mamba weights not registered)
- Mixer now manages its own state via `_conv_state`/`_ssm_state` with `_ensure_cache()`
- Restored `allocate_inference_cache` method for compatibility
- State priority: 1) forward args, 2) vLLM kv_cache, 3) internal caches

### 0.2.28
- **FIX**: Remove CustomOp inheritance - it conflicts with direct module calls
- MambaBase inheritance alone is sufficient for vLLM state allocation discovery
- Mixer now has standard nn.Module forward signature (returns output, accepts optional state)
- Removed `allocate_inference_cache` - state is now managed by vLLM via `bind_kv_cache()`
- Removed manual cache management (`_init_caches`, `_mamba_cache`, `_attn_cache`)
- Mixer gets state from `self.kv_cache` (bound by vLLM) or from forward args

### 0.2.27
- **MAJOR**: Proper vLLM V1 integration with @CustomOp.register + MambaBase
- Uses `@CustomOp.register("mambainllama_mixer")` decorator for correct callability
- Inherits from both `MambaBase` (for state allocation) and `CustomOp` (for dispatch)
- This makes layer discoverable by vLLM's state allocation system (via AttentionLayerBase)
- vLLM now properly allocates and binds `kv_cache` (conv_state, ssm_state) to each layer
- Implements `forward()`, `forward_cuda()`, `forward_native()` per CustomOp interface
- Uses vLLM's native ops (`selective_state_update`, `causal_conv1d_update`) with `cache_indices`
- State persistence should now work correctly with CUDA graphs
- Removed internal cache management - uses vLLM's unified allocator instead

### 0.2.26
- **FIX**: Don't inherit from MambaBase - it breaks nn.Module callability
- MambaBase inherits from AttentionLayerBase which requires CustomOp decorator
- Keep nn.Module as base, implement MambaBase interface methods separately
- This fixes "object is not callable" error and restores parameter registration

### 0.2.25
- **MAJOR**: Conform to vLLM's caching style for CUDA graph compatibility
- Implements `get_state_shape()`, `get_state_dtype()`, and `mamba_type` property
- Registers layers in `static_forward_context` for CUDA graph support
- Added `state_indices` support for proper batch indexing via `attn_metadata`
- Added `copy_inputs_before_cuda_graphs()` and `get_seqlen_agnostic_capture_inputs()`
- Passes `attn_metadata` through the model forward chain
- Should fix state persistence issues causing output degeneration/repetition

### 0.2.24
- **FIX**: Restore double bias in dt/delta computation
- Reference implementation intentionally applies dt_proj.bias twice:
  1. Once in `dt_proj(dt)` (Linear includes bias)
  2. Again in `softplus(dt + bias)` before discretization
- Model was trained with this double-bias behavior, so we must match it
- This fixes repetition issues from v0.2.22-0.2.23

### 0.2.23
- **CRITICAL FIX**: Wrong in_proj split order causing gibberish output
- Reference implementation uses: `[z(d_inner), x(d_xb), B(d_xb), C(d_inner), dt(dt_rank)]`
- Our code incorrectly had: `[z(d_inner), x(d_inner), B(d_xb), C(d_xb), dt(dt_rank)]`
- x is d_xb (needs repeat_kv expansion), C is d_inner (already full size)
- Fixed _prefill and _decode_step to handle x/C dimensions correctly

### 0.2.22
- **FIX**: Attempted to fix double bias (WRONG - model was trained with double bias)
- Removed redundant bias addition - this broke the model

### 0.2.21
- **FIX**: Dtype mismatch in rotary position embeddings
- Cast cos/sin to match q's dtype before applying rotation
- Fixes `RuntimeError: expected scalar type Float but found BFloat16` in Q×K matmul

### 0.2.20
- **FIX**: Dtype mismatch in attention matmul
- After softmax (computed in float32), convert to `v.dtype` instead of `q.dtype`
- Fixes `RuntimeError: expected scalar type Float but found BFloat16`

### 0.2.19
- **FIX**: Handle vLLM warmup where seq_len exceeds KV cache size
- During warmup/autotune, `max_num_batched_tokens=8192` but cache only holds 2048
- Skip KV caching when tokens don't fit, allowing warmup to complete

### 0.2.18
- Added extensive debug logging to diagnose attention layer shape issue
- Logs: input shape, batch_size, seq_len, Q/K/V shapes, rotary output, KV cache shapes

### 0.2.17
- Added debug logging in MHADecoderLayer to trace tensor shapes

### 0.2.16
- Fixed attention layer to handle vLLM's flattened 2D tensor format
- vLLM passes [total_tokens, hidden] but attention needs [batch, seq, hidden]
- Added automatic batch dimension handling in MHADecoderLayer

### 0.2.15
- Fixed attention layer KV cache shape mismatch
- Removed incorrect tensor transpositions in KV cache assignment

### 0.2.14
- Fixed `mamba_config.json` loading - removed `local_files_only=True` restriction
- Now properly downloads mamba_config.json from HuggingFace Hub if not cached
- Added more detailed logging for config loading

### 0.2.13
- **CRITICAL FIX**: Load `mamba_config.json` for `attn_layers`, `d_inner`, `d_xb`
- MambaInLlama models store Mamba-specific config in separate `mamba_config.json` file
- Main `config.json` has `model_type: "llama"` without Mamba params
- Fixed: Model was treating ALL layers as Mamba (attn_layers=[]) because config wasn't loaded
- Added better logging for weight loading diagnostics
- Attention layers at indices `[3, 8, 13, 18, 23, 27]` now properly recognized

### 0.2.12
- **CRITICAL FIX**: Corrected `d_xb` default to match qwerky-distill PR #81
- `d_xb = num_key_value_heads * head_dim` (GQA-style, e.g., 8×128=1024 for 8B)
- Fixed in_proj split: `[z(d_inner), x(d_inner), B(d_xb), C(d_xb), dt(dt_rank)]`
- Added repeat_kv expansion for C (same as B) in Mamba1 architecture
- Fixed head count: `num_heads = d_inner // d_state` after B/C expansion

### 0.2.11
- **CRITICAL FIX**: Changed `d_inner` default from `intermediate_size` to `hidden_size`
- MambaInLlama Mamba layers use `d_inner = hidden_size`, not `intermediate_size`
- Fixed `d_xb` default: `hidden_size // 16` (was `hidden_size // 4`)
- This fixes the shape mismatch for all Mamba layer weights (A_log, D, conv1d, dt_proj, in_proj, out_proj)

### 0.2.10
- Added debug logging to weight loading to diagnose parameter mapping issues
- Logs first 20 model params, first 20 checkpoint weights, and all skipped weights

### 0.2.9
- Fixed weight loading: split fused `mha.in_proj` into separate q/k/v projections
- Renamed `mha.out_proj` to `o_proj` for checkpoint compatibility
- Should now load all ~395 parameters instead of just 163

### 0.2.8
- Fixed dtype mismatch in SSM scan: `F.softplus`/`torch.exp` compute in float32, now cast back to original dtype
- This caused "expected BFloat16 but found Float" error in einsum

### 0.2.7
- Fixed tensor broadcasting bug in `_ssm_scan`: `A.unsqueeze(0).unsqueeze(-1)` -> `A.unsqueeze(0).unsqueeze(2)`
- This caused shape mismatch (8192 vs 16) during SSM discretization

### 0.2.6
- Added `embed_input_ids` method required by vLLM's `VllmModelForTextGeneration` interface
- This was the root cause of "This model does not support `--runner generate`" error

### 0.2.5
- Fixed vLLM runner detection: added `MambaInLlamaMambaForCausalLM` alias for HF config compatibility
- Added proper protocol inheritance (`HasInnerState`, `IsHybrid`) from `vllm.model_executor.models.interfaces`
- Fixed class variable type hints (`ClassVar[Literal[True]]`) for vLLM model inspection
- Simplified model registration code

### 0.2.4
- Complete architecture rewrite with explicit state cache management
- Separate prefill and decode paths for Mamba layers
- Grouped-head Mamba support (`num_xb_head`, `num_C_head`, `repeat_group`)
- Pure PyTorch SSM implementation (preparing for vLLM Triton op integration)

### 0.2.3
- Fixed `d_xb` default value computation in configuration
- Removed unsupported `device`/`dtype` kwargs from RMSNorm calls

### 0.2.2
- Fixed vLLM 0.14+ compatibility issues with Mamba ops API

### 0.2.1
- Updated README, removed SFT model reference

### 0.2.0
- Initial public release with vLLM plugin system integration

## License

Apache 2.0
