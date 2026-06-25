# Plan: Fix MPS dequantization segfault

## Problem
The MPS dequantization patches crash with segfault because they try to return tensors to MPS with FP8 dtypes:
```python
return dequantized.to(device='mps', dtype=output_dtype)
```

If `output_dtype` is `float8_e4m3fn` or `float8_e5m2`, MPS doesn't support it and segfaults silently.

## Solution
Modify the 3 dequantization patches to return float32 on CPU when the output_dtype is FP8:

### 1. `_ck_fp8_dequantize_patched` (line 42-51)
Change line 50 from:
```python
return dequantized.to(device='mps', dtype=output_dtype)
```
To:
```python
if output_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
    return dequantized.to(device='cpu', dtype=torch.float32)
return dequantized.to(device='mps', dtype=output_dtype)
```

### 2. `_ck_mxfp8_dequantize_patched` (line 62-71)
Change line 70 from:
```python
return dequantized.to(device='mps', dtype=output_dtype)
```
To:
```python
if output_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
    return dequantized.to(device='cpu', dtype=torch.float32)
return dequantized.to(device='mps', dtype=output_dtype)
```

### 3. `_ck_nvfp4_dequantize_patched` (line 81-90)
Change line 89 from:
```python
return dequantized.to(device='mps', dtype=output_dtype)
```
To:
```python
if output_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
    return dequantized.to(device='cpu', dtype=torch.float32)
return dequantized.to(device='mps', dtype=output_dtype)
```

### 4. Add hint in comfyuiplus.py
Add a logging message at startup to indicate CUDA is not available and FP8 operations will use CPU fallback.

## Files to modify
- `patches/comfy.quant_ops.py` - 3 dequantization patches
- `patches/comfyuiplus.py` - Add startup hint
