#!/usr/bin/env python3
"""
Test de dequantización FP8 en MPS con scale en MPS.
Reproduce el error del traceback y valida el fix.
"""
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"Has MPS: {torch.backends.mps.is_built()}")
print(f"MPS available: {torch.backends.mps.is_available()}")

try:
    import comfy_kitchen as ck
    print(f"comfy_kitchen version: {ck.__version__}")
except Exception as e:
    print(f"Failed to import comfy_kitchen: {e}")
    exit(1)

try:
    from comfy_kitchen.tensor import (
        QuantizedTensor,
        TensorCoreFP8Layout as CKFP8Layout,
        TensorCoreNVFP4Layout as CKNVFP4Layout,
    )
    from comfy_kitchen.tensor.base import dequantize_args
    print("Successfully imported comfy_kitchen tensors")
except ImportError as e:
    print(f"Failed to import layouts: {e}")
    exit(1)

# Apply the patch BEFORE creating quantized tensors
import comfy.quant_ops as quant_ops

def test_fp8_quantize_cpu_dequant_mps_scale():
    """
    Test: Quantize on CPU, then create QuantizedTensor with scale moved to MPS.
    This is the exact scenario that causes the error.
    Uses QuantizedTensor.dequantize() which is what the real code path uses.
    """
    print("\n=== Test: Quantize CPU, dequant via QuantizedTensor with MPS scale ===")
    
    try:
        # Create and quantize on CPU
        data_cpu = torch.randn(16, 32, dtype=torch.bfloat16, device='cpu')
        qdata, params = CKFP8Layout.quantize(data_cpu)
        
        # Move scale to MPS
        params_scale_mps = params.scale.to(device='mps')
        
        # Create QuantizedTensor with MPS scale
        qt = QuantizedTensor(qdata, "TensorCoreFP8Layout", 
                            CKFP8Layout.Params(scale=params_scale_mps,
                                              orig_dtype=params.orig_dtype,
                                              orig_shape=params.orig_shape))
        
        # This is the exact call path from the traceback:
        # dequantize_args -> QuantizedTensor.dequantize() -> layout_cls.dequantize()
        dequantized = qt.dequantize()
        
        print(f"SUCCESS: Dequantized shape: {dequantized.shape}, device: {dequantized.device}, dtype: {dequantized.dtype}")
        
        error = (data_cpu - dequantized.to('cpu')).abs().max()
        print(f"Max error vs original: {error}")
        
        return True
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fp8_dequantize_args_mps_scale():
    """
    Test: QuantizedTensor with MPS scale, dequantized via dequantize_args().
    This matches the exact error path from the traceback.
    """
    print("\n=== Test: QuantizedTensor with MPS scale via dequantize_args ===")
    
    try:
        data_cpu = torch.randn(16, 32, dtype=torch.bfloat16, device='cpu')
        qdata, params = CKFP8Layout.quantize(data_cpu)
        
        # Move scale to MPS
        params_scale_mps = params.scale.to(device='mps')
        qt = QuantizedTensor(qdata, "TensorCoreFP8Layout",
                            CKFP8Layout.Params(scale=params_scale_mps,
                                              orig_dtype=params.orig_dtype,
                                              orig_shape=params.orig_shape))
        
        # This is the exact path from the error:
        # _handle_fp8_linear -> dequantize_args((input_tensor, weight, bias))
        dequantized = dequantize_args(qt)
        
        print(f"SUCCESS: Dequantized shape: {dequantized.shape}, device: {dequantized.device}, dtype: {dequantized.dtype}")
        
        error = (data_cpu - dequantized.to('cpu')).abs().max()
        print(f"Max error vs original: {error}")
        
        return True
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fp8_dequant_direct_mps_scale():
    """
    Test: Direct layout dequantize call with MPS scale.
    This is the low-level call that fails without the patch.
    """
    print("\n=== Test: Direct layout dequantize with MPS scale ===")
    
    try:
        data_cpu = torch.randn(16, 32, dtype=torch.bfloat16, device='cpu')
        qdata, params = CKFP8Layout.quantize(data_cpu)
        
        params_scale_mps = params.scale.to(device='mps')
        new_params = CKFP8Layout.Params(scale=params_scale_mps,
                                        orig_dtype=params.orig_dtype,
                                        orig_shape=params.orig_shape)
        
        dequantized = CKFP8Layout.dequantize(qdata, new_params)
        
        print(f"SUCCESS: Dequantized shape: {dequantized.shape}, device: {dequantized.device}, dtype: {dequantized.dtype}")
        
        error = (data_cpu - dequantized.to('cpu')).abs().max()
        print(f"Max error vs original: {error}")
        
        return True
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fp8_cpu_only_baseline():
    """
    Baseline: Everything on CPU. Should work without any patch.
    """
    print("\n=== Test: FP8 CPU only (baseline) ===")
    
    try:
        data_cpu = torch.randn(16, 32, dtype=torch.bfloat16, device='cpu')
        qdata, params = CKFP8Layout.quantize(data_cpu)
        dequantized = CKFP8Layout.dequantize(qdata, params)
        
        print(f"SUCCESS: Dequantized shape: {dequantized.shape}, device: {dequantized.device}, dtype: {dequantized.dtype}")
        
        error = (data_cpu - dequantized).abs().max()
        print(f"Max error vs original: {error}")
        
        return True
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_nvfp4_quantize_cpu_dequant_mps_scale():
    """
    Test: NVFP4 quantize CPU, dequant via QuantizedTensor with MPS scale.
    """
    print("\n=== Test: NVFP4 Quantize CPU, dequant via QuantizedTensor with MPS scale ===")
    
    try:
        # NVFP4 requires 2D tensors
        data_cpu = torch.randn(16, 32, dtype=torch.bfloat16, device='cpu')
        qdata, params = CKNVFP4Layout.quantize(data_cpu)
        
        # Move scale to MPS
        params_scale_mps = params.scale.to(device='mps')
        new_params = CKNVFP4Layout.Params(
            scale=params_scale_mps,
            orig_dtype=params.orig_dtype,
            orig_shape=params.orig_shape,
            block_scale=params.block_scale,
        )
        
        qt = QuantizedTensor(qdata, "TensorCoreNVFP4Layout", new_params)
        dequantized = qt.dequantize()
        
        print(f"SUCCESS: Dequantized shape: {dequantized.shape}, device: {dequantized.device}, dtype: {dequantized.dtype}")
        
        # NVFP4 may have different output shape due to padding
        dequant_cpu = dequantized.to('cpu')
        min_h, min_w = min(data_cpu.shape[0], dequant_cpu.shape[0]), min(data_cpu.shape[1], dequant_cpu.shape[1])
        error = (data_cpu[:min_h, :min_w] - dequant_cpu[:min_h, :min_w]).abs().max()
        print(f"Max error vs original (overlap): {error}")
        
        return True
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    results = []
    
    results.append(("FP8 CPU baseline", test_fp8_cpu_only_baseline()))
    results.append(("FP8 QuantizedTensor MPS scale", test_fp8_quantize_cpu_dequant_mps_scale()))
    results.append(("FP8 dequantize_args MPS scale", test_fp8_dequantize_args_mps_scale()))
    results.append(("FP8 direct dequant MPS scale", test_fp8_dequant_direct_mps_scale()))
    results.append(("NVFP4 QuantizedTensor MPS scale", test_nvfp4_quantize_cpu_dequant_mps_scale()))
    
    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
    
    all_passed = all(r[1] for r in results)
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
