#!/usr/bin/env python3
"""
Test de cuantización FP8/MXFP8/NVFP4 en MPS con stochastic rounding.
Reproduce el error del traceback y valida el fix para cuantización.
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
    import comfy.float
    import comfy.quant_ops
    print("Successfully imported comfy float and quant_ops modules")
except ImportError as e:
    print(f"Failed to import comfy modules: {e}")
    exit(1)


def test_fp8_stochastic_quantization_cpu():
    """Test: FP8 stochastic quantization on CPU (baseline)."""
    print("\n=== Test: FP8 stochastic quantization CPU (baseline) ===")
    
    try:
        data_cpu = torch.randn(16, 32, dtype=torch.bfloat16, device='cpu')
        scale = torch.ones((), device='cpu', dtype=torch.float32)
        
        layout = comfy.quant_ops.TensorCoreFP8E4M3Layout
        qdata, params = layout.quantize(data_cpu, scale=scale, stochastic_rounding=42, inplace_ops=False)
        
        print(f"SUCCESS: Quantized shape: {qdata.shape}, device: {qdata.device}, dtype: {qdata.dtype}")
        return True
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fp8_stochastic_quantization_mps():
    """Test: FP8 stochastic quantization on MPS (requires MPS-safe patch)."""
    print("\n=== Test: FP8 stochastic quantization MPS ===")
    
    if not torch.backends.mps.is_available():
        print("SKIPPED: MPS not available")
        return True
    
    try:
        data_mps = torch.randn(16, 32, dtype=torch.bfloat16, device='mps')
        scale = torch.ones((), device='mps', dtype=torch.float32)
        
        layout = comfy.quant_ops.TensorCoreFP8E4M3Layout
        qdata, params = layout.quantize(data_mps, scale=scale, stochastic_rounding=42, inplace_ops=False)
        
        print(f"SUCCESS: Quantized shape: {qdata.shape}, device: {qdata.device}, dtype: {qdata.dtype}")
        
        # MPS-safe version should return float32
        if qdata.dtype == torch.float32:
            print("CORRECT: MPS-safe version returns float32")
        
        return True
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fp8_non_stochastic_quantization_cpu():
    """Test: FP8 non-stochastic quantization on CPU (baseline)."""
    print("\n=== Test: FP8 non-stochastic quantization CPU (baseline) ===")
    
    try:
        data_cpu = torch.randn(16, 32, dtype=torch.bfloat16, device='cpu')
        scale = torch.ones((), device='cpu', dtype=torch.float32)
        
        layout = comfy.quant_ops.TensorCoreFP8E4M3Layout
        qdata, params = layout.quantize(data_cpu, scale=scale, stochastic_rounding=0, inplace_ops=False)
        
        print(f"SUCCESS: Quantized shape: {qdata.shape}, device: {qdata.device}, dtype: {qdata.dtype}")
        return True
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fp8_non_stochastic_quantization_mps():
    """Test: FP8 non-stochastic quantization on MPS (requires MPS-safe patch)."""
    print("\n=== Test: FP8 non-stochastic quantization MPS ===")
    
    if not torch.backends.mps.is_available():
        print("SKIPPED: MPS not available")
        return True
    
    try:
        data_mps = torch.randn(16, 32, dtype=torch.bfloat16, device='mps')
        scale = torch.ones((), device='mps', dtype=torch.float32)
        
        layout = comfy.quant_ops.TensorCoreFP8E4M3Layout
        qdata, params = layout.quantize(data_mps, scale=scale, stochastic_rounding=0, inplace_ops=False)
        
        print(f"SUCCESS: Quantized shape: {qdata.shape}, device: {qdata.device}, dtype: {qdata.dtype}")
        
        # MPS-safe version should return float32
        if qdata.dtype == torch.float32:
            print("CORRECT: MPS-safe version returns float32")
        
        return True
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fp8_dequantization_cpu():
    """Test: FP8 dequantization on CPU (baseline)."""
    print("\n=== Test: FP8 dequantization CPU (baseline) ===")
    
    try:
        data_cpu = torch.randn(16, 32, dtype=torch.bfloat16, device='cpu')
        scale = torch.ones((), device='cpu', dtype=torch.float32)
        
        layout = comfy.quant_ops.TensorCoreFP8E4M3Layout
        qdata, params = layout.quantize(data_cpu, scale=scale, stochastic_rounding=0, inplace_ops=False)
        dequantized = layout.dequantize(qdata, params)
        
        print(f"SUCCESS: Dequantized shape: {dequantized.shape}, device: {dequantized.device}, dtype: {dequantized.dtype}")
        return True
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fp8_dequantization_mps_scale():
    """Test: FP8 dequantization with scale on MPS (requires MPS-safe patch)."""
    print("\n=== Test: FP8 dequantization with MPS scale ===")
    
    if not torch.backends.mps.is_available():
        print("SKIPPED: MPS not available")
        return True
    
    try:
        data_cpu = torch.randn(16, 32, dtype=torch.bfloat16, device='cpu')
        scale = torch.ones((), device='cpu', dtype=torch.float32)
        
        layout = comfy.quant_ops.TensorCoreFP8E4M3Layout
        qdata, params = layout.quantize(data_cpu, scale=scale, stochastic_rounding=0, inplace_ops=False)
        
        # Move scale to MPS
        params_scale_mps = params.scale.to(device='mps')
        new_params = layout.Params(scale=params_scale_mps, orig_dtype=params.orig_dtype, orig_shape=params.orig_shape)
        
        dequantized = layout.dequantize(qdata, new_params)
        print(f"SUCCESS: Dequantized shape: {dequantized.shape}, device: {dequantized.device}, dtype: {dequantized.dtype}")
        return True
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_nvfp4_quantization_mps():
    """Test: NVFP4 quantization on MPS (requires MPS-safe patch)."""
    print("\n=== Test: NVFP4 quantization MPS ===")
    
    if not torch.backends.mps.is_available():
        print("SKIPPED: MPS not available")
        return True
    
    try:
        data_mps = torch.randn(16, 32, dtype=torch.bfloat16, device='mps')
        scale = torch.amax(data_mps.abs()) / (1.0 * 6.0)
        
        layout = comfy.quant_ops.TensorCoreNVFP4Layout
        qdata, params = layout.quantize(data_mps, scale=scale, stochastic_rounding=0, inplace_ops=False)
        
        print(f"SUCCESS: NVFP4 quantized shape: {qdata.shape}, device: {qdata.device}, dtype: {qdata.dtype}")
        return True
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mxfp8_quantization_mps():
    """Test: MXFP8 quantization on MPS (requires MPS-safe patch)."""
    print("\n=== Test: MXFP8 quantization MPS ===")
    
    if not torch.backends.mps.is_available():
        print("SKIPPED: MPS not available")
        return True
    
    if not comfy.quant_ops._CK_MXFP8_AVAILABLE:
        print("SKIPPED: MXFP8 not available on MPS (requires CUDA)")
        return True
    
    try:
        data_mps = torch.randn(32, 64, dtype=torch.bfloat16, device='mps')
        
        layout = comfy.quant_ops.TensorCoreMXFP8Layout
        qdata, params = layout.quantize(data_mps, scale=None, stochastic_rounding=0, inplace_ops=False)
        
        print(f"SUCCESS: MXFP8 quantized shape: {qdata.shape}, device: {qdata.device}, dtype: {qdata.dtype}")
        return True
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    results = []
    
    results.append(("FP8 stochastic quantization CPU", test_fp8_stochastic_quantization_cpu()))
    results.append(("FP8 stochastic quantization MPS", test_fp8_stochastic_quantization_mps()))
    results.append(("FP8 non-stochastic quantization CPU", test_fp8_non_stochastic_quantization_cpu()))
    results.append(("FP8 non-stochastic quantization MPS", test_fp8_non_stochastic_quantization_mps()))
    results.append(("FP8 dequantization CPU", test_fp8_dequantization_cpu()))
    results.append(("FP8 dequantization MPS scale", test_fp8_dequantization_mps_scale()))
    results.append(("NVFP4 quantization MPS", test_nvfp4_quantization_mps()))
    results.append(("MXFP8 quantization MPS", test_mxfp8_quantization_mps()))
    
    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
    
    all_passed = all(r[1] for r in results)
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
