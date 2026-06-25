import torch
import logging

from comfy.cli_args import args

try:
    import comfy_kitchen as ck
    from comfy_kitchen.tensor import (
        QuantizedTensor,
        QuantizedLayout,
        TensorCoreFP8Layout as _CKFp8Layout,
        TensorCoreNVFP4Layout as _CKNvfp4Layout,
        register_layout_op,
        register_layout_class,
        get_layout_class,
    )
    _CK_AVAILABLE = True
    if torch.version.cuda is None:
        ck.registry.disable("cuda")
    else:
        cuda_version = tuple(map(int, str(torch.version.cuda).split('.')))
        if cuda_version < (13,):
            ck.registry.disable("cuda")
            logging.warning("WARNING: You need pytorch with cu130 or higher to use optimized CUDA operations.")

    if args.enable_triton_backend:
        try:
            import triton
            logging.info("Found triton %s. Enabling comfy-kitchen triton backend.", triton.__version__)
        except ImportError as e:
            logging.error(f"Failed to import triton, Error: {e}, the comfy-kitchen triton backend will not be available.")
            ck.registry.disable("triton")
    else:
        ck.registry.disable("triton")
    for k, v in ck.list_backends().items():
        logging.info(f"Found comfy_kitchen backend {k}: {v}")
    
    _CKFP8_LAYOUT = _CKFp8Layout
    original_dequantize = _CKFP8_LAYOUT.dequantize
    
    @classmethod
    def _ck_fp8_dequantize_patched(cls, qdata, params):
        if qdata.device.type == 'mps' or params.scale.device.type == 'mps':
            output_dtype = params.orig_dtype
            qdata_cpu = qdata.to(device='cpu')
            scale_cpu = params.scale.to(device='cpu')
            qdata_fp32 = qdata_cpu.to(dtype=torch.float32)
            scale_fp32 = scale_cpu.to(dtype=torch.float32)
            dequantized = qdata_fp32 * scale_fp32
            return dequantized.to(device='mps', dtype=output_dtype)
        return original_dequantize(qdata, params)
    
    _CKFP8_LAYOUT.dequantize = _ck_fp8_dequantize_patched
    
    _CKMXFP8_LAYOUT = None
    try:
        from comfy_kitchen.tensor import TensorCoreMXFP8Layout as _CKMxfp8Layout
        _CKMXFP8_LAYOUT = _CKMxfp8Layout
        original_mxfp8_dequantize = _CKMXFP8_LAYOUT.dequantize
        
        @classmethod
        def _ck_mxfp8_dequantize_patched(cls, qdata, params):
            if qdata.device.type == 'mps' or params.scale.device.type == 'mps':
                output_dtype = params.orig_dtype
                qdata_cpu = qdata.to(device='cpu')
                scale_cpu = params.scale.to(device='cpu')
                qdata_fp32 = qdata_cpu.to(dtype=torch.float32)
                scale_fp32 = scale_cpu.to(dtype=torch.float32)
                dequantized = qdata_fp32 * scale_fp32
                return dequantized.to(device='mps', dtype=output_dtype)
            return original_mxfp8_dequantize(qdata, params)
        
        _CKMXFP8_LAYOUT.dequantize = _ck_mxfp8_dequantize_patched
    except ImportError:
        pass
    
    _CKNVFP4_LAYOUT = _CKNvfp4Layout
    original_nvfp4_dequantize = _CKNVFP4_LAYOUT.dequantize
    
    @classmethod
    def _ck_nvfp4_dequantize_patched(cls, qdata, params):
        if qdata.device.type == 'mps' or params.scale.device.type == 'mps':
            output_dtype = params.orig_dtype
            qdata_cpu = qdata.to(device='cpu')
            scale_cpu = params.scale.to(device='cpu')
            qdata_fp32 = qdata_cpu.to(dtype=torch.float32)
            scale_fp32 = scale_cpu.to(dtype=torch.float32)
            dequantized = qdata_fp32 * scale_fp32
            return dequantized.to(device='mps', dtype=output_dtype)
        return original_nvfp4_dequantize(qdata, params)
    
    _CKNVFP4_LAYOUT.dequantize = _ck_nvfp4_dequantize_patched
    
    # MPS-safe quantization patches for eager backend
    if torch.version.cuda is None and not torch.backends.mps.is_available():
        # Only patch on non-CUDA, non-MPS systems (i.e., MPS systems)
        pass
    elif torch.backends.mps.is_available():
        try:
            from comfy_kitchen.backends.eager.quantization import quantize_per_tensor_fp8 as _ck_quantize_per_tensor_fp8
            
            def quantize_per_tensor_fp8_mps_safe(x, scale, output_type):
                """MPS-safe quantize_per_tensor_fp8 that returns float32 instead of float8."""
                F8_E4M3_MAX = 448.0
                F8_E5M2_MAX = 57344.0
                
                if output_type == torch.float8_e4m3fn:
                    lp_max = F8_E4M3_MAX
                elif output_type == torch.float8_e5m2:
                    lp_max = F8_E5M2_MAX
                else:
                    raise ValueError(f"Unsupported output_type: {output_type}")
                
                # Move to CPU and compute in float32
                x_cpu = x.to(device='cpu')
                scale_cpu = scale.to(device='cpu')
                temp = x_cpu * (1.0 / scale_cpu).to(x_cpu.dtype)
                temp = torch.clamp(temp, -lp_max, lp_max, out=temp)
                # Return as float32 instead of float8
                return temp.to(dtype=torch.float32)
            
            # Replace the function in the eager backend module
            import comfy_kitchen.backends.eager.quantization as ck_eager_quant
            ck_eager_quant.quantize_per_tensor_fp8 = quantize_per_tensor_fp8_mps_safe
            
            logging.info("Applied MPS-safe quantize_per_tensor_fp8 patch")
        except Exception as e:
            logging.warning(f"Failed to apply MPS quantize_per_tensor_fp8 patch: {e}")
except ImportError as e:
    logging.error(f"Failed to import comfy_kitchen, Error: {e}, fp8 and fp4 support will not be available.")
    _CK_AVAILABLE = False

    class QuantizedTensor:
        pass

    class _CKFp8Layout:
        pass

    class _CKNvfp4Layout:
        pass

    def register_layout_class(name, cls):
        pass

    def get_layout_class(name):
        return None

_CK_MXFP8_AVAILABLE = False
if _CK_AVAILABLE:
    try:
        from comfy_kitchen.tensor import TensorCoreMXFP8Layout as _CKMxfp8Layout
        _CK_MXFP8_AVAILABLE = True
    except ImportError:
        logging.warning("comfy_kitchen does not support MXFP8, please update comfy_kitchen.")

if not _CK_MXFP8_AVAILABLE:
    class _CKMxfp8Layout:
        pass

import comfy.float

# ==============================================================================
# FP8 Layouts with Comfy-Specific Extensions
# ==============================================================================

class _TensorCoreFP8LayoutBase(_CKFp8Layout):
    FP8_DTYPE = None  # Must be overridden in subclass

    @classmethod
    def quantize(cls, tensor, scale=None, stochastic_rounding=0, inplace_ops=False):
        if cls.FP8_DTYPE is None:
            raise NotImplementedError(f"{cls.__name__} must define FP8_DTYPE")

        orig_dtype = tensor.dtype
        orig_shape = tuple(tensor.shape)

        if isinstance(scale, str) and scale == "recalculate":
            scale = torch.amax(tensor.abs()).to(dtype=torch.float32) / torch.finfo(cls.FP8_DTYPE).max
            if tensor.dtype not in [torch.float32, torch.bfloat16]:  # Prevent scale from being too small
                tensor_info = torch.finfo(tensor.dtype)
                scale = (1.0 / torch.clamp((1.0 / scale), min=tensor_info.min, max=tensor_info.max))

        if scale is None:
            scale = torch.ones((), device=tensor.device, dtype=torch.float32)
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale, device=tensor.device, dtype=torch.float32)

        if stochastic_rounding > 0:
            if inplace_ops:
                tensor *= (1.0 / scale).to(tensor.dtype)
            else:
                tensor = tensor * (1.0 / scale).to(tensor.dtype)
            qdata = comfy.float.stochastic_rounding(tensor, dtype=cls.FP8_DTYPE, seed=stochastic_rounding)
        else:
            if comfy.float._is_mps_device(tensor.device):
                # MPS doesn't support float8, use float32 for quantization
                lp_max = torch.finfo(cls.FP8_DTYPE).max
                temp = tensor.to(device='cpu') * (1.0 / scale.to('cpu')).to(tensor.dtype)
                temp = torch.clamp(temp, -lp_max, lp_max, out=temp)
                qdata = temp.to(dtype=torch.float32)
            else:
                qdata = ck.quantize_per_tensor_fp8(tensor, scale, cls.FP8_DTYPE)

        params = cls.Params(scale=scale.float(), orig_dtype=orig_dtype, orig_shape=orig_shape)
        return qdata, params


class TensorCoreMXFP8Layout(_CKMxfp8Layout):
    @classmethod
    def quantize(cls, tensor, scale=None, stochastic_rounding=0, inplace_ops=False):
        if tensor.dim() != 2:
            raise ValueError(f"MXFP8 requires 2D tensor, got {tensor.dim()}D")

        orig_dtype = tensor.dtype
        orig_shape = tuple(tensor.shape)

        padded_shape = cls.get_padded_shape(orig_shape)
        needs_padding = padded_shape != orig_shape

        if stochastic_rounding > 0:
            qdata, block_scale = comfy.float.stochastic_round_quantize_mxfp8_by_block(tensor, pad_32x=needs_padding, seed=stochastic_rounding)
        else:
            if comfy.float._is_mps_device(tensor.device):
                # MPS doesn't support float8, use float32 for quantization
                F8_E4M3_MAX = 448.0
                E8M0_BIAS = 127
                BLOCK_SIZE = 32
                
                padded = tensor
                if needs_padding:
                    rows, cols = tensor.shape
                    padded_rows = ((rows + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
                    padded_cols = ((cols + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
                    if padded_rows != rows or padded_cols != cols:
                        padded = torch.nn.functional.pad(tensor, (0, padded_cols - cols, 0, padded_rows - rows))
                
                x_blocked = padded.reshape(padded.shape[0], -1, BLOCK_SIZE)
                max_abs = torch.amax(torch.abs(x_blocked), dim=-1)
                scale_needed = torch.clamp(max_abs.float() / F8_E4M3_MAX, min=2**(-127))
                log2_scale = torch.log2(scale_needed)
                exp_biased = torch.ceil(log2_scale).to(torch.int32) + E8M0_BIAS
                exp_biased = torch.clamp(exp_biased, 0, 254)
                block_scales_e8m0 = exp_biased.to(torch.uint8)
                block_scales_f32 = (block_scales_e8m0.to(torch.int32) << 23).view(torch.float32)
                zero_mask = (max_abs == 0)
                block_scales_f32 = torch.where(zero_mask, torch.ones_like(block_scales_f32), block_scales_f32)
                data_scaled = x_blocked.float() / block_scales_f32.unsqueeze(-1)
                data_scaled = torch.where(zero_mask.unsqueeze(-1), torch.zeros_like(data_scaled), data_scaled)
                data_scaled = torch.clamp(data_scaled, -F8_E4M3_MAX, F8_E4M3_MAX)
                data_fp8 = data_scaled.reshape(padded.shape).to(dtype=torch.float32)
                block_scales_e8m0 = torch.where(zero_mask, torch.zeros_like(block_scales_e8m0), block_scales_e8m0)
                qdata = data_fp8[:orig_shape[0], :orig_shape[1]]
                block_scale = comfy.float.to_blocked(block_scales_e8m0, flatten=False)
            else:
                qdata, block_scale = ck.quantize_mxfp8(tensor, pad_32x=needs_padding)

        params = cls.Params(
            scale=block_scale,
            orig_dtype=orig_dtype,
            orig_shape=orig_shape,
        )
        return qdata, params


class TensorCoreNVFP4Layout(_CKNvfp4Layout):
    @classmethod
    def quantize(cls, tensor, scale=None, stochastic_rounding=0, inplace_ops=False):
        if tensor.dim() != 2:
            raise ValueError(f"NVFP4 requires 2D tensor, got {tensor.dim()}D")

        orig_dtype = tensor.dtype
        orig_shape = tuple(tensor.shape)

        if scale is None or (isinstance(scale, str) and scale == "recalculate"):
            scale = torch.amax(tensor.abs()) / (ck.float_utils.F8_E4M3_MAX * ck.float_utils.F4_E2M1_MAX)

        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale)
        scale = scale.to(device=tensor.device, dtype=torch.float32)

        padded_shape = cls.get_padded_shape(orig_shape)
        needs_padding = padded_shape != orig_shape

        if stochastic_rounding > 0:
            qdata, block_scale = comfy.float.stochastic_round_quantize_nvfp4_by_block(tensor, scale, pad_16x=needs_padding, seed=stochastic_rounding)
        else:
            if comfy.float._is_mps_device(tensor.device):
                # MPS doesn't support float8, use float32 for quantization
                F4_E2M1_MAX = 6.0
                F8_E4M3_MAX = 448.0
                
                padded = tensor
                if needs_padding:
                    rows, cols = tensor.shape
                    padded_rows = ((rows + 15) // 16) * 16
                    padded_cols = ((cols + 15) // 16) * 16
                    if padded_rows != rows or padded_cols != cols:
                        padded = torch.nn.functional.pad(tensor, (0, padded_cols - cols, 0, padded_rows - rows))
                
                block_size = 16
                padded_shape_padded = padded.shape
                x_blocked = padded.reshape(padded_shape_padded[0], -1, block_size)
                max_abs = torch.amax(torch.abs(x_blocked), dim=-1)
                block_scale_fp8 = max_abs.to(torch.float32) / F4_E2M1_MAX
                scaled_block_scales = block_scale_fp8 / scale
                scaled_block_scales_fp8 = torch.clamp(scaled_block_scales, max=F8_E4M3_MAX)
                scaled_block_scales_fp32 = torch.clamp(scaled_block_scales_fp8, min=-F8_E4M3_MAX, max=F8_E4M3_MAX)
                total_scale = scale * scaled_block_scales_fp32
                zero_scale_mask = (total_scale == 0)
                total_scale_safe = torch.where(zero_scale_mask, torch.ones_like(total_scale), total_scale)
                data_scaled = x_blocked.float() / total_scale_safe.unsqueeze(-1)
                data_scaled = torch.where(zero_scale_mask.unsqueeze(-1), torch.zeros_like(data_scaled), data_scaled)
                data_scaled = torch.clamp(data_scaled, -F4_E2M1_MAX, F4_E2M1_MAX)
                data_fp4 = data_scaled.view(padded_shape_padded)
                data_fp4 = data_fp4[:orig_shape[0], :orig_shape[1]]
                out_scales = scaled_block_scales_fp8
                out_scales = out_scales[:padded.shape[0], :padded.shape[1] // block_size]
                block_scale = comfy.float.to_blocked(out_scales, flatten=False)
                qdata = data_fp4
            else:
                qdata, block_scale = ck.quantize_nvfp4(tensor, scale, pad_16x=needs_padding)

        params = cls.Params(
            scale=scale,
            orig_dtype=orig_dtype,
            orig_shape=orig_shape,
            block_scale=block_scale,
        )
        return qdata, params


class TensorCoreFP8E4M3Layout(_TensorCoreFP8LayoutBase):
    FP8_DTYPE = torch.float8_e4m3fn


class TensorCoreFP8E5M2Layout(_TensorCoreFP8LayoutBase):
    FP8_DTYPE = torch.float8_e5m2


# Backward compatibility alias - default to E4M3
TensorCoreFP8Layout = TensorCoreFP8E4M3Layout


# ==============================================================================
# Registry
# ==============================================================================

register_layout_class("TensorCoreFP8Layout", TensorCoreFP8Layout)
register_layout_class("TensorCoreFP8E4M3Layout", TensorCoreFP8E4M3Layout)
register_layout_class("TensorCoreFP8E5M2Layout", TensorCoreFP8E5M2Layout)
register_layout_class("TensorCoreNVFP4Layout", TensorCoreNVFP4Layout)
if _CK_MXFP8_AVAILABLE:
    register_layout_class("TensorCoreMXFP8Layout", TensorCoreMXFP8Layout)

QUANT_ALGOS = {
    "float8_e4m3fn": {
        "storage_t": torch.float8_e4m3fn,
        "parameters": {"weight_scale", "input_scale"},
        "comfy_tensor_layout": "TensorCoreFP8E4M3Layout",
    },
    "float8_e5m2": {
        "storage_t": torch.float8_e5m2,
        "parameters": {"weight_scale", "input_scale"},
        "comfy_tensor_layout": "TensorCoreFP8E5M2Layout",
    },
    "nvfp4": {
        "storage_t": torch.uint8,
        "parameters": {"weight_scale", "weight_scale_2", "input_scale"},
        "comfy_tensor_layout": "TensorCoreNVFP4Layout",
        "group_size": 16,
    },
}

if _CK_MXFP8_AVAILABLE:
    QUANT_ALGOS["mxfp8"] = {
        "storage_t": torch.float8_e4m3fn,
        "parameters": {"weight_scale", "input_scale"},
        "comfy_tensor_layout": "TensorCoreMXFP8Layout",
        "group_size": 32,
    }


# ==============================================================================
# Re-exports for backward compatibility
# ==============================================================================

__all__ = [
    "QuantizedTensor",
    "QuantizedLayout",
    "TensorCoreFP8Layout",
    "TensorCoreFP8E4M3Layout",
    "TensorCoreFP8E5M2Layout",
    "TensorCoreNVFP4Layout",
    "QUANT_ALGOS",
    "register_layout_op",
]
