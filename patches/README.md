# Patches de ComfyUI

## mps_fp8_fix.diff

### Problema

PyTorch en MPS (Apple Silicon) no soporta el dtype `Float8_e4m3fn`. Cuando `comfy_kitchen` intenta hacer la descuantización de tensores FP8, el método `dequantize` mueve los datos al dispositivo del tensor de escala (`params.scale`). Si `params.scale` está en MPS mientras `qdata` (los weights) está en CPU, PyTorch intenta mover el tensor a MPS y falla con:

```
TypeError: Trying to convert Float8_e4m3fn to the MPS backend but it does not have support for that dtype.
```

### Solución

Parchear los métodos `dequantize` de los layouts de `comfy_kitchen` para detectar cuando cualquiera de los dos tensores (qdata o scale) está en MPS, y en ese caso:

1. Mover ambos tensores a CPU
2. Convertir a float32
3. Hacer la operación de descuantización (multiplicación) en CPU
4. Retornar el resultado al dispositivo original (MPS) en el dtype original

### Archivos parcheados

- `comfy/quant_ops.py` — Se parchean los métodos `dequantize` de:
  - `TensorCoreFP8Layout` (FP8 E4M3)
  - `TensorCoreMXFP8Layout` (MXFP8, con try/except para cuando no está disponible)
  - `TensorCoreNVFP4Layout` (NVFP4)
- `comfyuiplus.py` — Se añade la variable de entorno `PYTORCH_ENABLE_MPS_FALLBACK=1` al ejecutar ComfyUI

### Rendimiento

La descuantización se ejecuta en **CPU**, no en MPS. Es un fallback necesario porque MPS no soporta el dtype FP8. La GPU (MPS) solo recibe el resultado ya descuantizado.

### Aplicar el parche

```bash
git apply patches/mps_fp8_fix.diff
```

### Desaplicar el parche

```bash
git apply -R patches/mps_fp8_fix.diff
```
