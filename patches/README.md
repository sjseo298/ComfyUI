# Parches para ComfyUI-Manager

Este directorio contiene parches (patches) que se aplican a ComfyUI-Manager para mantener la compatibilidad y funcionalidad necesaria.

## Parches disponibles

### comfyui-manager-hf-token.patch

Aplica los cambios necesarios para soportar descarga de modelos gated de HuggingFace y detección de modelos instalados en paths personalizados:

1. **version_code = [4, 2, 2]** en `glob/manager_core.py`
   - Evita el error "ComfyUI-Manager requiere versión 4.2.1 o superior"
   - El Manager verifica la versión desde la web y muestra error si es menor

2. **HF_TOKEN** en `glob/manager_downloader.py`
   - Agrega `HF_TOKEN = os.getenv('HF_TOKEN')` al inicio
   - En `download_url()`: retry con header Authorization si `torchvision_download_url` falla con 401 Unauthorized en URLs de HuggingFace

3. **is_exists** en `glob/manager_server.py` - chequeo de download_model_base
   - Agrega chequeo de existencia de modelos en `download_model_base` y `download_model_base/etc/`
   - Permite detectar modelos instalados en el Samsung2T drive

4. **process_model_phase** en `glob/manager_server.py` - custom model types
   - Cuando `model_dir_name is None` (tipos de modelo custom como stage_a, effnet_encoder),
     chequea existencia en `download_model_base` y `download_model_base/etc/`
   - Evita marcar modelos como "not_installed" cuando están en el Samsung2T drive

5. **onQueueCompleted** en `js/model-manager.js` - corrección de UI post-instalación
   - Fix: `result.length == 0` → `Object.keys(result).length == 0` (result es dict, no array)
   - Fix: Actualiza `item.installed = "True"` antes de llamar `updateCell` para instalación exitosa

## Cómo aplicar parches

### Manual
```bash
cd custom_nodes/comfyui-manager
git apply --reject /Users/atlantis/Documents/Desarrollo/ComfyUI/patches/comfyui-manager-hf-token.patch
```

### Automático
Después de actualizar el Manager con `git pull`:
```bash
./post-update.sh
```

## Notas para LLMs

- La variable `HF_TOKEN` se setea en `.env` y se carga con `python-dotenv` antes de ejecutar `main.py`
- El Manager lee `HF_TOKEN` del entorno para descargar modelos gated de HuggingFace
- Si `manager_downloader.py` se modifica en el futuro, verificar que `HF_TOKEN` siga disponible
- Si `manager_core.py` se modifica, verificar que `version_code` no se sobrescriba
- `download_model_base` en `extra_model_paths.yaml` debe apuntar a /Volumes/Samsung2T/ComfyUI/models/
- Los modelos custom (stage_a, effnet_encoder, etc.) se guardan en `download_model_base/etc/`
- El `is_exists` function debe chequear ambos paths: `download_model_base` y `download_model_base/etc/`
