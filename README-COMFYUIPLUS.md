# ComfyUI Plus

ComfyUI Plus es un script de gestión que permite ejecutar ComfyUI y actualizar el fork desde el upstream con resumen automático generado por un LLM local.

## Requisitos

- Python 3.10+
- [rich](https://github.com/Textualize/rich) - Para la interfaz de menú:
  ```bash
  pip install rich
  ```
- [requests](https://docs.python-requests.org/) - Para comunicar con llama-server:
  ```bash
  pip install requests
  ```

## Estructura

```
ComfyUI/
├── comfyuiplus          # Wrapper script (invoca a comfyuiplus.py)
├── comfyuiplus.py       # Script principal con la lógica
└── README-COMFYUIPLUS.md # Este archivo
```

### comfyuiplus
Wrapper que importa y ejecuta el script principal. Es similar a la estructura de `opencodeplus` del proyecto opencode.

### comfyuiplus.py
Contiene toda la lógica:
- Menú interactivo con Rich
- Ejecución de ComfyUI
- Actualización del fork desde upstream
- Generación de resumen de cambios con llama-server

## Instalación

### 1. Instalar dependencias

```bash
pip install rich requests
```

### 2. Configurar llama-server

Asegúrate de tener llama-server corriendo en el puerto 9999:

```bash
# Puerto 9999: Model selector proxy (gestiona todos los modelos)
# Puerto 53204: Servidor con el modelo cargado
```

El script detecta automáticamente el modelo cargado y su contexto máximo para generar el resumen.

### 3. Agregar al PATH

Agrega el directorio del proyecto al PATH en tu `~/.zshrc`:

```bash
export PATH="/Users/atlantis/Documents/Desarrollo/ComfyUI:$PATH"
```

Luego recarga:

```bash
source ~/.zshrc
```

## Uso

### Ejecutar el menú

```bash
# Desde cualquier directorio (se usa el wrapper que importa el módulo)
comfyuiplus

# O desde el directorio del proyecto con el venv activo
source .venv/bin/activate
./comfyuiplus
```

Esto mostrará el menú principal:

```
╭──────────────────────────────── ComfyUI Plus ────────────────────────────────╮
│                                                                              │
│                                                                              │
│     1       Run ComfyUI - Iniciar ComfyUI                                    │
│     2       Update & Summary - Actualizar y generar resumen                  │
│                                                                              │
│                                                                              │
╰─────── v0.26.0 | Modelo: Qwopus3-6-35B-A3B-v1-Q5_K_M (262144 tokens) ────────╯

Choose [0/1/2] (0):
```

### Opciones

1. **Run ComfyUI** - Inicia ComfyUI en modo bloqueante (Ctrl+C para detener)
2. **Update & Summary** - Actualiza el fork desde upstream y genera un resumen de cambios con el LLM local
0. **Salir** - Cierra el script

### Update & Summary

Al seleccionar la opción 2, el script:

1. Detecta el modelo activo en llama-server y su contexto máximo
2. Verifica si hay cambios nuevos en upstream
3. Trae los cambios con `git fetch` y `git pull`
4. Instala/actualiza dependencias con `pip install -r requirements.txt`
5. Sincroniza la versión en `comfyui_version.py` con `pyproject.toml`
6. Verifica la sintaxis de `main.py` y `server.py` con `py_compile`
7. Envía los diffs al LLM para generar un resumen
8. Muestra el resumen generado con formato markdown

## Notas

- El script se ejecuta desde la raíz del repositorio ComfyUI
- No tiene timeout con llama-server (puede tardar varios minutos en generar el resumen)
- El wrapper permite importar `comfyuiplus.py` como módulo desde la ruta del script
