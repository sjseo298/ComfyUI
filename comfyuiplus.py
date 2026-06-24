#!/usr/bin/env python3

# ComfyUI Plus - Script de gestión de ComfyUI con resumen LLM

import os
import sys
import json
import shutil
import subprocess
import re

# Try to import dependencies, install them if not available
try:
    import requests
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "requests"], 
                   capture_output=True, check=True)
    import requests

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.prompt import Prompt
    from rich.markdown import Markdown
    from rich.table import Table
    from rich import box
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "rich"], 
                   capture_output=True, check=True)
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.prompt import Prompt
    from rich.markdown import Markdown
    from rich.table import Table
    from rich import box

console = Console()
COMFYUI_DIR = os.path.dirname(os.path.abspath(__file__))
LLAMA_SERVER_URL = "http://127.0.0.1:9999"
COMFYUI_PID_FILE = os.path.join(COMFYUI_DIR, ".comfyui.pid")
VENV_DIR = os.path.join(COMFYUI_DIR, ".venv")
MANAGER_DIR = os.path.join(COMFYUI_DIR, "custom_nodes", "comfyui-manager")


def get_required_python_version():
    """Obtener la versión de Python requerida desde pyproject.toml."""
    pyproject_path = os.path.join(COMFYUI_DIR, "pyproject.toml")
    if not os.path.exists(pyproject_path):
        return None
    try:
        with open(pyproject_path) as f:
            content = f.read()
        match = re.search(r'requires-python\s*=\s*">=([0-9]+\.[0-9]+)"', content)
        if match:
            return match.group(1)
    except:
        pass
    return None


def find_python_executable(required_version):
    """Buscar un ejecutable de Python compatible con la versión requerida."""
    if required_version is None:
        required_version = "3.10"
    required_major, required_minor = map(int, required_version.split("."))

    # Lista de candidatos a buscar
    candidates = [
        "/opt/homebrew/bin/python3.13",
        "/opt/homebrew/bin/python3.12",
        "/opt/homebrew/bin/python3.11",
        "/opt/homebrew/bin/python3.10",
        "/usr/local/bin/python3.13",
        "/usr/local/bin/python3.12",
        "/usr/local/bin/python3.11",
        "/usr/local/bin/python3.10",
    ]

    # También buscar en el PATH
    for candidate in candidates:
        if os.path.exists(candidate):
            major_minor = subprocess.run(
                [candidate, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
                capture_output=True, text=True
            ).stdout.strip()
            if major_minor:
                major, minor = map(int, major_minor.split("."))
                if major >= required_major and minor >= required_minor:
                    return candidate

    # Buscar en el PATH con shutil.which
    for version in ["python3.13", "python3.12", "python3.11", "python3.10"]:
        path = shutil.which(version)
        if path:
            major_minor = subprocess.run(
                [path, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
                capture_output=True, text=True
            ).stdout.strip()
            if major_minor:
                major, minor = map(int, major_minor.split("."))
                if major >= required_major and minor >= required_minor:
                    return path

    return None


def get_venv_python_version():
    """Obtener la versión de Python que tiene el venv actual."""
    venv_python = os.path.join(VENV_DIR, "bin", "python3")
    if not os.path.exists(venv_python):
        venv_python = os.path.join(VENV_DIR, "bin", "python")
    if not os.path.exists(venv_python):
        return None
    try:
        result = subprocess.run(
            [venv_python, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
            capture_output=True, text=True
        )
        return result.stdout.strip()
    except:
        return None


def create_venv(python_exec):
    """Crear un venv con el ejecutable de Python especificado."""
    console.print(f"\n[bold blue]Creando venv con {python_exec}...[/]")
    try:
        subprocess.run(
            [python_exec, "-m", "venv", VENV_DIR],
            capture_output=True, check=True
        )
        console.print("[green]✓ Venv creado[/]")
        return True
    except subprocess.CalledProcessError as e:
        console.print(f"[red]✗ Error al crear venv: {e.stderr.decode().strip()}[/]")
        return False


def recreate_venv(new_python_exec):
    """Borrar el venv actual y crear uno nuevo con dependencias."""
    console.print(f"\n[bold yellow]⚠ Venv desactualizado. Recreando con {new_python_exec}...[/]")
    try:
        shutil.rmtree(VENV_DIR)
        if not create_venv(new_python_exec):
            return False
        # Instalar dependencias automáticamente
        console.print(f"[bold]Instalando dependencias de ComfyUI...[/]")
        venv_python = os.path.join(VENV_DIR, "bin", "python3")
        if not os.path.exists(venv_python):
            venv_python = os.path.join(VENV_DIR, "bin", "python")
        result = subprocess.run([venv_python, "-m", "pip", "install", "-r", "requirements.txt"], 
                      capture_output=True, text=True, cwd=COMFYUI_DIR)
        if result.returncode != 0:
            console.print(f"[red]✗ Error instalando dependencias: {result.stderr}[/]")
            return False
        console.print("[green]✓ Venv recreado y dependencias instaladas[/]")
        return True
    except Exception as e:
        console.print(f"[red]✗ Error al recrear venv: {e}[/]")
        return False

console = Console()
COMFYUI_DIR = os.path.dirname(os.path.abspath(__file__))
LLAMA_SERVER_URL = "http://127.0.0.1:9999"
COMFYUI_PID_FILE = os.path.join(COMFYUI_DIR, ".comfyui.pid")


def get_loaded_model():
    """Obtener el modelo cargado en llama-server"""
    try:
        response = requests.get(f"{LLAMA_SERVER_URL}/v1/models", timeout=5)
        if response.status_code != 200:
            return "ERROR", 0
        
        data = response.json()
        loaded = [m for m in data.get("data", []) if m.get("status", {}).get("value") == "loaded"]
        
        if not loaded:
            return "ERROR", 0
        
        model_id = loaded[0]["id"]
        max_tokens = loaded[0].get("meta", {}).get("n_ctx", 0)
        
        return model_id, max_tokens
    except Exception:
        return "ERROR", 0


def get_llm_summary(model_id, max_tokens, prompt):
    """Llamar a llama-server para obtener resumen"""
    if not prompt:
        return "No hay cambios para resumir."
    
    try:
        response = requests.post(
            f"{LLAMA_SERVER_URL}/v1/chat/completions",
            json={
                "model": model_id,
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "messages": [
                    {
                        "role": "user",
                        "content": f"Resume los siguientes cambios de ComfyUI de forma concisa en español:\n\n{prompt}"
                    }
                ]
            },
            timeout=600  # Sin timeout - 10 minutos máximo
        )
        
        if response.status_code != 200:
            return "ERROR: No se pudo conectar con llama-server"
        
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        if not content:
            return "No se pudo generar el resumen (respuesta vacía del LLM)"
        
        return content
    except Exception as e:
        return f"ERROR al conectar con llama-server: {e}"


def is_comfyui_running():
    """Verificar si ComfyUI está corriendo"""
    if os.path.exists(COMFYUI_PID_FILE):
        try:
            with open(COMFYUI_PID_FILE) as f:
                pid = int(f.read().strip())
            os.kill(pid, 0)
            return True
        except (ValueError, ProcessLookupError, PermissionError):
            return False
    return False


def update_comfyui_manager():
    """Actualizar ComfyUI-Manager"""
    if not os.path.exists(MANAGER_DIR):
        console.print(Panel("[yellow]ComfyUI-Manager no instalado. Instalando...[/yellow]", border_style="yellow"))
        console.print(f"[dim]Clonando en {MANAGER_DIR}[/dim]")
        result = subprocess.run(
            ["git", "clone", "https://github.com/ltdrdata/ComfyUI-Manager", "comfyui-manager"],
            cwd=os.path.join(COMFYUI_DIR, "custom_nodes"),
            capture_output=True, text=True
        )
        if result.returncode != 0:
            console.print(f"[red]✗ Error al clonar: {result.stderr}[/]")
            return False
    else:
        console.print(Panel("[cyan]Actualizando ComfyUI-Manager...[/cyan]", border_style="cyan"))
        result = subprocess.run(
            ["git", "pull", "origin", "main"],
            cwd=MANAGER_DIR,
            capture_output=True, text=True
        )
        if result.returncode != 0:
            console.print(f"[yellow]⚠ No se pudo actualizar ComfyUI-Manager: {result.stderr}[/]")
        else:
            console.print("[green]✓ ComfyUI-Manager actualizado[/]")
    return True


def run_comfyui():
    """Ejecutar ComfyUI"""
    if is_comfyui_running():
        console.print(Panel("[bold yellow]ComfyUI ya se está ejecutando[/bold yellow]", border_style="yellow"))
        return
    
    # Verificar y actualizar ComfyUI-Manager antes de iniciar
    console.print(Panel("[cyan]Verificando ComfyUI-Manager...[/cyan]", border_style="cyan"))
    update_comfyui_manager()
    
    console.print(Panel.fit(
        "[bold green]Iniciando ComfyUI...[/bold green]\n"
        "Presiona Ctrl+C para detener",
        border_style="green",
        subtitle="ComfyUI"
    ))
    
    # Usar Python del venv si existe
    python_exec = None
    venv_python = os.path.join(VENV_DIR, "bin", "python3")
    if os.path.exists(venv_python):
        python_exec = venv_python
    else:
        required_version = get_required_python_version()
        python_exec = find_python_executable(required_version)
    
    if python_exec is None:
        console.print("[red]✗ No se encontró Python compatible. Instala Python 3.10+[/]")
        return
    
    with open(COMFYUI_PID_FILE, "w") as f:
        f.write(str(os.getpid()))
    
    # Load .env to set environment variables (e.g., HF_TOKEN for HuggingFace)
    try:
        from dotenv import load_dotenv
        env_path = os.path.join(COMFYUI_DIR, ".env")
        if os.path.exists(env_path):
            load_dotenv(env_path)
    except ImportError:
        pass
    
    # HuggingFace integration with ComfyUI-Manager:
    # - The Manager (custom_nodes/comfyui-manager/glob/manager_downloader.py) reads HF_TOKEN from env
    # - When downloading gated models (401 Unauthorized), it retries with Authorization header
    # - The token is set in .env and loaded via python-dotenv before main.py runs
    # - HfApi() in download_repo_in_bytes() automatically uses the token from env
    #
    # IMPORTANT: If manager_downloader.py is modified to change how HF_TOKEN is used,
    # ensure the token is still available via os.getenv('HF_TOKEN') before running main.py
    #
    # Dependencies: python-dotenv (installed in .venv)
    # Config: .env file with HF_TOKEN=hf_xxx, added to .gitignore
    
    try:
        subprocess.run([python_exec, "main.py", "--enable-manager"], cwd=COMFYUI_DIR)
    except KeyboardInterrupt:
        pass
    finally:
        if os.path.exists(COMFYUI_PID_FILE):
            os.remove(COMFYUI_PID_FILE)


def update_and_summary():
    """Actualizar repo y generar resumen"""
    # Verificar y gestionar el venv
    with Progress(SpinnerColumn(), TextColumn("[bold blue]{task.description}[/]"), console=console) as progress:
        task = progress.add_task("Verificando entorno Python...", total=None)
        
        required_version = get_required_python_version()
        console.print(f"\n[bold]Requisito Python:[/] {required_version or 'No especificado'}")
        
        current_venv_version = get_venv_python_version()
        if current_venv_version:
            console.print(f"[dim]Venv actual: {current_venv_version}[/dim]")
        
        # Si no existe el venv, o tiene una versión menor a la requerida, recrearlo
        if current_venv_version is None:
            # No existe venv, crear uno nuevo
            python_exec = find_python_executable(required_version)
            if python_exec is None:
                console.print("[red]✗ No se encontró Python compatible. Instala Python 3.10+[/]")
                return
            if not create_venv(python_exec):
                return
        else:
            # Comparar versiones numéricamente
            try:
                venv_major, venv_minor = map(int, current_venv_version.split("."))
                req_major, req_minor = map(int, required_version.split("."))
                if venv_major < req_major or (venv_major == req_major and venv_minor < req_minor):
                    # Venv tiene versión menor a la requerida, recrearlo
                    python_exec = find_python_executable(required_version)
                    if python_exec is None:
                        console.print("[red]✗ No se encontró Python compatible. Instala Python 3.10+[/]")
                        return
                    if not recreate_venv(python_exec):
                        return
                else:
                    console.print(f"[green]✓ Venv actualizado: {current_venv_version}[/green]")
            except:
                console.print(f"[green]✓ Venv actualizado: {current_venv_version}[/green]")
        
        # Verificar que las dependencias estén instaladas
        venv_python = os.path.join(VENV_DIR, "bin", "python3")
        if not os.path.exists(venv_python):
            venv_python = os.path.join(VENV_DIR, "bin", "python")
        
        # Verificar si las dependencias de ComfyUI están instaladas
        check_deps = subprocess.run(
            [venv_python, "-c", "import sqlalchemy, filelock, yaml, blake3"],
            capture_output=True, text=True, cwd=COMFYUI_DIR
        )
        if check_deps.returncode != 0:
            console.print("[yellow]⚠ Dependencias faltantes, instalando...[/]")
            progress.update(task, description="Instalando dependencias...")
            result = subprocess.run(
                [venv_python, "-m", "pip", "install", "-r", "requirements.txt"],
                capture_output=True, text=True, cwd=COMFYUI_DIR
            )
            if result.returncode != 0:
                console.print(f"[red]✗ Error instalando dependencias: {result.stderr}[/]")
                return
            console.print("[green]✓ Dependencias instaladas[/]")
        
        progress.remove_task(task)
    
    # Verificar modelo LLM
    with Progress(SpinnerColumn(), TextColumn("[bold blue]{task.description}[/]"), console=console) as progress:
        task = progress.add_task("Verificando modelo LLM...", total=None)
        model_id, max_tokens = get_loaded_model()
        progress.remove_task(task)
    
    if model_id == "ERROR":
        console.print(Panel(
            "[red]ERROR: No se pudo conectar con llama-server (puerto 9999)[/red]\n"
            "Asegúrate de que llama-server esté corriendo",
            border_style="red"
        ))
        return
    
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        task = progress.add_task("Verificando cambios upstream...", total=None)
        before_commits = subprocess.run(
            ["git", "log", "origin..upstream/master", "--oneline"],
            capture_output=True, text=True, cwd=COMFYUI_DIR
        ).stdout.strip()
        
        if not before_commits:
            progress.update(task, description="No hay cambios nuevos en upstream.")
            console.print(Panel("No hay cambios nuevos en upstream", style="cyan"))
            return
        
        progress.update(task, description=f"{len(before_commits.split(chr(10)))} commits encontrados")
        
        task.update(description="Trayendo cambios de upstream...")
        subprocess.run(["git", "fetch", "upstream"], capture_output=True, cwd=COMFYUI_DIR)
        subprocess.run(["git", "pull", "upstream", "master"], capture_output=True, cwd=COMFYUI_DIR)
        
        task.update(description="Actualizando ComfyUI-Manager...")
        if os.path.exists(MANAGER_DIR):
            subprocess.run(["git", "pull", "origin", "main"], capture_output=True, cwd=MANAGER_DIR)
        
        task.update(description="Instalando dependencias...")
        subprocess.run([venv_python, "-m", "pip", "install", "-r", "requirements.txt"], 
                      capture_output=True, cwd=COMFYUI_DIR)
        
        task.update(description="Sincronizando versión...")
        try:
            with open(os.path.join(COMFYUI_DIR, "pyproject.toml")) as f:
                content = f.read()
                version = re.search(r'version\s*=\s*"([^"]+)"', content)
                if version:
                    v = version.group(1)
                    version_file = os.path.join(COMFYUI_DIR, "comfyui_version.py")
                    if os.path.exists(version_file):
                        with open(version_file) as vf:
                            current = vf.read()
                            if v not in current:
                                with open(version_file, "w") as vf:
                                    vf.write(f'__version__ = "{v}"\n')
                                console.print(f"[green]✓ Versión actualizada: {v}[/green]")
                            else:
                                console.print(f"[green]✓ Versión sincronizada: {v}[/green]")
                    else:
                        with open(version_file, "w") as vf:
                            vf.write(f'__version__ = "{v}"\n')
                        console.print(f"[green]✓ Versión creada: {v}[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠ Error sincronizando versión: {e}[/yellow]")
        
        task.update(description="Verificando sintaxis...")
        for pyfile in ["main.py", "server.py"]:
            filepath = os.path.join(COMFYUI_DIR, pyfile)
            if os.path.exists(filepath):
                result = subprocess.run([venv_python, "-m", "py_compile", filepath], 
                                       capture_output=True)
                if result.returncode == 0:
                    console.print(f"[green]✓ {pyfile}: OK[/green]")
                else:
                    console.print(f"[red]✗ {pyfile}: ERROR[/red]")
        
        progress.remove_task(task)
    
    after_commits = subprocess.run(
        ["git", "log", "origin..upstream/master", "--oneline"],
        capture_output=True, text=True, cwd=COMFYUI_DIR
    ).stdout.strip()
    
    files_changed = subprocess.run(
        ["git", "diff", "origin..upstream/master", "--stat"],
        capture_output=True, text=True, cwd=COMFYUI_DIR
    ).stdout.strip()
    
    code_diffs = subprocess.run(
        ["git", "diff", "origin..upstream/master", "--", "*.py", "*.yaml", "*.json"],
        capture_output=True, text=True, cwd=COMFYUI_DIR
    ).stdout.strip()
    
    prompt = f"Commits nuevos:\n{after_commits}\n\nArchivos modificados:\n{files_changed}"
    if code_diffs:
        prompt += f"\n\nDiffs de código:\n{code_diffs}"
    
    with Progress(SpinnerColumn(), TextColumn("[bold]{task.description}[/]"), console=console) as progress:
        task = progress.add_task("Generando resumen con LLM...", total=None)
        summary = get_llm_summary(model_id, max_tokens, prompt)
        progress.remove_task(task)
    
    console.print(Panel(
        Markdown(summary),
        title="[bold blue]Resumen de cambios[/bold blue]",
        subtitle=f"[dim]Modelo: {model_id} | Contexto: {max_tokens} tokens[/dim]",
        border_style="blue",
        padding=(0, 1)
    ))


def show_menu(venv_ready=True):
    """Mostrar menú principal con Rich"""
    version = "0.0.0"
    version_file = os.path.join(COMFYUI_DIR, "comfyui_version.py")
    if os.path.exists(version_file):
        try:
            with open(version_file) as f:
                match = re.search(r'__version__\s*=\s*"([^"]+)"', f.read())
                if match:
                    version = match.group(1)
        except:
            pass
    
    model_id, max_tokens = get_loaded_model()
    model_info = f"{model_id} ({max_tokens} tokens)" if model_id != "ERROR" else "No conectado"
    venv_status = f"[green]Venv: {version}[/green]" if venv_ready else "[yellow]Venv: No creado[/]"
    
    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
    table.add_column("Opción", style="bold cyan", width=3)
    table.add_column("Descripción", style="white")
    
    if venv_ready:
        table.add_row("1", "[bold green]Run ComfyUI[/bold green] - Iniciar ComfyUI")
    else:
        table.add_row("1", "[dim]Run ComfyUI[/dim] - Requiere venv (opción 2)")
    
    table.add_row("2", "[bold magenta]Update & Summary[/bold magenta] - Actualizar y generar resumen")
    
    main_panel = Panel(
        table,
        title=Text("ComfyUI Plus", style="bold blue"),
        subtitle=Text(f"v{version} | Modelo: {model_info} | {venv_status}", style="dim"),
        border_style="blue",
        padding=(1, 2)
    )
    
    console.print(main_panel)
    console.print()
    
    choices = ["0", "1", "2"]
    if not venv_ready:
        choices = ["0", "2"]
    choice = Prompt.ask("Choose", choices=choices, default="0")
    return choice


def main():
    """Loop principal del menú"""
    venv_ready = os.path.exists(VENV_DIR)
    
    while True:
        try:
            choice = show_menu(venv_ready)
            
            if choice == "0":
                console.print("[bold cyan]Saliendo de ComfyUI Plus...[/]\n")
                break
            elif choice == "1":
                if not venv_ready:
                    console.print(Panel(
                        "[yellow]Venv no creado aún. Ejecuta la opción 2 primero.[/]",
                        border_style="yellow"
                    ))
                    continue
                run_comfyui()
            elif choice == "2":
                update_and_summary()
                venv_ready = os.path.exists(VENV_DIR)
            else:
                console.print(Panel("Opción inválida", style="red"))
        except KeyboardInterrupt:
            console.print("\n[bold yellow]Saliendo de ComfyUI Plus...[/]\n")
            break
        except Exception as e:
            console.print(Panel(f"Error: {e}", style="red"))


if __name__ == "__main__":
    main()
