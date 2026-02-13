import json
import os
import sys
from pathlib import Path

def verify_notebook(nb_path):
    print(f"Verifying {nb_path}...")
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    code_cells = [cell['source'] for cell in nb['cells'] if cell['cell_type'] == 'code']
    
    full_code = []
    for cell_source in code_cells:
        if isinstance(cell_source, list):
            full_code.extend(cell_source)
        else:
            full_code.append(cell_source)
        full_code.append("\n")
    
    script_content = "".join(full_code)
    
    # Mocking plotly.show() to avoid opening browser/hanging
    # Use 'json' renderer as it's less likely to try opening a window
    script_content = "import plotly.io as pio\npio.renderers.default = 'json'\n" + script_content
    script_content = script_content.replace(".show()", "")
    
    # Speed up for verification
    script_content = script_content.replace("epochs=10", "epochs=1")
    script_content = script_content.replace("num_epochs=100", "num_epochs=1")
    script_content = script_content.replace("num_epochs=10", "num_epochs=1")
    
    temp_script = f"temp_verify_{os.path.basename(nb_path)}.py"
    with open(temp_script, 'w') as f:
        f.write(script_content)
    
    try:
        import subprocess
        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")
        
        # Capture both stdout and stderr
        result = subprocess.run([sys.executable, temp_script], env=env, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ {nb_path} failed verification.")
            print("Error Output:")
            print(result.stdout)
            print(result.stderr)
            return False
        else:
            print(f"✅ {nb_path} passed verification.")
            return True
    except Exception as e:
        print(f"❌ {nb_path} failed due to unexpected error: {e}")
        return False
    finally:
        if os.path.exists(temp_script):
            os.remove(temp_script)

if __name__ == "__main__":
    notebooks = sorted(list(Path("notebooks").glob("*.ipynb")))
    all_passed = True
    for nb in notebooks:
        if not verify_notebook(str(nb)):
            all_passed = False
    
    if not all_passed:
        sys.exit(1)
    else:
        print("\nAll notebooks verified successfully!")
