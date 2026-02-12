import subprocess
import sys
import os
from pathlib import Path


def run_example(script_path):
    print(f"\n--- Running {script_path} ---")
    try:
        # Run with PYTHONPATH set to current directory
        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")

        result = subprocess.run(
            [sys.executable, script_path],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        print("Success!")
        # Print only the last few lines of output to keep it clean
        print("\n".join(result.stdout.splitlines()[-5:]))
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed! Exit code: {e.returncode}")
        print("Output:")
        print(e.stdout)
        print("Error:")
        print(e.stderr)
        return False


def main():
    examples_dir = Path("examples")
    scripts = sorted(
        [str(p) for p in examples_dir.glob("*.py") if p.name != "__init__.py"]
    )

    print(f"Found {len(scripts)} examples to test.")

    failures = []
    for script in scripts:
        if not run_example(script):
            failures.append(script)

    if failures:
        print(f"\nErrors occurred in the following scripts: {failures}")
        sys.exit(1)
    else:
        print("\nAll examples passed successfully!")


if __name__ == "__main__":
    main()
