import subprocess
import time

script1 = "whale_vessel_detector.py"
script2 = "whale_vessel_detector_optimized.py"

cli_args = [
    "./data/",
    "bh",
    "--mode", "w"
]

def run(script):
    start = time.perf_counter()

    result = subprocess.run(
        ["python", script] + cli_args,
        stdout=subprocess.DEVNULL,   # Ignore output while timing
        stderr=subprocess.PIPE,
        text=True
    )

    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        print(f"{script} failed:")
        print(result.stderr)

    return elapsed

t1 = run(script1)
t2 = run(script2)

print(f"{script1}: {t1:.2f} s")
print(f"{script2}: {t2:.2f} s")
