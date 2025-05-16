import subprocess
import os

if __name__ == "__main__":
    # Get all pod names
    result = subprocess.run(
        ["kubectl", "get", "pods", "-o", "name"],
        capture_output=True, text=True
    )
    all_pods = result.stdout.strip().split("\n")
    
    # Filter out the one to exclude
    to_delete = [pod for pod in all_pods if "some-memcached" not in pod]
    for pod in to_delete:
        subprocess.run(["kubectl", "delete", pod, "--force"])

    subprocess.run("kubectl delete jobs --all", shell=True, text=True)
