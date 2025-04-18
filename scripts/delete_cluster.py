import argparse
import os
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments to Launch Cluster on GCloud")
    parser.add_argument("--part", type=str, choices=["1", "2a", "2b", "3", "4"], help="Cluster to launch")
    args = parser.parse_args()

    # Add environment variables
    os.environ["KOPS_STATE_STORE"] = "gs://cca-eth-2025-group-73-jfleitas/"
    os.environ["PROJECT"] = "cca-eth-2025-group-73"

    # Delete all jobs and pods in the cluster
    subprocess.run("kubectl delete pods --all", shell=True, text=True)
    subprocess.run("kubectl delete jobs --all", shell=True, text=True)

    # Delete the cluster
    subprocess.run(f"kops delete cluster --name part{args.part}.k8s.local --yes", shell=True, text=True)
