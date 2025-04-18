import argparse
import os
import subprocess

# Utility to get name of instance
def get_instance_name(prefix: str):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments to Launch Cluster on GCloud")
    parser.add_argument("--part", type=str, choices=["1", "2a", "2b", "3", "4"], help="Cluster to launch")
    args = parser.parse_args()

    # Add environment variables and create resources
    os.environ["KOPS_STATE_STORE"] = "gs://cca-eth-2025-group-73-jfleitas/"
    os.environ["PROJECT"] = "cca-eth-2025-group-73"
    subprocess.run(f"kops create -f part{args.part}.yaml", text=True, shell=True)

    # Deploy the cluster
    subprocess.run(f"kops update cluster --name part{args.part}.k8s.local --yes --admin", text=True, shell=True)

    # Validate the cluster
    subprocess.run("kops validate cluster --wait 10m", text=True, shell=True)

    # SSH into 
