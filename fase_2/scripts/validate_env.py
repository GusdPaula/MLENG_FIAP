#!/usr/bin/env python3
"""
Environment Validation Script for Tech Challenge Fase 2.
This script checks Python version, required packages, environment variables
(via Pydantic Settings), PyTorch GPU capabilities, MLflow server reachability,
and AWS credentials.
"""

import os
import sys
import urllib.error
import urllib.request

# ANSI Escape Sequences for premium console outputs
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_header(title):
    print(f"\n{BOLD}{BLUE}=== {title} ==={RESET}")


def print_success(message):
    print(f"  {GREEN}✔{RESET} {message}")


def print_failure(message):
    print(f"  {RED}✘{RESET} {message}")


def print_warning(message):
    print(f"  {YELLOW}⚠{RESET} {message}")


def _check_python_version():
    print_header("1. Python Version Check")
    req_major, req_minor = 3, 12
    cur_major, cur_minor = sys.version_info.major, sys.version_info.minor
    print(f"  Running Python {sys.version.split()[0]}")
    if (cur_major, cur_minor) < (req_major, req_minor):
        print_failure(f"Python version must be >= {req_major}.{req_minor}. Found {cur_major}.{cur_minor}")
        return True
    print_success(f"Python version satisfies requirements (>= {req_major}.{req_minor})")
    return False


def _check_package_imports():
    print_header("2. Package Dependency Check")
    required_packages = [
        ("dotenv", "python-dotenv"),
        ("pydantic_settings", "pydantic-settings"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("sklearn", "scikit-learn"),
        ("torch", "pytorch"),
        ("mlflow", "mlflow"),
        ("dvc", "dvc"),
        ("boto3", "boto3"),
        ("yaml", "pyyaml"),
    ]
    imported_packages = {}
    has_errors = False
    for mod_name, pkg_name in required_packages:
        try:
            imported_packages[mod_name] = __import__(mod_name)
            print_success(f"Successfully imported {pkg_name} ({mod_name})")
        except ImportError as e:
            print_failure(f"Failed to import {pkg_name} ({mod_name}): {e}")
            has_errors = True
    return imported_packages, has_errors


def _validate_env_vars(fase2_dir, imported_packages):
    print_header("3. Environment Variables (.env) via Pydantic Settings")
    env_path = os.path.join(fase2_dir, ".env")
    if os.path.exists(env_path):
        print_success(f"Found .env file at {env_path}")
    else:
        print_warning(f"No .env file found at {env_path}. Using system environment variables.")

    src_path = os.path.join(fase2_dir, "ecommerce_recommender", "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    env_vars, has_errors = {}, False
    try:
        from recommender.config import Settings

        settings = Settings(_env_file=env_path if os.path.exists(env_path) else None)
        print_success("Pydantic Settings loaded and validated successfully!")
        print(f"    MLFLOW_TRACKING_URI = {settings.mlflow_tracking_uri}")
        env_vars = {
            "MLFLOW_TRACKING_URI": settings.mlflow_tracking_uri,
            "AWS_DEFAULT_REGION": settings.aws_default_region,
            "AWS_REGION": settings.aws_region,
            "AWS_PROFILE": settings.aws_profile,
        }
    except Exception as e:
        print_failure(f"Pydantic Settings validation failed: {e}")
        has_errors = True
        if "dotenv" in imported_packages and os.path.exists(env_path):
            imported_packages["dotenv"].load_dotenv(env_path)
        for var in [
            "MLFLOW_TRACKING_URI",
            "AWS_DEFAULT_REGION",
            "AWS_REGION",
            "AWS_PROFILE",
        ]:
            if val := os.getenv(var):
                env_vars[var] = val
    return env_vars, has_errors


def _check_pytorch_device(imported_packages):
    if "torch" not in imported_packages:
        return
    print_header("4. PyTorch Hardware Acceleration Check")
    torch_mod = imported_packages["torch"]
    print(f"  PyTorch Version: {torch_mod.__version__}")
    if torch_mod.cuda.is_available():
        print_success("CUDA (NVIDIA GPU) is available!")
        print(f"    Device Name: {torch_mod.cuda.get_device_name(0)}")
    elif hasattr(torch_mod.backends, "mps") and torch_mod.backends.mps.is_available():
        print_success("MPS (Apple Silicon GPU) is available!")
    else:
        print_warning("No GPU acceleration found. PyTorch will run on CPU.")


def _check_mlflow_server(env_vars):
    if "MLFLOW_TRACKING_URI" not in env_vars:
        return
    print_header("5. MLflow Server Connectivity Check")
    uri = env_vars["MLFLOW_TRACKING_URI"]
    if not uri.startswith("http"):
        print_success(f"MLflow configured for local storage: {uri}")
        return
    print(f"  Pinging MLflow tracking server: {uri} ...")
    try:
        req = urllib.request.Request(uri, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as response:
            status_code = response.getcode()
            if status_code in (200, 301, 302):
                print_success(f"Connection to MLflow server successful! (Status: {status_code})")
            else:
                print_warning(f"MLflow server returned status code: {status_code}")
    except Exception as e:
        print_warning(f"Could not reach MLflow tracking server at {uri}. Details: {e}")


def _check_aws_connectivity(imported_packages, env_vars):
    if "boto3" not in imported_packages:
        return
    print_header("6. AWS S3 Connectivity Check")
    profile = env_vars.get("AWS_PROFILE")
    region = env_vars.get("AWS_REGION") or env_vars.get("AWS_DEFAULT_REGION")
    try:
        boto3_mod = imported_packages["boto3"]
        session = boto3_mod.Session(profile_name=profile, region_name=region) if profile else boto3_mod.Session(region_name=region)
        caller = session.client("sts").get_caller_identity()
        print_success(f"AWS Credentials verified! Account: {caller.get('Account')}")
        buckets = session.client("s3").list_buckets()
        print_success(f"Successfully connected to S3. Account has {len(buckets.get('Buckets', []))} buckets.")
    except Exception as e:
        print_warning(f"Could not verify AWS/S3 connection. Details: {e}")


def main():
    print(f"{BOLD}{BLUE}==============================================={RESET}")
    print(f"{BOLD}{BLUE}      MLENG FIAP - PHASE 2 ENV VALIDATOR       {RESET}")
    print(f"{BOLD}{BLUE}==============================================={RESET}")

    err1 = _check_python_version()
    imported_packages, err2 = _check_package_imports()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    fase2_dir = os.path.dirname(script_dir)
    env_vars, err3 = _validate_env_vars(fase2_dir, imported_packages)

    _check_pytorch_device(imported_packages)
    _check_mlflow_server(env_vars)
    _check_aws_connectivity(imported_packages, env_vars)

    print(f"\n{BOLD}{BLUE}==============================================={RESET}")
    if err1 or err2 or err3:
        print(f"{BOLD}{RED}  ENV VALIDATION FAILED! Please resolve the errors.{RESET}")
        print(f"{BOLD}{BLUE}==============================================={RESET}")
        sys.exit(1)

    print(f"{BOLD}{GREEN}  ENV VALIDATION SUCCESSFUL! Everything is ready.{RESET}")
    print(f"{BOLD}{BLUE}==============================================={RESET}")
    sys.exit(0)


if __name__ == "__main__":
    main()
