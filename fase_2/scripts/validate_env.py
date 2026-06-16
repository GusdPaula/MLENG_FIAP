#!/usr/bin/env python3
"""
Environment Validation Script for Tech Challenge Fase 2.
This script checks Python version, required packages, environment variables
(via Pydantic Settings), PyTorch GPU capabilities, MLflow server reachability,
and AWS credentials.
"""

import sys
import os
import urllib.request
import urllib.error

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

def main():
    print(f"{BOLD}{BLUE}==============================================={RESET}")
    print(f"{BOLD}{BLUE}      MLENG FIAP - PHASE 2 ENV VALIDATOR       {RESET}")
    print(f"{BOLD}{BLUE}==============================================={RESET}")

    has_errors = False

    # 1. Check Python Version
    print_header("1. Python Version Check")
    req_major, req_minor = 3, 12
    cur_major, cur_minor = sys.version_info.major, sys.version_info.minor
    print(f"  Running Python {sys.version.split()[0]}")
    if (cur_major, cur_minor) < (req_major, req_minor):
        print_failure(f"Python version must be >= {req_major}.{req_minor}. Found {cur_major}.{cur_minor}")
        has_errors = True
    else:
        print_success(f"Python version satisfies requirements (>= {req_major}.{req_minor})")

    # 2. Check Package Imports
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
        ("yaml", "pyyaml")
    ]

    imported_packages = {}
    for mod_name, pkg_name in required_packages:
        try:
            mod = __import__(mod_name)
            imported_packages[mod_name] = mod
            print_success(f"Successfully imported {pkg_name} ({mod_name})")
        except ImportError as e:
            print_failure(f"Failed to import {pkg_name} ({mod_name}): {e}")
            has_errors = True

    # 3. Validate Environment Variables via Pydantic Settings
    print_header("3. Environment Variables (.env) via Pydantic Settings")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fase2_dir = os.path.dirname(script_dir)
    env_path = os.path.join(fase2_dir, ".env")

    if os.path.exists(env_path):
        print_success(f"Found .env file at {env_path}")
    else:
        print_warning(f"No .env file found at {env_path}. Using system environment variables.")

    # Add project src to path so we can import Settings
    src_path = os.path.join(fase2_dir, "ecommerce_recommender", "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    env_vars = {}
    try:
        from recommender.config import Settings
        settings = Settings(_env_file=env_path if os.path.exists(env_path) else None)
        print_success("Pydantic Settings loaded and validated successfully!")
        print(f"    MLFLOW_TRACKING_URI = {settings.mlflow_tracking_uri}")
        print(f"    AWS_DEFAULT_REGION  = {settings.aws_default_region}")
        print(f"    AWS_REGION          = {settings.aws_region}")
        print(f"    AWS_PROFILE         = {settings.aws_profile}")
        env_vars = {
            "MLFLOW_TRACKING_URI": settings.mlflow_tracking_uri,
            "AWS_DEFAULT_REGION": settings.aws_default_region,
            "AWS_REGION": settings.aws_region,
            "AWS_PROFILE": settings.aws_profile,
        }
    except Exception as e:
        print_failure(f"Pydantic Settings validation failed: {e}")
        has_errors = True
        # Fallback to manual check
        required_var_names = ["MLFLOW_TRACKING_URI", "AWS_DEFAULT_REGION", "AWS_REGION", "AWS_PROFILE"]
        if "dotenv" in imported_packages and os.path.exists(env_path):
            imported_packages["dotenv"].load_dotenv(env_path)
        for var in required_var_names:
            val = os.getenv(var)
            if val:
                env_vars[var] = val

    # 4. Check PyTorch Device / Hardware Acceleration
    if "torch" in imported_packages:
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

    # 5. Check MLflow Server Connection
    if "MLFLOW_TRACKING_URI" in env_vars:
        print_header("5. MLflow Server Connectivity Check")
        uri = env_vars["MLFLOW_TRACKING_URI"]
        if uri.startswith("http"):
            print(f"  Pinging MLflow tracking server: {uri} ...")
            try:
                req = urllib.request.Request(
                    uri,
                    headers={'User-Agent': 'Mozilla/5.0'}
                )
                with urllib.request.urlopen(req, timeout=5) as response:
                    status_code = response.getcode()
                    if status_code in (200, 301, 302):
                        print_success(f"Connection to MLflow server successful! (Status: {status_code})")
                    else:
                        print_warning(f"MLflow server returned status code: {status_code}")
            except Exception as e:
                print_warning(f"Could not reach MLflow tracking server at {uri}. Details: {e}")
        else:
            print_success(f"MLflow configured for local storage: {uri}")

    # 6. Check AWS / S3 Connectivity
    if "boto3" in imported_packages:
        print_header("6. AWS S3 Connectivity Check")
        profile = env_vars.get("AWS_PROFILE")
        region = env_vars.get("AWS_REGION") or env_vars.get("AWS_DEFAULT_REGION")

        try:
            session = None
            if profile:
                session = imported_packages["boto3"].Session(profile_name=profile, region_name=region)
            else:
                session = imported_packages["boto3"].Session(region_name=region)

            sts = session.client('sts')
            caller = sts.get_caller_identity()
            print_success(f"AWS Credentials verified! Account: {caller.get('Account')}, User/Role: {caller.get('Arn').split('/')[-1]}")

            s3 = session.client('s3')
            buckets = s3.list_buckets()
            bucket_names = [b['Name'] for b in buckets.get('Buckets', [])]
            print_success(f"Successfully connected to S3. Account has {len(bucket_names)} buckets.")
        except Exception as e:
            print_warning(f"Could not verify AWS/S3 connection. Details: {e}")
            print(f"    Please ensure AWS CLI is configured and profile '{profile}' is set up.")

    print(f"\n{BOLD}{BLUE}==============================================={RESET}")
    if has_errors:
        print(f"{BOLD}{RED}  ENV VALIDATION FAILED! Please resolve the errors.{RESET}")
        print(f"{BOLD}{BLUE}==============================================={RESET}")
        sys.exit(1)
    else:
        print(f"{BOLD}{GREEN}  ENV VALIDATION SUCCESSFUL! Everything is ready.{RESET}")
        print(f"{BOLD}{BLUE}==============================================={RESET}")
        sys.exit(0)

if __name__ == "__main__":
    main()
