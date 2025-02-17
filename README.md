# NTN Channel Simulator

## Prerequisites

1. **UV Package Manager is recomended** ([uv-astral](https://docs.astral.sh/uv/getting-started/installation/))
2. **CUDA and cuDNN Compatibility**: Refer to the table below to ensure compatibility with TensorFlow versions. [Check installation guide](https://developer.nvidia.com/cudnn-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local)
3.

| TensorFlow Version | Python Version | cuDNN | CUDA |
| ------------------ | -------------- | ----- | ---- |
| 2.15.0             | 3.9 - 3.11     | 8.9   | 12.2 |

Since the project specifies **tensorflow[and-cuda]>=2.15.1**, ensure you have CUDA 12.3 or higher and cuDNN 8.9 or higher installed.

## Installation Guide

### 1. Install uv-astral and Set Up Python Virtual Environment

1. **Create a Virtual Environment with uv**:

   ```bash
   uv venv
   ```

2. **Activate Virtual Environment**:

   ```bash
   source .venv/bin/activate
   # or for fish
   source .venv/bin/activate.fish
   # or
   source .venv/bin/activate.csh
   ```

3. **Install Python 3.11.6**:

   ```bash
   uv python install "3.11.6"
   ```

4. **Pin Python 3.11.6 in your enviroment**:
   ```bash
   uv python pin "3.11.6"
   ```

### 2. Install Project Dependencies

1. **Sync an environment with a lockfile**:

   ```bash
   uv sync
   ```

2. **Sync an environment with a lockfile.** (if needed):
   ```bash
   uv pip sync myproject.toml
   ```

### 3. Configure CUDA and cuDNN

Ensure the correct versions of CUDA and cuDNN are installed to match your TensorFlow version. Refer to the table above for version alignment.

````

## Notes

- Ensure GPU drivers are up to date.
- Verify environment variables (`PATH`, `LD_LIBRARY_PATH`) include CUDA and cuDNN paths.
- Test TensorFlow GPU installation with:
  ```python
  import tensorflow as tf
  print(tf.config.list_physical_devices('GPU'))
````

---

Happy coding!
