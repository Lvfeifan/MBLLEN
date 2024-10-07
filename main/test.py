{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "gpuType": "T4"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "7ZCu6SpJFM8U",
        "outputId": "6134427d-aa5d-47ee-d360-aca147eaf314"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: tensorflow in /usr/local/lib/python3.10/dist-packages (2.17.0)\n",
            "Requirement already satisfied: absl-py>=1.0.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (1.4.0)\n",
            "Requirement already satisfied: astunparse>=1.6.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (1.6.3)\n",
            "Requirement already satisfied: flatbuffers>=24.3.25 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (24.3.25)\n",
            "Requirement already satisfied: gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (0.6.0)\n",
            "Requirement already satisfied: google-pasta>=0.1.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (0.2.0)\n",
            "Requirement already satisfied: h5py>=3.10.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (3.11.0)\n",
            "Requirement already satisfied: libclang>=13.0.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (18.1.1)\n",
            "Requirement already satisfied: ml-dtypes<0.5.0,>=0.3.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (0.4.1)\n",
            "Requirement already satisfied: opt-einsum>=2.3.2 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (3.3.0)\n",
            "Requirement already satisfied: packaging in /usr/local/lib/python3.10/dist-packages (from tensorflow) (24.1)\n",
            "Requirement already satisfied: protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (3.20.3)\n",
            "Requirement already satisfied: requests<3,>=2.21.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (2.32.3)\n",
            "Requirement already satisfied: setuptools in /usr/local/lib/python3.10/dist-packages (from tensorflow) (71.0.4)\n",
            "Requirement already satisfied: six>=1.12.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (1.16.0)\n",
            "Requirement already satisfied: termcolor>=1.1.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (2.4.0)\n",
            "Requirement already satisfied: typing-extensions>=3.6.6 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (4.12.2)\n",
            "Requirement already satisfied: wrapt>=1.11.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (1.16.0)\n",
            "Requirement already satisfied: grpcio<2.0,>=1.24.3 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (1.64.1)\n",
            "Requirement already satisfied: tensorboard<2.18,>=2.17 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (2.17.0)\n",
            "Requirement already satisfied: keras>=3.2.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (3.4.1)\n",
            "Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (0.37.1)\n",
            "Requirement already satisfied: numpy<2.0.0,>=1.23.5 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (1.26.4)\n",
            "Requirement already satisfied: wheel<1.0,>=0.23.0 in /usr/local/lib/python3.10/dist-packages (from astunparse>=1.6.0->tensorflow) (0.44.0)\n",
            "Requirement already satisfied: rich in /usr/local/lib/python3.10/dist-packages (from keras>=3.2.0->tensorflow) (13.8.1)\n",
            "Requirement already satisfied: namex in /usr/local/lib/python3.10/dist-packages (from keras>=3.2.0->tensorflow) (0.0.8)\n",
            "Requirement already satisfied: optree in /usr/local/lib/python3.10/dist-packages (from keras>=3.2.0->tensorflow) (0.12.1)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.21.0->tensorflow) (3.3.2)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.21.0->tensorflow) (3.10)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.21.0->tensorflow) (2.2.3)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.21.0->tensorflow) (2024.8.30)\n",
            "Requirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.18,>=2.17->tensorflow) (3.7)\n",
            "Requirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.18,>=2.17->tensorflow) (0.7.2)\n",
            "Requirement already satisfied: werkzeug>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.18,>=2.17->tensorflow) (3.0.4)\n",
            "Requirement already satisfied: MarkupSafe>=2.1.1 in /usr/local/lib/python3.10/dist-packages (from werkzeug>=1.0.1->tensorboard<2.18,>=2.17->tensorflow) (2.1.5)\n",
            "Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.10/dist-packages (from rich->keras>=3.2.0->tensorflow) (3.0.0)\n",
            "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.10/dist-packages (from rich->keras>=3.2.0->tensorflow) (2.18.0)\n",
            "Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.10/dist-packages (from markdown-it-py>=2.2.0->rich->keras>=3.2.0->tensorflow) (0.1.2)\n",
            "Requirement already satisfied: keras in /usr/local/lib/python3.10/dist-packages (3.4.1)\n",
            "Requirement already satisfied: absl-py in /usr/local/lib/python3.10/dist-packages (from keras) (1.4.0)\n",
            "Requirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (from keras) (1.26.4)\n",
            "Requirement already satisfied: rich in /usr/local/lib/python3.10/dist-packages (from keras) (13.8.1)\n",
            "Requirement already satisfied: namex in /usr/local/lib/python3.10/dist-packages (from keras) (0.0.8)\n",
            "Requirement already satisfied: h5py in /usr/local/lib/python3.10/dist-packages (from keras) (3.11.0)\n",
            "Requirement already satisfied: optree in /usr/local/lib/python3.10/dist-packages (from keras) (0.12.1)\n",
            "Requirement already satisfied: ml-dtypes in /usr/local/lib/python3.10/dist-packages (from keras) (0.4.1)\n",
            "Requirement already satisfied: packaging in /usr/local/lib/python3.10/dist-packages (from keras) (24.1)\n",
            "Requirement already satisfied: typing-extensions>=4.5.0 in /usr/local/lib/python3.10/dist-packages (from optree->keras) (4.12.2)\n",
            "Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.10/dist-packages (from rich->keras) (3.0.0)\n",
            "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.10/dist-packages (from rich->keras) (2.18.0)\n",
            "Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.10/dist-packages (from markdown-it-py>=2.2.0->rich->keras) (0.1.2)\n",
            "Requirement already satisfied: opencv-python in /usr/local/lib/python3.10/dist-packages (4.10.0.84)\n",
            "Requirement already satisfied: numpy>=1.21.2 in /usr/local/lib/python3.10/dist-packages (from opencv-python) (1.26.4)\n"
          ]
        }
      ],
      "source": [
        "!pip install tensorflow\n",
        "!pip install keras\n",
        "!pip install opencv-python"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!git clone --recursive https://github.com/Lvfeifan/MBLLEN.git\n",
        "%cd MBLLEN\n",
        "%cd main"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "BM5bVZtGFckV",
        "outputId": "30a6cb5a-968b-4a60-e281-2b08c841a7d7"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Cloning into 'MBLLEN'...\n",
            "remote: Enumerating objects: 41, done.\u001b[K\n",
            "remote: Counting objects: 100% (3/3), done.\u001b[K\n",
            "remote: Compressing objects: 100% (3/3), done.\u001b[K\n",
            "remote: Total 41 (delta 0), reused 0 (delta 0), pack-reused 38 (from 1)\u001b[K\n",
            "Receiving objects: 100% (41/41), 21.37 MiB | 19.82 MiB/s, done.\n",
            "Resolving deltas: 100% (7/7), done.\n",
            "/content/MBLLEN\n",
            "/content/MBLLEN/main\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Thay file test.py ở /content/MBLLEN/main/test.py thành https://drive.google.com/file/d/1ZZhILbHmN3BYBe9L884V9G5xOaaFlVtI/view?usp=sharing"
      ],
      "metadata": {
        "id": "iqak87QfYKci"
      }
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "3EFwpmaCYJxN"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "!python test.py"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "XhO8tb1vFneW",
        "outputId": "a9cab218-721a-49cb-eb27-486d56462870"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "2024-09-24 09:32:48.577998: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:485] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered\n",
            "2024-09-24 09:32:48.598431: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:8454] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered\n",
            "2024-09-24 09:32:48.604577: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1452] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered\n",
            "2024-09-24 09:32:48.618934: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.\n",
            "To enable the following instructions: AVX2 AVX512F FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.\n",
            "2024-09-24 09:32:49.615621: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT\n",
            "WARNING: All log messages before absl::InitializeLog() is called are written to STDERR\n",
            "I0000 00:00:1727170371.168719    2374 cuda_executor.cc:1015] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355\n",
            "I0000 00:00:1727170371.214804    2374 cuda_executor.cc:1015] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355\n",
            "I0000 00:00:1727170371.215090    2374 cuda_executor.cc:1015] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355\n",
            "I0000 00:00:1727170371.215801    2374 cuda_executor.cc:1015] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355\n",
            "I0000 00:00:1727170371.216019    2374 cuda_executor.cc:1015] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355\n",
            "I0000 00:00:1727170371.216189    2374 cuda_executor.cc:1015] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355\n",
            "I0000 00:00:1727170371.314519    2374 cuda_executor.cc:1015] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355\n",
            "I0000 00:00:1727170371.314824    2374 cuda_executor.cc:1015] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355\n",
            "2024-09-24 09:32:51.314966: W tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.cc:47] Overriding orig_value setting because the TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original config value was 0.\n",
            "I0000 00:00:1727170371.315070    2374 cuda_executor.cc:1015] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355\n",
            "2024-09-24 09:32:51.315222: I tensorflow/core/common_runtime/gpu/gpu_device.cc:2021] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 13949 MB memory:  -> device: 0, name: Tesla T4, pci bus id: 0000:00:04.0, compute capability: 7.5\n",
            "WARNING: All log messages before absl::InitializeLog() is called are written to STDERR\n",
            "I0000 00:00:1727170372.806180    2404 service.cc:146] XLA service 0x7ed790002490 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:\n",
            "I0000 00:00:1727170372.806235    2404 service.cc:154]   StreamExecutor device (0): Tesla T4, Compute Capability 7.5\n",
            "2024-09-24 09:32:52.928222: I tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.cc:268] disabling MLIR crash reproducer, set env var `MLIR_CRASH_REPRODUCER_DIRECTORY` to enable.\n",
            "2024-09-24 09:32:53.428926: I external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:531] Loaded cuDNN version 8906\n",
            "I0000 00:00:1727170376.082204    2404 device_compiler.h:188] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.\n",
            "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m4s\u001b[0m 4s/step\n",
            "The 1th image's Time:5.189787017999947s.\n",
            "E0000 00:00:1727170381.777526    2403 gpu_timer.cc:183] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.\n",
            "E0000 00:00:1727170381.949126    2403 gpu_timer.cc:183] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.\n",
            "E0000 00:00:1727170383.766364    2403 gpu_timer.cc:183] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.\n",
            "E0000 00:00:1727170383.945751    2403 gpu_timer.cc:183] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.\n",
            "E0000 00:00:1727170385.401383    2403 gpu_timer.cc:183] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.\n",
            "E0000 00:00:1727170385.575846    2403 gpu_timer.cc:183] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.\n",
            "E0000 00:00:1727170385.749812    2403 gpu_timer.cc:183] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.\n",
            "E0000 00:00:1727170387.324611    2403 gpu_timer.cc:183] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.\n",
            "E0000 00:00:1727170387.502978    2403 gpu_timer.cc:183] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.\n",
            "E0000 00:00:1727170390.040011    2403 gpu_timer.cc:183] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.\n",
            "E0000 00:00:1727170390.232573    2403 gpu_timer.cc:183] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.\n",
            "E0000 00:00:1727170392.164886    2403 gpu_timer.cc:183] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.\n",
            "E0000 00:00:1727170392.389915    2403 gpu_timer.cc:183] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.\n",
            "E0000 00:00:1727170392.710726    2403 gpu_timer.cc:183] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.\n",
            "E0000 00:00:1727170392.931025    2403 gpu_timer.cc:183] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.\n",
            "2024-09-24 09:33:14.343605: E external/local_xla/xla/service/slow_operation_alarm.cc:65] Trying algorithm eng0{} for conv (f32[1,16,1720,2296]{3,2,1,0}, u8[0]{0}) custom-call(f32[1,32,1716,2292]{3,2,1,0}, f32[32,16,5,5]{3,2,1,0}), window={size=5x5}, dim_labels=bf01_oi01->bf01, custom_call_target=\"__cudnn$convBackwardInput\", backend_config={\"operation_queue_id\":\"0\",\"wait_on_operation_queues\":[],\"cudnn_conv_backend_config\":{\"conv_result_scale\":1,\"activation_mode\":\"kNone\",\"side_input_scale\":0,\"leakyrelu_alpha\":0},\"force_earliest_schedule\":false} is taking a while...\n",
            "2024-09-24 09:33:14.769937: E external/local_xla/xla/service/slow_operation_alarm.cc:133] The operation took 1.426507564s\n",
            "Trying algorithm eng0{} for conv (f32[1,16,1720,2296]{3,2,1,0}, u8[0]{0}) custom-call(f32[1,32,1716,2292]{3,2,1,0}, f32[32,16,5,5]{3,2,1,0}), window={size=5x5}, dim_labels=bf01_oi01->bf01, custom_call_target=\"__cudnn$convBackwardInput\", backend_config={\"operation_queue_id\":\"0\",\"wait_on_operation_queues\":[],\"cudnn_conv_backend_config\":{\"conv_result_scale\":1,\"activation_mode\":\"kNone\",\"side_input_scale\":0,\"leakyrelu_alpha\":0},\"force_earliest_schedule\":false} is taking a while...\n",
            "E0000 00:00:1727170395.241417    2403 gpu_timer.cc:183] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.\n",
            "E0000 00:00:1727170395.420200    2403 gpu_timer.cc:183] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.\n",
            "E0000 00:00:1727170395.661245    2403 gpu_timer.cc:183] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.\n",
            "E0000 00:00:1727170395.843781    2403 gpu_timer.cc:183] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.\n",
            "E0000 00:00:1727170396.823528    2403 gpu_timer.cc:183] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.\n",
            "E0000 00:00:1727170397.004809    2403 gpu_timer.cc:183] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.\n",
            "E0000 00:00:1727170397.238711    2403 gpu_timer.cc:183] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.\n",
            "E0000 00:00:1727170397.412266    2403 gpu_timer.cc:183] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.\n",
            "E0000 00:00:1727170397.586523    2403 gpu_timer.cc:183] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.\n",
            "E0000 00:00:1727170400.217079    2403 gpu_timer.cc:183] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.\n",
            "E0000 00:00:1727170400.472797    2403 gpu_timer.cc:183] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.\n",
            "2024-09-24 09:33:21.497865: E external/local_xla/xla/service/slow_operation_alarm.cc:65] Trying algorithm eng0{} for conv (f32[1,32,1728,2304]{3,2,1,0}, u8[0]{0}) custom-call(f32[1,32,1728,2304]{3,2,1,0}, f32[32,32,3,3]{3,2,1,0}, f32[32]{0}), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_oi01->bf01, custom_call_target=\"__cudnn$convBiasActivationForward\", backend_config={\"operation_queue_id\":\"0\",\"wait_on_operation_queues\":[],\"cudnn_conv_backend_config\":{\"conv_result_scale\":1,\"activation_mode\":\"kRelu\",\"side_input_scale\":0,\"leakyrelu_alpha\":0},\"force_earliest_schedule\":false} is taking a while...\n",
            "2024-09-24 09:33:21.743652: E external/local_xla/xla/service/slow_operation_alarm.cc:133] The operation took 1.24587797s\n",
            "Trying algorithm eng0{} for conv (f32[1,32,1728,2304]{3,2,1,0}, u8[0]{0}) custom-call(f32[1,32,1728,2304]{3,2,1,0}, f32[32,32,3,3]{3,2,1,0}, f32[32]{0}), window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_oi01->bf01, custom_call_target=\"__cudnn$convBiasActivationForward\", backend_config={\"operation_queue_id\":\"0\",\"wait_on_operation_queues\":[],\"cudnn_conv_backend_config\":{\"conv_result_scale\":1,\"activation_mode\":\"kRelu\",\"side_input_scale\":0,\"leakyrelu_alpha\":0},\"force_earliest_schedule\":false} is taking a while...\n",
            "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m24s\u001b[0m 24s/step\n",
            "The 2th image's Time:41.22801462899997s.\n",
            "E0000 00:00:1727170421.014680    2403 gpu_timer.cc:183] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.\n",
            "E0000 00:00:1727170421.158303    2403 gpu_timer.cc:183] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.\n",
            "E0000 00:00:1727170421.294863    2403 gpu_timer.cc:183] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.\n",
            "E0000 00:00:1727170421.875953    2403 gpu_timer.cc:183] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.\n",
            "E0000 00:00:1727170422.012786    2403 gpu_timer.cc:183] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.\n",
            "E0000 00:00:1727170422.396073    2403 gpu_timer.cc:183] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.\n",
            "E0000 00:00:1727170422.532466    2403 gpu_timer.cc:183] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.\n",
            "E0000 00:00:1727170422.919864    2403 gpu_timer.cc:183] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.\n",
            "E0000 00:00:1727170423.058360    2403 gpu_timer.cc:183] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.\n",
            "E0000 00:00:1727170423.811942    2403 gpu_timer.cc:183] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.\n",
            "E0000 00:00:1727170423.953453    2403 gpu_timer.cc:183] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.\n",
            "E0000 00:00:1727170424.661252    2403 gpu_timer.cc:183] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.\n",
            "E0000 00:00:1727170424.807508    2403 gpu_timer.cc:183] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.\n",
            "E0000 00:00:1727170424.972919    2403 gpu_timer.cc:183] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.\n",
            "E0000 00:00:1727170425.117733    2403 gpu_timer.cc:183] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.\n",
            "E0000 00:00:1727170425.849613    2403 gpu_timer.cc:183] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.\n",
            "E0000 00:00:1727170425.988665    2403 gpu_timer.cc:183] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.\n",
            "E0000 00:00:1727170426.138971    2403 gpu_timer.cc:183] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.\n",
            "E0000 00:00:1727170426.276671    2403 gpu_timer.cc:183] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.\n",
            "E0000 00:00:1727170426.816553    2403 gpu_timer.cc:183] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.\n",
            "E0000 00:00:1727170426.951136    2403 gpu_timer.cc:183] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.\n",
            "E0000 00:00:1727170427.098919    2403 gpu_timer.cc:183] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.\n",
            "E0000 00:00:1727170427.233364    2403 gpu_timer.cc:183] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.\n",
            "E0000 00:00:1727170427.760144    2403 gpu_timer.cc:183] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.\n",
            "E0000 00:00:1727170427.913178    2403 gpu_timer.cc:183] Delay kernel timed out: measured time has sub-optimal accuracy. There may be a missing warmup execution, please investigate in Nsight Systems.\n",
            "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m8s\u001b[0m 8s/step\n",
            "The 3th image's Time:10.279682398999967s.\n",
            "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m4s\u001b[0m 4s/step\n",
            "The 4th image's Time:5.1550550060001115s.\n"
          ]
        }
      ]
    }
  ]
}
