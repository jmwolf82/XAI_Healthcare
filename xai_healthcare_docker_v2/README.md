# Deployment of XAI Healthcare into AWS Cloud

One of the primary requirement for our code is the availability of CUDA and pytorch having setup with Nvida drivers and cuda enabled.

# EC2 Instance
After a fair bit of trial and errors with Amazon Linux 2, we finally chose `Deeplearning AMI GPU PyTorch 1.11.0 Ubuntu 20.04 20221004` which is `64-bit (x86)` architecture and has `AMI ID ami-0c968d7ef8a4b0c34`.

	Built with PyTorch conda environment, NVIDIA CUDA, cuDNN, NCCL, GPU Driver, Docker, NVIDIA-Docker and EFA support. 

Despite all the claim, when tested, the Vm returned the following

```
ubuntu@ip-172-31-190-101:~$ nvidia-smi
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running.
```

# Install CUDA from scratch

```sh
sudo apt-get update
# Install Dynamic Kernel Module Support (DKMS)
sudo apt-get install dkms #> dkms is already the newest version (2.8.1-5ubuntu2).
```
Now install Nvidia Cuda Toolkit

```sh
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

The installation prompts for a restart

	*****************************************************************************
	*** Reboot your computer and verify that the NVIDIA graphics driver can   ***
	*** be loaded.                                                            ***
	*****************************************************************************

# Retry

Finally found the architechture of NVIDIA CUDA
https://github.com/NVIDIA/nvidia-docker

The base system does not need Cuda toolkit, but does need nvidia drivers.

### Step 1
Verify You Have a CUDA-Capable GPU

	lspci | grep -i nvidia

Our system failed here.

	root@ip-172-31-190-101:/home/ubuntu# lspci
	00:00.0 Host bridge: Intel Corporation 440FX - 82441FX PMC [Natoma] (rev 02)
	00:01.0 ISA bridge: Intel Corporation 82371SB PIIX3 ISA [Natoma/Triton II]
	00:01.1 IDE interface: Intel Corporation 82371SB PIIX3 IDE [Natoma/Triton II]
	00:01.3 Bridge: Intel Corporation 82371AB/EB/MB PIIX4 ACPI (rev 01)
	00:02.0 VGA compatible controller: Cirrus Logic GD 5446
	00:03.0 Unassigned class [ff80]: XenSource, Inc. Xen Platform Device (rev 01)

# Finding AWS and NVIDIA instances
Found more info on this at https://aws.amazon.com/nvidia/

	Complementing P4d and P3 instances, for ML inference, Amazon EC2 G4 instances featuring NVIDIA T4 Tensor Core GPUs deliver the most cost-effective GPU instances in the cloud for ML inference.

Further

	Amazon EC2 G4 instances are the industry’s most cost-effective and versatile GPU instances for deploying machine learning models such as image classification, object detection, and speech recognition, and for graphics-intensive applications such as remote graphics workstations, game streaming, and graphics rendering. G4 instances are available with a choice of NVIDIA GPUs (G4dn) or AMD GPUs (G4ad).

So G4dn it is!

	So we found G4dn-xlarge as the smalled Instance we could find.

# So we have CUDA compatible Instance 

```
ubuntu@ip-172-31-30-17:~$ lspci| grep NVIDIA
00:1e.0 3D controller: NVIDIA Corporation TU104GL [Tesla T4] (rev a1)
```