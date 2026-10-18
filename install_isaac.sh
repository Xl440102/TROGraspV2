pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1   -i https://mirrors.aliyun.com/pypi/simple
pip install matplotlib loguru tqdm matplotlib pybullet pytorch-kinematics viser
pip install -U fpsample

wget "https://developer.nvidia.com/isaac-gym-preview-4"
mv isaac-gym-preview-4 isaac-gym-preview-4.tar.gz
tar -xvf isaac-gym-preview-4.tar.gz
rm isaac-gym-preview-4.tar.gz

cd isaacgym/python
pip install -e .
