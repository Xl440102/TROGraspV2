mkdir -p ~/.pip
cat > ~/.pip/pip.conf <<'EOF'
[global]
index-url = https://mirrors.aliyun.com/pypi/simple
EOF

conda create -n TROV2 python=3.10 -y
conda activate TROV2

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1   -i https://mirrors.aliyun.com/pypi/simple
pip install matplotlib loguru tqdm matplotlib pybullet pytorch-kinematics viser
pip install -U fpsample

pip install git+https://github.com/otaheri/chamfer_distance
pip install git+https://github.com/otaheri/bps_torch
pip install chamferdist --no-build-isolation
pip install "git+https://github.com/facebookresearch/pytorch3d.git" --no-build-isolation
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.4.1+cu124.html  --no-build-isolation
pip install accelerate tensorboard omegaconf
# TROV2 uses Theseus SO3 geometry; its optional Cholmod solver needs SuiteSparse headers.
pip install torchlie==0.1.0 torchkin==0.1.1 semantic-version==2.10.0
pip install theseus-ai==0.2.3 --no-deps
