# Arsenal Environment
ARSENAL_PATH="/home/hiroki11x/workspace/TSUBAMEGC_202105/pytorch_dnn_arsenal"
export ARSENAL_PATH

# FID Model Path
# Ref https://github.com/mseitzer/pytorch-fid/releases
FID_WEIGHTS_PATH="/home/hiroki11x/workspace/pytorch_gan/utils/pt_inception-2015-12-05-6726825d.pth"
export FID_WEIGHTS_PATH

# OUTPUT_DIR Path
OUTPUT_DIR="/home/hiroki11x/tmp/logs/cggan/"
export OUTPUT_DIR

# Pyenv VirtualEnv Environment
PYTHON_PATH="/home/hiroki11x/dl/virtualenv_py387/bin"
export PYTHON_PATH

CLUSTER_NAME="rio_cluster"
export CLUSTER_NAME

# # ======== Modules ========
source /etc/profile.d/modules.sh
module load cuda/11.0
module load cudnn/cuda-11.0/8.0