#!/bin/bash
#YBATCH -r any_1
#SBATCH -N 1
#SBATCH -J conjugate_gradient.sh
#SBATCH --time=1:00:00

# ======== Module, Virtualenv and Other Dependencies======
source ./rio_env.sh
echo "PYTHON Environment: $PYTHON_PATH"
export PYTHONPATH=.
export PATH=$PYTHON_PATH:$PATH
which python

# ======== Copy Data ======
# t0=$(date +%s)
# tar zxf /mnt/nfs/datasets/OOD/cifar-10-python.tar.gz -C $HINADORI_LOCAL_SCRATCH
# t1=$(date +%s)
# echo "Time for dataset stage-in: $((t1 - t0)) sec"

#======== Configurations ========
CMD_EXECUTE="python cgd_example.py --lr=0.01 --optimizer-name=cgd --beta-update-rule=DY"

echo "Job started on $(date)"
echo "................................"
echo "[CMD_EXECUTE] :  $CMD_EXECUTE"
echo ""

# pushd ../../
eval $CMD_EXECUTE
# popd
echo "................................"
echo "Job is over on $(date)"