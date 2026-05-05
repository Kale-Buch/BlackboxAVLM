To run the attacks you need to create two virtual enviornments:   
conda create -n vlm python=3.10.19
conda activate vlm
pip install -r requirements_vlm.txt

conda create -n vlmg python=3.10.19
conda activate vlmg
pip install -r requirements_vlmg.t


FOR INSTALLNG FLASH_ATTENTION
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
export TORCH_CUDA_ARCH_LIST="9.0"
pip install psutil
pip install ninja
export TMPDIR=$HOME/tmp
mkdir -p $TMPDIR
module load cuda/12.9.1
pip install flash-attn==2.8.3 --no-build-isolation --no-cache-dir


For guard eval
Need sentencepiece for some reason
Need transformers 4.50
Need outlines==0.1.1
