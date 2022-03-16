mkdir -p kin2dyn && \
conda create -n kin2dyn -y python=3.7 cmake=3.14.0 && \
conda activate kin2dyn && \
cd kin2dyn && \
git clone https://github.com/facebookresearch/habitat-sim.git && \
git clone --branch kin2dyn git@github.com:joannetruong/habitat-lab.git && \
cd habitat-sim && \
git checkout 1fb3f693e40279db09d0e0c9e5fa1357c30ab03c && \
pip install -r requirements.txt && \
python setup.py install --bullet --headless
echo "Finished habitat-sim installation." && \
cd ../habitat-lab && \
pip uninstall numpy -y; \
pip uninstall numpy -y; \
pip uninstall numba -y; \
pip uninstall imageio -y; \
pip install typing-extensions~=3.7.4 google-auth==1.6.3 simplejson braceexpand pybullet cython pkgconfig squaternion && \
conda install -y pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch && \
pip install -r requirements.txt && \
python setup.py develop --all && \
mkdir -p data/datasets/pointnav_hm3d_gibson && \
mkdir -p data/scene_datasets && \
pip install gdown && \
gdown "https://drive.google.com/uc?id=1_JtEWPIgPaZVkH8fw2VNVssyLqr9pD7T" -O data/datasets/pointnav_hm3d_gibson/pointnav_spot_0.3.zip && \
cd data/datasets/pointnav_hm3d_gibson && unzip pointnav_spot_0.3.zip && rm pointnav_spot_0.3.zip && \
cd ../../scene_datasets && \
ln -s /datasets01/hm3d/090121/ hm3d && \
gdown "https://drive.google.com/uc?id=1EH-429McoUV81lIlAJlPCVPR8WTQWaWE" -O data/URDF_demo_assets.zip && \
cd .. && unzip URDF_demo_assets.zip && rm URDF_demo_assets.zip && \
echo "Finished habitat-lab installation."