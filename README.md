# Installation

Habitat-Sim: (`articulated-objects-prototype` branch)
```
git clone https://github.com/facebookresearch/habitat-sim.git && \
git checkout 4613b736964cd00dc4763a74996c46d1fbe67563

conda create -n habitat-urdf python=3.6 cmake=3.14.0
conda activate habitat-urdf
pip install -r requirements.txt
pip install squaternion
conda install -c conda-forge bullet -y
python setup.py install --bullet --headless
```

If building Habitat-sim fails, try to build with parallel processes:

```
build.sh --parallel 2 --bullet --headless
modify your .bashrc: export PYTHONPATH=$PYTHONPATH:/path/to/habitat-sim/
```
or `python setup.py build_ext --parallel 2 install --bullet --headless`

Habitat-Lab:  
```
git clone https://github.com/facebookresearch/habitat-lab.git && \
git checkout ac937fde9e14a47968eaf221857eb4e65f48383e

cd habitat-lab
pip install -r requirements.txt
python setup.py develop --all # install habitat and habitat_baselines
```

#### Run instructions for creating a video of spot using the controller

The relevant python script is `spot_walking_test.py`. To run this file, use `walking_test.sh`...be sure to change relevant symlinks and directory changes. All relevant data is in `/srv/share3/mrudolph8/data/data/`
