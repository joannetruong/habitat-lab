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
```

Joanne has changes to Habitat-sim for getting link rigid velocity, and to change FOV of camera. Please copy the files from `habitat-sim-change` to `habitat-sim` and then build habitat-sim:

```
cp habitat-sim-changes/PhysicsManager.h habitat-sim/src/esp/physics/PhysicsManager.h && \
cp habitat-sim-changes/ArticulatedObject.h habitat-sim/src/esp/physics/ArticulatedObject.h && \
cp habitat-sim-changes/BulletArticulatedObject.h habitat-sim/src/esp/physics/bullet/BulletArticulatedObject.h && \
cp habitat-sim-changes/BulletArticulatedObject.cpp habitat-sim/src/esp/physics/bullet/BulletArticulatedObject.cpp && \
cp habitat-sim-changes/CameraSensor.cpp habitat-sim/src/esp/sensor/CameraSensor.cpp && \
cp habitat-sim-changes/CameraSensor.h habitat-sim/src/esp/sensor/CameraSensor.h && \
cp habitat-sim-changes/Simulator.cpp habitat-sim/src/esp/sim/Simulator.cpp && \
cp habitat-sim-changes/Simulator.h habitat-sim/src/esp/sim/Simulator.h && \
cp habitat-sim-changes/SensorBindings.cpp habitat-sim/src/esp/bindings/SensorBindings.cpp && \
cp habitat-sim-changes/SimBindings.cpp habitat-sim/src/esp/bindings/SimBindings.cpp 
```

`python setup.py install --bullet --headless`


If building Habitat-sim fails, try to build with parallel processes:

`python setup.py build_ext --parallel 2 install --bullet --headless`
 
or

```
./build.sh --parallel 2 --bullet --headless
modify your .bashrc: export PYTHONPATH=$PYTHONPATH:/path/to/habitat-sim/
```

Habitat-Lab:
## Install Habitat-Lab
```
git clone git@github.com:joannetruong/habitat-cont.git && \
cd habitat-cont && \ 
git checkout new_habitat && \ 
cd habitat-lab && \
pip install -r requirements.txt && \
python setup.py develop --all
```

## Add data, change paths
* URDF: `habitat-sim/data/URDF_demo_assets/spot_hybrid_urdf/habitat_spot_urdf/urdf/spot_hybrid.urdf`
* Scene: `spot_urdf_test/data/scene_datasets/coda/coda_hard.glb`
* Policies: `spot_urdf_test/ddppo_policies/ckpt.11.pth`
* Config path: `spot_urdf_test/config/ddppo_pointnav.yaml`, change line 2

## Run Spot URDF
* Walking Test: `python spot_walking_test.py Spot`
* Evaluate DDPPO Policy: `python spot_eval_ddppo.py Spot`. A GUI will appear to show the depth and 3rd person view. To cancel, press `q` in the GUI, and it will exit + save a video to `dddpo_vids`.

