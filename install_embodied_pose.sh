# Additional python lib
python -m pip install lxml joblib numpy-stl

# Install mujoco for visualization
python -m pip install -U 'mujoco-py<2.2,>=2.1'
# download mujoco210 from https://github.com/openai/mujoco-py#install-and-use-mujoco-py and move it to ~/.mujoco/mujoco210/bin

# Add the following to your .bashrc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/USER/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so