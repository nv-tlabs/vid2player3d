# Addtional python libs
python -m pip install rl-games==1.1.4 pyvista==0.34.2 pyglet==1.5.27 tqdm

# The default install may install `vtk>=9.2.0` which will lead to program crash when running pyvista. 
# A temporary fix is to install vtk in an older version
python -m pip install --ignore-installed vtk==9.1.0

# Downgrade numpy
conda install numpy=1.23.5 --force -c conda-forge