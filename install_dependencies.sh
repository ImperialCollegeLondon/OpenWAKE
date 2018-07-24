curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install
python3 firedrake-install --install thetis
#python3 -m pip install git+https://github.com/firedrakeproject/firedrake.git
#python3 -m pip install git+https://github.com/thetisproject/thetis.git
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade setuptools
python3 -m pip install ez_setup
python3 -m pip install numpy
python3 -m pip install matplotlib
python3 -m pip install scipy
python3 -m easy_install ipython
python3 -m pip install patsy
python3 -m pip install pandas
python3 -m pip install git+https://bitbucket.org/dolfin-adjoint/pyadjoint.git@2017.2.0
python3 -m pip install git+https://github.com/FUSED-Wind/windIO.git
python3 -m pip install utm
python3 -m pip install pyAMI
python3 -m pip install jsonschema
python3 -m pip install plotly

#python3 -m pip install git+https://github.com/OpenTidalFarm/OpenTidalFarm.git
#python3 -m pip install git+https://bitbucket.org/fenics-project/dolfin.git
