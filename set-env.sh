PWD=$(pwd)

export FUEL_DATA_PATH=$PWD/datasets
export PYTHONPATH=$PWD/libs/Lasagne:$PWD/libs/Theano:$PWD/libs/fuel:$PWD/libs/picklable-itertools:$PWD/libs/progressbar2:$PYTHONPATH
export THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,nvcc.fastmath=True
