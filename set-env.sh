PWD=$(pwd)

export FUEL_DATA_PATH=/pio/scratch/1/i247948/
export PYTHONPATH=$PWD/libs/Lasagne:$PWD/libs/Theano:$PWD/libs/fuel:$PWD/libs/picklable-itertools:$PWD/libs/progressbar2:$PYTHONPATH
export THEANO_FLAGS=device=gpu
