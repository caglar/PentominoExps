#!/bin/bash -x

PYSCRIPT=$1

if [[ -n "$PYSCRIPT" ]]; then
    export PYTHONPATH=../codes/:$PYTHONPATH
    THEANO_FLAGS="floatX=float32,device=gpu0,force_device=True,lib.cnmem=0.85" python $PYSCRIPT
fi
