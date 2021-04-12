#!/bin/sh

BASEDIR=`pwd`

if [ ! -d "$BASEDIR" ]; then
    mkdir "$BASEDIR"
fi


# fb15k-237
if [ ! -d "$BASEDIR/fb15k-237" ]; then
    echo Downloading fb15k-237
    cd $BASEDIR
    curl -O http://web.informatik.uni-mannheim.de/pi1/kge-datasets/fb15k-237.tar.gz
    tar xvf fb15k-237.tar.gz
else
    echo fb15k-237 already present
fi
if [ ! -f "$BASEDIR/fb15k-237/dataset.yaml" ]; then
    python preprocess/preprocess_default.py fb15k-237
else
    echo fb15k-237 already prepared
fi

# yago3-10
if [ ! -d "$BASEDIR/yago3-10" ]; then
    echo Downloading yago3-10
    cd $BASEDIR
    curl -O http://web.informatik.uni-mannheim.de/pi1/kge-datasets/yago3-10.tar.gz
    tar xvf yago3-10.tar.gz
else
    echo yago3-10 already present
fi
if [ ! -f "$BASEDIR/yago3-10/dataset.yaml" ]; then
    python preprocess/preprocess_default.py yago3-10
else
    echo yago3-10 already prepared
fi