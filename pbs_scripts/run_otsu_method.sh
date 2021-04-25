#!/bin/bash

BUILDDIR=$HOME/ece569/build_dir;
DATADIR=$HOME/ece569/shadow_removal/data;
OUTDIR=$HOME/ece569/shadow_removal/output;
#DATADIR=$BUILDDIR/TestImages/Dataset/0;

#if [ ! -d "$BUILDDIR/TestImages" ] 
#then
#    echo "Generating test data"
#    cd $BUILDDIR
#    $BUILDDIR/Image_DatasetGenerator
#fi


$BUILDDIR/OtsuUnitTest_Solution -i $DATADIR/plt4.ppm -o $OUTDIR/outsu_ut.csv -t image

