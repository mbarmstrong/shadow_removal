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


$BUILDDIR/ColorConvertUnitTest_Solution -i $DATADIR/plt.ppm -o $OUTDIR/color_convert_ut.csv -t image

