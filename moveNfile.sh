#!/bin/bash

echo "Moving $1 files..."

InputPath="/home/optalysys/OptalysysSoftware/externalapi/Samples/ChemInformatics/SampleData/Inputs/"
FilterPath="/home/optalysys/OptalysysSoftware/externalapi/Samples/ChemInformatics/SampleData/PretransformedFilters/"


for files in $(ls -U ./SingleImages/Reference_Page_*.png | head -$1)
do	       
    cp $files $InputPath
done

for files in $(ls -U ./SingleImages/Filter*.png | head -$1)
do	       
    cp $files $FilterPath
done

