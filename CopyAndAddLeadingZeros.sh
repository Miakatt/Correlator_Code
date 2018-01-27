#!/bin/bash                                                                                                           
for file in /Volumes/SANDISK64G/Picoscope_Data/2017*([0-6798]).csv;
 do
    echo 'copying $file to InnovateUK'
    cp "$file" ./
done
