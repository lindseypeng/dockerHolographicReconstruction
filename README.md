This is a docker image made for image processing scripts for in line holographic microscopy .

The base image is from https://github.com/waleedka/modern-deep-learning-docker, which includes most of the image processing libraries and
deep learning libraries as well


the docker file simply adds another few layers of tifffile and imutils, as well as the script for microscopy reconstruction

TO USE THIS :

sudo docker run -it -v /absolutepathtohostmountfolder:/app/data holographtest2 /bin/bash

python DHMcombined.py -i location/to/fileread/indatafolder -o location/to/filesave/indatafolder -ci startingframe -cf endframe

EXAMPLE:
suppose that my file is in /home/alinsi/Desktop/DHMdata
I can choose to mount to the general location /home/alinsi to the /app/data folder inside the container

sudo docker run -it -v /home/alinsi:/app/data holographtest2 /bin/bash
python DHMcombined.py -i /app/data/Desktop/DHMdata -o /app/data/Desktop/DHMdata -ci 1 -cf 100

