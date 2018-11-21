#base image with OpenCV 3.4, Python 3.6 on Debian.
FROM waleedka/modern-deep-learning

WORKDIR /app
RUN mkdir /app/data
COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

#copy the current directory contents into the container at the working aidrectory
COPY . .
 
#Run program when the container launches
CMD ["/bin/bash"]
