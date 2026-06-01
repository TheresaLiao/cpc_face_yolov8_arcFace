# docker settings
imagename="yolov8face_arcface_202405_cpc:v00"
#containername=`date +"%s%6N"`
containername="cpc_face"
workdir="/home/faceRecog_yolov8face_arcface"

xhost +local:


docker run --rm \
--gpus all \
-v /etc/localtime:/etc/localtime:ro \
-e DISPLAY=$DISPLAY \
-p 8086:8086 \
--net host \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-v $(pwd)/img_db:${workdir}/img_db \
--name $containername \
--volume $(pwd):$workdir \
-it $imagename /bin/bash \
-c "
python3 script/main_poc_v4h.py
"

