#!/bin/bash

IN=$HOME/TFG/videoReferencial/ref.mp4
VIEWER=nvoverlaysink
#VIEWER=fakesink

# 0 : none    
# 1 : ERROR   
# 2 : WARNING 
# 3 : FIXME   
# 4 : INFO    
# 5 : DEBUG   
# 6 : LOG     
# 7 : TRACE   
# 9 : MEMDUMP 
export GST_DEBUG=3

# 1 - entrada, salida = video
gst-launch-1.0 -v -e filesrc location=$IN ! qtdemux name=in qtmux name=out ! filesink location=res.mp4 in.video_0 ! h264parse ! nvv4l2decoder ! nvvidconv bl-output=false ! nvvidtrans filter=$HOME/TFG/libdeproject.so fargs=197,138,1267,205,168,631,1111,647 ! nvvidtrans filter=$HOME/TFG/libfilter.so ! 'video/x-raw(memory:NVMM), format=(string)NV12' ! nvv4l2h264enc ! h264parse ! queue ! out. in.audio_0 ! avdec_aac ! audioconvert ! voaacenc ! queue ! out.

#gst-launch-1.0 -v -e filesrc location=$IN ! qtdemux ! h264parse ! nvv4l2decoder ! nvvidconv bl-output=false ! nvvidtrans filter=$HOME/TFG/libdeproject.so fargs=197,138,1267,205,168,631,1111,647 ! nvvidtrans filter=$HOME/TFG/libfilter.so ! 'video/x-raw(memory:NVMM), format=(string)NV12' ! omxh264enc ! h264parse ! qtmux ! filesink location=res.mp4

#gst-launch-1.0 filesrc location=res.mp4 ! qtdemux ! queue ! h264parse ! nvv4l2decoder ! nv3dsink -e

# entrada, salidda=directo
#gst-launch-1.0 -v -e v4l2src ! nvvidconv bl-output=false ! capsfilter caps='video/x-raw(memory:NVMM), format=NV12' ! nvvidtrans filter=$HOME/TFG/libdeproject.so fargs=197,138,1267,205,168,631,1111,647 ! nvvidtrans filter=$HOME/TFG/libfilter.so ! 'video/x-raw(memory:NVMM), format=(string)NV12' ! nvvidconv flip-method=horizontal-flip ! identity drop-allocation=true ! v4l2sink device=/dev/video10

# entrada=video, salida = en directo
#gst-launch-1.0 -v -e filesrc location=$IN ! qtdemux ! h264parse ! omxh264dec ! nvvidconv bl-output=false ! nvvidtrans filter=$HOME/TFG/libdeproject.so fargs=197,138,1267,205,168,631,1111,647 ! nvvidtrans filter=$HOME/TFG/libfilter.so ! 'video/x-raw(memory:NVMM),format=(string)NV12' ! nvvidconv flip-method=horizontal-flip ! identity drop-allocation=true ! v4l2sink device=/dev/video10