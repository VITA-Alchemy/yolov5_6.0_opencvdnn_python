# yolov5_6.0_opencvdnn_python
基于numpy改写ultralytics/yolov5 v6.0 opencv推理代码，无需依赖pytorch;前处理后处理使用numpy代替了pytroch。

# Environment
```bash
pytorch >= 1.7.0    # export.py  need
opencv-python >= 4.5.4
onnxruntime == 1.5.2
```

# RUN
```bash
git clone -b v6.0 https://github.com/ultralytics/yolov5.git
download https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt
python export.py --simplify --opset 12 
#官方版测试
python detect.py --weights ./yolov5s.onnx --dnn
python main_dnn.py
```

# Reference
```bash
https://github.com/ultralytics/yolov5
https://github.com/hpc203/yolov5-dnn-cpp-python
```
