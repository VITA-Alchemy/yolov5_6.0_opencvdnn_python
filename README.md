# yolov5_6.0_opencvdnn_python
基于opencv与numpy改写ultralytics版yolov5-v6.0 opencv推理代码，无需依赖pytorch。

# RUN
git clone -b v6.0 https://github.com/ultralytics/yolov5.git
download https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt
python export.py --simplify --opset 12 
python main_dnn.py
