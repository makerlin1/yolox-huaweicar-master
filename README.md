# 2021年华为云人工智能大赛-目标检测部分代码
YOLOX用于交通标识检测
#### 训练yolox
1.参考[官方安装依赖](https://github.com/Megvii-BaseDetection/YOLOX)
* 安装相关依赖
```
pip3 install -r requirements.txt
python3 setup.py develop
```

* 获取apex,安装相关依赖
```
git clone https://github.com/NVIDIA/apex
cd apex
pip3 install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
* 安装pycocotools
```
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```
2.新建文件夹dataset

```
mkdir dataset
```

3.将数据集放在以下参考路径
```
\YOLOX-HUAWEICAR\dataset\VOCdevkit\VOC2021
```

4.开始训练
```
#训练
python tools/train.py -f yolox_voc_s.py --fp16 -d 1 -b 8 -o -c yolox_s.pth 
#评测
python tools/eval.py -f yolox_voc_s.py -c best_ckpt.pth.tar -b 8 -d 1 --conf 0.001
#注 best_ckpt.pth.tar 为指定的加载模型参数的路径，应以实际的路径为依据
# -d[device]:显卡卡数
# -b[batchsize]
```
可视化推理
```
python tools/demo.py "image" --path datasets/dataset_origin/data_val --save_result -f yolox_voc_tiny.py -c YOLOX_outputs_h2_65/yolox_voc_tiny/best_ckpt.pth.tar --device 1 --conf 0.01 --nms 0.3 --fp16
```
5.修改配置文件
```
修改配置文件yolox_voc_s.py的模型超参数，如训练策略，图像增强等
```

6.导出模型为onnx
```
python3 tools/export_onnx.py --output-name yolox_s_h4.onnx -f yolox_voc_s.py -c best_ckpt.pth.tar
```
7.转换为mindspore推理格式
```
#注意环境依赖
conda install mindspore=1.3
pip install onnx, onnxruntime,onnxoptimizer
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp #加速转换
mindconverter --model_file yolox_s_h4.onnx  --shape 1,3,640,640 --input_nodes images --output_nodes output  --report
```
mindspore 的推理样例见ms_inference.ipynb

8.视频标注
依赖：opencv-python
```
python yolox/utils/ann_tool.py -fp [保存图像的目录] -path [视频路径]
#按键W:表示不保存读取下一帧。
#按键S:表示保存并读取下一帧。
```

9.图像相似度筛查
```
python yolox/utils/image_filter.py -fp [图像文件夹] -thres 0.5
#会生成result.json文件，其中字典的keys即为独立的图像，对应的value即为相似的图像名。
```
筛选后的数据集 data-F:[链接](http://pan-yz.chaoxing.com/external/m/file/633745651781562368)
9.更新日志
```
8.15.2021 上传yolox文件
8.14.2021 上传标注程序ann_tool.py，图像相似度筛查程序，onnx导出与转换为mindspore方法。

```