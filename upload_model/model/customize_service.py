import threading
import logging
import time
import mindspore
from mindinfer import Model, preproc,multiclass_nms, demo_postprocess,get_logger
from mindspore import load_checkpoint, load_param_into_net, context, Tensor
from PIL import Image
import cv2
import numpy as np
from model_service.model_service import SingleNodeService
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

class YOLOX_service(SingleNodeService):
    def __init__(self, model_name, model_path):
        self.model_name = model_name
        self.model_path = model_path
        logger.info("self.model_name:%s self.model_path: %s", self.model_name, self.model_path)
        self.network = None
        # 非阻塞方式加载模型，防止阻塞超时
        thread = threading.Thread(target=self.load_model)
        thread.start()
        self.input_shape = (640,640)
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.ratio = None
        self.CLASSES = ("green_go", "pedestrian_crossing", "red_stop", "speed_limited", "speed_unlimited", "yellow_back")

    def load_model(self):
        logger.info("load network ... \n")
        self.network  = Model()
        # 需要根据实际模型更改名称
        ckpt_file = self.model_path + "/yolox_s.ckpt"
        logger.info("ckpt_file: %s", ckpt_file)
        param_dict = load_checkpoint(ckpt_file)
        load_param_into_net(self.network, param_dict)
        self.network.set_train(False)
        # 模型预热，否则首次推理的时间会很长
        self.network_warmup()
        logger.info("load network successfully ! \n")
    
    def network_warmup(self):
        # 模型预热，否则首次推理的时间会很长
        logger.info("warmup network ... \n")
        images = np.array(np.random.randn(1, 3, 640, 640), dtype=np.float32)
        inputs = Tensor(images, mindspore.float32)
        _ = self.network(inputs)
        logger.info("warmup network successfully ! \n")

    def _preprocess(self, input_data):
        images = []
        for k, v in input_data.items():
            for file_name, file_content in v.items():
                image = Image.open(file_content)
                image = np.array(image)[:,:,::-1]
                img, self.ratio = preproc(image, self.input_shape, self.mean, self.std)
                img = img.reshape(1,3,640,640)
                images.append(Tensor(img))
        return images[0]

    def _inference(self, image):
        t1 = time.time()
        output = self.network(image).asnumpy()
        predictions = demo_postprocess(output[0], self.input_shape, p6=False)
        t2 = time.time()
        print("infer time: ", t1 - t2)
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
        boxes_xyxy /= self.ratio
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
        if dets is not None:
            return dets[:, :4], dets[:, 4], dets[:, 5]
        else:
            return None
        
    def _postprocess(self, data):
        result = {"detection_classes":[], 'detection_boxes':[], 'detection_scores':[]}
        if data is  not None:
            final_boxes, final_scores, final_cls_inds = data
            for i in range(final_cls_inds.shape[0]):
                xmin, ymin, xmax, ymax = final_boxes[i]
                score = final_scores[i]
                cls= self.CLASSES(final_cls_inds[i])
                result['detection_classes'].append(cls)
                result['detection_boxes'].append([float(ymin), float(xmin), float(ymax), float(xmax)])
                result['detection_scores'].append(float(score))
        return result