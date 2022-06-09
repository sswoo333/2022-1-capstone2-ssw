from importlib.metadata import metadata
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2 import model_zoo

from detectron2.utils.visualizer import ClolorMode

import random 
import cv2
import matplotlib.pyplot as plt

def plot_samples(dataset_name, n=1):
    dataset_custom = DatasetCatalog.get(dataset_name)
    dataset_custom_metadata = MetadataCatalog.get(dataset_name)

    for s in random.sample(dataset_custom, n):
        img = cv2.imread(s["file_name"])
        v = Visualizer(img[:,:,::-1], metadata=dataset_custom_metadata, scale=0.5)
        v = v.draw_dataset_dict(s)
        plt.figure(figsize=(15,20))
        plt.imshow(v.get_image())
        plt.show()

def get_train_cfg(config_file_path, checkpoint_url, train_datasetname, test_dataset_name, num_classes, device, output_dir):
    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file(config_file_path))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkpoint_url)
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (test_dataset_name,)

    cfg.DATALOADER.NUM_WORKERS = 2

    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 1000
    cfg.SOLVER.STEPS = []

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.DEVICE = device
    cfg.OUPUT_DIR = output_dir

    return cfg

def on_image(image_path, predictor) :
    im = cv2.imread(image_path)
    outputs = predictor(im)
    v = Visualizer(im[:,:,::-1], metadata={}, scale=0.5, instance_mode=ColorMode.SEGMENTATION)
    v = v.draw_instance_predictions(outputs["instances".to("cpu")])

    plt.figure(figsize=(14,10))
    plt.imshow(v.get_image())
    plt.show()

def on_video(videoPath, predictor):
    cap = cv2.VideoCapture(videoPath)
    if (cap.isOpened()==False):
        print("Error opening file....")
        return

    (success,image) = cap.read()
    while success:
        predictions = predictor(image)
        v = Visualizer(image[:,:,::-1], metadata={}, scale=0.5, instance_mode=ColorMode.SEGMENTATION)
        output = v.draw_instance_predictions(predictions["instances".to("cpu")])

        cv2.imshow("Result", output.get_image()[:,:,::-1])

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        (success,image) = cap.read()

    
     









    cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")) #yaml 파일 가져오기
cfg.DATASETS.TRAIN = ("category_train",) #dataset 등록
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # 초기 모델 설정
#cfg.MODEL.WEIGHTS = "./output_1_50_50epoch/model_pretrained.pth" #1~50 images (50epoch) pretrained
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025    # 학습률 설정
cfg.SOLVER.MAX_ITER = 100      # Iteration 설정(Detectron2에는 Epoch 지정 따로 X)

#cfg.SOLVER.STEPS = [300000, 600000]
#cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # 본 데이터셋에 대한 속도, 성능 (default: 512)
#cfg.SOLVER.CHECKPOINT_PERIOD = 20000

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # class의 개수