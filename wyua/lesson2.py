import mmcv
import mmengine
from mmdet.apis import init_detector, inference_detector
from mmdet.apis import DetInferencer
from mmdet.utils import register_all_modules
# Choose to use a config and initialize the detector
config_file = '/gemini/code/mmdetection/configs/mask_rcnn/mask-rcnn_r50-caffe_fpn_ms-poly-3x_coco.py'
# Setup a checkpoint file to load
checkpoint_file = '/gemini/code/mmdetection/demo/checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'

# register all modules in mmdet into the registries
register_all_modules()

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')  # or device='cuda:0'


# Use the detector to do inference
# image = mmcv.imread('/gemini/code/mmdetection/demo/demo.jpg',channel_order='rgb')
# result = inference_detector(model, image)
# print(result)

# Initialize the DetInferencer
inferencer = DetInferencer(config_file, checkpoint_file, device='cuda:0')

# Use the detector to do inference
img = './demo/cat.jpeg'
result = inferencer(img, out_dir='./output')


"""
from mmdet.registry import VISUALIZERS
# init visualizer(run the block only once in jupyter notebook)
visualizer = VISUALIZERS.build(model.cfg.visualizer)
# the dataset_meta is loaded from the checkpoint and
# then pass to the model in init_detector
visualizer.dataset_meta = model.dataset_meta

# show the results
visualizer.add_datasample(
    'result',
    image,
    data_sample=result,
    draw_gt = None,
    wait_time=0,
)
visualizer.show()
"""