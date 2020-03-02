from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from numpy import mean
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.utils import Dataset
from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
import tensorflow as tf
import os

class KangarooDataset(Dataset):
	def load_dataset(self, dataset_dir, is_train=True):
		self.add_class("dataset", 1, "kangaroo")
		images_dir = dataset_dir + '/images/'
		annotations_dir = dataset_dir + '/annots/'
		for filename in listdir(images_dir):
			image_id = filename[:-4]
			if is_train and int(image_id) >= 150:
				continue
			if not is_train and int(image_id) < 150:
				continue
			img_path = images_dir + filename
			ann_path = annotations_dir + image_id + '.xml'
			self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

	def extract_boxes(self, filename):
		tree = ElementTree.parse(filename)
		root = tree.getroot()
		boxes = list()
		for box in root.findall('.//bndbox'):
			xmin = int(box.find('xmin').text)
			ymin = int(box.find('ymin').text)
			xmax = int(box.find('xmax').text)
			ymax = int(box.find('ymax').text)
			coors = [xmin, ymin, xmax, ymax]
			boxes.append(coors)
		width = int(root.find('.//size/width').text)
		height = int(root.find('.//size/height').text)
		return boxes, width, height

	def load_mask(self, image_id):
		info = self.image_info[image_id]
		path = info['annotation']
		boxes, w, h = self.extract_boxes(path)
		masks = zeros([h, w, len(boxes)], dtype='uint8')
		class_ids = list()
		for i in range(len(boxes)):
			box = boxes[i]
			row_s, row_e = box[1], box[3]
			col_s, col_e = box[0], box[2]
			masks[row_s:row_e, col_s:col_e, i] = 1
			class_ids.append(self.class_names.index('kangaroo'))
		return masks, asarray(class_ids, dtype='int32')

	def image_reference(self, image_id):
		info = self.image_info[image_id]
		return info['path']

class KangarooConfig(Config):
	NAME = "kangaroo_cfg"
	NUM_CLASSES = 1 + 1
	STEPS_PER_EPOCH = 132

	IMAGES_PER_GPU = 1

class PredictionConfig(Config):
	NAME = "kangaroo_cfg"
	NUM_CLASSES = 1 + 1
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1


def evaluate_model(dataset, model, cfg):
	APs = list()
	for image_id in dataset.image_ids:
		image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
		scaled_image = mold_image(image, cfg)
		sample = expand_dims(scaled_image, 0)
		yhat = model.detect(sample, verbose=0)
		r = yhat[0]
		AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
		APs.append(AP)
	mAP = mean(APs)
	return mAP

# Run this to use GPU in testing. Does not work in evaluation 
# gpus = tf.config.experimental.list_physical_devices('GPU')

# for gpu in gpus:
# 	tf.config.experimental.set_memory_growth(gpu, True)
# 	logical_gpus = tf.config.experimental.list_logical_devices('GPU')
# 	print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

# Forces tensorflow to use CPU instead of GPU. Used for evaluation as GPU does not work for some reason.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

train_set = KangarooDataset()
train_set.load_dataset('kangaroo', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))
test_set = KangarooDataset()
test_set.load_dataset('kangaroo', is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))
cfg = PredictionConfig()
model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
model.load_weights('mask_rcnn_kangaroo_cfg_0004.h5', by_name=True)
train_mAP = evaluate_model(train_set, model, cfg)
print("Train mAP: %.3f" % train_mAP)

test_mAP = evaluate_model(test_set, model, cfg)
print("Test mAP: %.3f" % test_mAP)

# # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# train_set = KangarooDataset()
# train_set.load_dataset('kangaroo', is_train=True)
# train_set.prepare()
# print('Train: %d' % len(train_set.image_ids))
# # prepare test/val set
# test_set = KangarooDataset()
# test_set.load_dataset('kangaroo', is_train=False)
# test_set.prepare()
# print('Test: %d' % len(test_set.image_ids))
# # prepare config
# config = KangarooConfig()
# config.display()
# # define the model
# model = MaskRCNN(mode='training', model_dir='./', config=config)
# # load weights (mscoco) and exclude the output layers
# model.load_weights('code/mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
# # train weights (output layers or 'heads')
# model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads')