from mmdet.core import eval_map, eval_recalls
from mmdet.core.evaluation import eval_point_map, eval_point_bbox_map
from ..core import eval_recalls as eval_points_recalls
from mmdet.datasets import DATASETS, XMLDataset
import mmcv
import os.path as osp
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image


@DATASETS.register_module()
class SHIPDataset8(XMLDataset):
    CLASSES = ('burke', 'nimitz', 'freedom', 'wasp', 'ticonderoga', 'bridge', 'radar', 'island')
    NUM_CLASSES = 5
    NUM_PTS_CLASSES = 3
    PALETTE = [(220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
               (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)]

    def __init__(self, **kwargs):
        super(SHIPDataset8, self).__init__(**kwargs)

    def load_annotations(self, ann_file):
        """Load annotation from XML style ann_file.

        Args:
            ann_file (str): Path of XML file.

        Returns:
            list[dict]: Annotation info from XML file.
        """
        data_infos = []
        img_ids = mmcv.list_from_file(ann_file)
        for img_id in img_ids:
            # filename = f'JPEGImages/{img_id}.jpg'
            filename = f'{img_id}.jpg'
            # print(filename)
            # xml_path = osp.join(self.img_prefix, 'Annotations',
            xml_path = osp.join(self.img_prefix,
                                f'{img_id}.xml')
            xml_path = xml_path.replace('ImageSets', 'Annotations')
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            width = 0
            height = 0
            if size is not None:
                width = int(size.find('width').text)
                height = int(size.find('height').text)
            else:
                # img_path = osp.join(self.img_prefix, 'JPEGImages',
                img_path = osp.join(self.img_prefix,
                                    '{}.jpg'.format(img_id))
                img = Image.open(img_path)
                width, height = img.size
            if width == 0 or height == 0:
                img_path = osp.join(self.img_prefix,
                                    '{}.jpg'.format(img_id))
                img = Image.open(img_path)
                width, height = img.size
            data_infos.append(
                dict(id=img_id, filename=filename, width=width, height=height))
        return data_infos

    def get_ann_info(self, idx):
        """Get annotation from XML file by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        # xml_path = osp.join(self.img_prefix, 'Annotations', f'{img_id}.xml')
        xml_path = osp.join(self.img_prefix, f'{img_id}.xml')
        xml_path = xml_path.replace('ImageSets', 'Annotations')
        # print(img_id, xml_path)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bboxes = []
        pbboxes = []
        labels = []
        points = []
        plabels = []
        bboxes_ignore = []
        labels_ignore = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in self.CLASSES:
                continue
            label = self.cat2label[name]
            difficult = int(obj.find('difficult').text)
            bnd_box = obj.find('bndbox')
            # TODO: check whether it is necessary to use int
            # Coordinates may be float type
            bbox = [
                int(float(bnd_box.find('xmin').text)),
                int(float(bnd_box.find('ymin').text)),
                int(float(bnd_box.find('xmax').text)),
                int(float(bnd_box.find('ymax').text))
            ]

            if label >= self.NUM_CLASSES:  # points label
                point = [
                    (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
                ]
                points.append(point)
                plabels.append(label - self.NUM_CLASSES)  # re-label
                pbboxes.append(bbox)

            # ignore = False
            # if self.min_size:
            #     assert not self.test_mode
            #     w = bbox[2] - bbox[0]
            #     h = bbox[3] - bbox[1]
            #     if w < self.min_size or h < self.min_size:
            #         ignore = True
            #         print(xml_path)
            # if difficult or ignore:
            #     bboxes_ignore.append(bbox)
            #     labels_ignore.append(label)
            # else:
            #     bboxes.append(bbox)
            #     labels.append(label)
            bboxes.append(bbox)
            labels.append(label)
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0,))
        else:
            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)
        if not points:
            points = np.zeros((0, 2))
            plabels = np.zeros((0,))
            pbboxes = np.zeros((0, 4))
        else:
            points = np.array(points, ndmin=2) - 1
            plabels = np.array(plabels)
            pbboxes = np.array(pbboxes, ndmin=2) - 1
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0,))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
            labels_ignore = np.array(labels_ignore)

        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            points=points.astype(np.float32),
            plabels=plabels.astype(np.int64),
            pbboxes=pbboxes.astype(np.float32),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        return ann

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None):
        """Evaluate in VOC protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'mAP', 'recall'.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. It must be a float
                when evaluating mAP, and can be a list when evaluating recall.
                Default: 0.5.
            scale_ranges (list[tuple], optional): Scale ranges for evaluating
                mAP. If not specified, all bounding boxes would be included in
                evaluation. Default: None.

        Returns:
            dict[str, float]: AP/recall metrics.
        """
        if isinstance(results[0], tuple):
            assert len(results[0]) == 2
            bbox_result, point_results = [r[0] for r in results], [r[1] for r in results]
        else:
            bbox_result, point_results = results, None
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['mAP', 'recall', 'pts_mAP', 'pb_mAP', 'pts_recall']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = {}
        if 'mAP' in metrics:
            assert isinstance(iou_thr, float)
            ds_name = self.CLASSES
            mean_ap, _ = eval_map(
                bbox_result,
                annotations,
                scale_ranges=None,
                iou_thr=iou_thr,
                dataset=ds_name,
                logger=logger)
            eval_results['mAP'] = mean_ap
        if 'pts_mAP' in metrics:
            # assert isinstance(iou_thr, float)
            ds_name = self.CLASSES[-self.NUM_PTS_CLASSES:]
            mean_ap, _ = eval_point_map(
                point_results,
                annotations,
                scale_ranges=None,
                dis_thr=4,
                dataset=ds_name,
                logger=logger)
            eval_results['pts_mAP_4'] = mean_ap
            mean_ap, _ = eval_point_map(
                point_results,
                annotations,
                scale_ranges=None,
                dis_thr=8,
                dataset=ds_name,
                logger=logger)
            eval_results['pts_mAP_8'] = mean_ap
            mean_ap, _ = eval_point_map(
                point_results,
                annotations,
                scale_ranges=None,
                dis_thr=16,
                dataset=ds_name,
                logger=logger)
            eval_results['pts_mAP_16'] = mean_ap
        if 'pb_mAP' in metrics:
            # assert isinstance(iou_thr, float)
            ds_name = self.CLASSES[-self.NUM_PTS_CLASSES:]
            mean_ap, _ = eval_point_bbox_map(
                point_results,
                annotations,
                scale_ranges=None,
                dis_thr=1,
                dataset=ds_name,
                logger=logger)
            eval_results['pb_mAP'] = mean_ap
        if 'recall' in metrics:
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            if isinstance(iou_thr, float):
                iou_thr = [iou_thr]
            recalls = eval_recalls(
                gt_bboxes, bbox_result, proposal_nums, iou_thr, logger=logger)
            for i, num in enumerate(proposal_nums):
                for j, iou in enumerate(iou_thr):
                    eval_results[f'recall@{num}@{iou}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
        if 'pts_recall' in metrics:
            gt_points = [ann['points'] for ann in annotations]
            # if isinstance(iou_thr, float):
            #     iou_thr = [iou_thr]
            dis_thr = [4]
            recalls = eval_points_recalls(
                gt_points, point_results, proposal_nums, dis_thr, logger=logger)
            for i, num in enumerate(proposal_nums):
                for j, dis in enumerate(dis_thr):
                    eval_results[f'recall@{num}@{dis}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'pAR@{num}'] = ar[i]
        return eval_results

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without annotation."""
        valid_inds = []
        for i, img_info in enumerate(self.data_infos):
            if min(img_info['width'], img_info['height']) < min_size:
                continue
            if self.filter_empty_gt:
                img_id = img_info['id']
                xml_path = osp.join(self.img_prefix,
                                    f'{img_id}.xml')
                xml_path = xml_path.replace('ImageSets', 'Annotations')
                tree = ET.parse(xml_path)
                root = tree.getroot()
                for obj in root.findall('object'):
                    name = obj.find('name').text
                    if name in self.CLASSES:
                        valid_inds.append(i)
                        break
            else:
                valid_inds.append(i)
        return valid_inds


@DATASETS.register_module()
class SHIPDataset5(SHIPDataset8):
    CLASSES = ('burke', 'nimitz', 'freedom', 'wasp', 'ticonderoga')

    def __init__(self, **kwargs):
        super(SHIPDataset5, self).__init__(**kwargs)