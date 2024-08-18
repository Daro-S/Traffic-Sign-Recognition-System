from ultralytics import YOLO

class TrafficSignDetector:
    def __init__(self, model_path, target_class_id=11, iou_threshold=0.5):
        self.model = YOLO(model_path)
        self.target_class_id = target_class_id
        self.iou_threshold = iou_threshold
        self.saved_bboxes = []
    
    def detect_and_display(self, video_path, conf=0.4, save=False):
        results = self.model.predict(source=video_path, conf=conf, show=True, save=save)
        return results

    def detect(self, frame):
        results = self.model(frame)
        bboxes = []
        for result in results:
            for box in result.boxes:
                cls = box.cls.item()
                if cls == self.target_class_id:
                    bbox = box.xyxy.cpu().numpy().astype(int)[0]
                    conf = box.conf.item()
                    if not self.is_duplicate(bbox, conf):
                        bboxes.append([*bbox, conf, cls])
        return bboxes

    def is_duplicate(self, new_bbox, new_conf):
        new_bbox_tuple = tuple(new_bbox)
        for bbox, conf in self.saved_bboxes:
            iou = self.compute_iou(new_bbox, bbox)
            if iou > self.iou_threshold:
                if new_conf > conf:
                    self.saved_bboxes.remove((bbox, conf))
                    self.saved_bboxes.append((new_bbox_tuple, new_conf))
                return True
        self.saved_bboxes.append((new_bbox_tuple, new_conf))
        return False

    def compute_iou(self, bbox1, bbox2):
        x1, y1, x2, y2 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        xA = max(x1, x1_2)
        yA = max(y1, y1_2)
        xB = min(x2, x2_2)
        yB = min(y2, y2_2)
        
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (x2 - x1 + 1) * (y2 - y1 + 1)
        boxBArea = (x2_2 - x1_2 + 1) * (y2_2 - y1_2 + 1)
        
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou
