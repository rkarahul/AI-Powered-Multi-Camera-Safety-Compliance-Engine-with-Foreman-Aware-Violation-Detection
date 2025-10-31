# import cv2
# import torch
# import numpy as np
# from queue import Queue
# import time
# import torchvision
# import base64
# from shapely.geometry import Point, Polygon
# import traceback
# from new_cap import process_image
# from scipy.spatial.distance import cdist  # For potential distance merging, but using IoU here

# def mylogger(text):
#     try:
#         with open("backlogger.txt","a") as logfile:
#             logfile.write(text + "\n")
#     except:
#         pass

# class ObjectDetection:    
#     def __init__(self, weights):
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         self.model = self.load_model(weights)
#         self.model.to(self.device)
#         #self.model.to(self.device).eval()
#         self.classes = self.model.names
#         print("\n\nDevice Used:", self.device)


#     def load_model(self, weights):
#         model = torch.hub.load('yolov5', 'custom', path=weights,source='local')#,force_reload=True)
#         print("loaded_model")
#         return model

#     def xywh2xyxy(self,x):
#         # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
#         y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
#         y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
#         y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
#         y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
#         y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
#         return y
    
#     def box_iou(self,box1, box2, eps=1e-7):
#         """
#         Return intersection-over-union (Jaccard index) of boxes.
#         Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
#         Arguments:
#             box1 (Tensor[N, 4])
#             box2 (Tensor[M, 4])
#         Returns:
#             iou (Tensor[N, M]): the NxM matrix containing the pairwise
#                 IoU values for every element in boxes1 and boxes2
#         """

#         # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
#         (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
#         inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

#         # IoU = inter / (area1 + area2 - inter)
#         return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

#     def non_max_suppression(self,
#         prediction,
#         conf_thres=0.5,
#         iou_thres=0.45,
#         classes=None,
#         agnostic=False,
#         multi_label=False,
#         labels=(),
#         max_det=300,
#         nm=0
#         ):

#         """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

#         Returns:
#             list of detections, on (n,6) tensor per image [xyxy, conf, cls]
#         """

#         # Checks
#         assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
#         assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
#         if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
#             prediction = prediction[0]  # select only inference output

#         device = prediction.device
#         mps = 'mps' in device.type  # Apple MPS
#         if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
#             prediction = prediction.cpu()
#         bs = prediction.shape[0]  # batch size
#         nc = prediction.shape[2] - nm - 5  # number of classes
#         xc = prediction[..., 4] > conf_thres  # candidates

#         # Settingszz
#         # min_wh = 2  # (pixels) minimum box width and height
#         max_wh = 7680  # (pixels) maximum box width and height
#         max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
#         time_limit = 0.5 + 0.05 * bs  # seconds to quit after
#         redundant = True  # require redundant detections
#         multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
#         merge = False  # use merge-NMS

#         t = time.time()
#         mi = 5 + nc  # mask start column index
#         output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
#         for xi, x in enumerate(prediction):  # image index, image inference
#             # Apply constraints
#             # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
#             x = x[xc[xi]]  # confidence

#             # Cat apriori labels if autolabelling
#             if labels and len(labels[xi]):
#                 lb = labels[xi]
#                 v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
#                 v[:, :4] = lb[:, 1:5]  # box
#                 v[:, 4] = 1.0  # conf
#                 v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
#                 x = torch.cat((x, v), 0)

#             # If none remain process next image
#             if not x.shape[0]:
#                 continue

#             # Compute conf
#             x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

#             # Box/Mask
#             box = self.xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
#             mask = x[:, mi:]  # zero columns if no masks

#             # Detections matrix nx6 (xyxy, conf, cls)
#             if multi_label:
#                 i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
#                 x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
#             else:  # best class only
#                 conf, j = x[:, 5:mi].max(1, keepdim=True)
#                 x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

#             # Filter by class
#             if classes is not None:
#                 x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

#             # Apply finite constraint
#             # if not torch.isfinite(x).all():
#             #     x = x[torch.isfinite(x).all(1)]

#             # Check shape
#             n = x.shape[0]  # number of boxes
#             if not n:  # no boxes
#                 continue
#             x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

#             # Batched NMS
#             c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
#             boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
#             i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
#             i = i[:max_det]  # limit detections
#             if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
#                 # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
#                 iou = self.box_iou(boxes[i], boxes) > iou_thres  # iou matrix
#                 weights = iou * scores[None]  # box weights
#                 x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
#                 if redundant:
#                     i = i[iou.sum(1) > 1]  # require redundancy

#             output[xi] = x[i]
#             if mps:
#                 output[xi] = output[xi].to(device)
#             if (time.time() - t) > time_limit:
#                 print(f'WARNING NMS time limit {time_limit:.3f}s exceeded')
#                 break  # time limit exceeded

#         return output


#     def prepro_image(self, imgs):
#         # imgs: list of 3 images (BGR numpy arrays)
#         batch = []
#         for img in imgs:
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             img = torch.from_numpy(img.transpose((2, 0, 1))).float()
#             img /= 255.0
#             batch.append(img)
#         batch = torch.stack(batch, dim=0)  # Shape: [3, 3, H, W]
#         return batch.to(self.device)

# # ==== Load Detection Model ====
# detectionmodel = ObjectDetection(weights=r'./yolov5/best.pt')

# dectparam = {
#     ((149,267), (364,259), (484,264), (638,284),(639,256),(639,218),(789,214),
#      (1013,227),(1278,281),(1280,354), (1377,327),(1596,276), (1764,244), (1796,393),(1658,440),
#      (1458, 486),(1279,498),(1279,420),(1009,410),(809,386),(639,349),(639,423),(371,442),(129,433)): 2
# }

# def compute_iou_numpy(box1, box2):
#     """Compute IoU between two numpy boxes [x1,y1,x2,y2]."""
#     # Intersection
#     x1_inter = np.maximum(box1[0], box2[0])
#     y1_inter = np.maximum(box1[1], box2[1])
#     x2_inter = np.minimum(box1[2], box2[2])
#     y2_inter = np.minimum(box1[3], box2[3])
#     inter_area = np.maximum(0, x2_inter - x1_inter) * np.maximum(0, y2_inter - y1_inter)
    
#     # Areas
#     area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
#     area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
#     union_area = area1 + area2 - inter_area
#     iou = inter_area / (union_area + 1e-7)
#     return iou

# def apply_mask(frame, mask_start, mask_end=None):
#     """Apply black mask to specified x-range of frame."""
#     height, width = frame.shape[:2]
#     mask = np.zeros((height, width), dtype=np.uint8)
#     if mask_end is None:
#         mask[:, mask_start:] = 255  # Mask from start to end
#     else:
#         mask[:, mask_start:mask_end] = 255  # Mask between start and end
#     frame_masked = cv2.bitwise_and(frame, frame, mask=~mask)
#     return frame_masked

# def score_img(frame_list):
#     # Apply masks to frames
#     masked_frames = []
#     masked_frames.append(apply_mask(frame_list[0], 515))  # Image 1: Mask from 500 to end
#     masked_frames.append(apply_mask(frame_list[1], 525))  # Image 2: Mask from 565 to end
#     masked_frames.append(apply_mask(frame_list[2], 0, 85))  # Image 3: Mask from 0 to 443

#     # Prepare safe frames
#     safe_frames = []
#     for i in range(3):
#         f = masked_frames[i] if i < len(masked_frames) and masked_frames[i] is not None else np.zeros((640, 640, 3), np.uint8)
#         if f.shape[:2] != (640, 640):
#             f = cv2.resize(f, (640, 640))
#         safe_frames.append(f)

#     # Batch inference
#     batch_imgs = detectionmodel.prepro_image(safe_frames)
#     results = detectionmodel.model(batch_imgs)
#     dets_list = detectionmodel.non_max_suppression(results, conf_thres=0.4, iou_thres=0.65)

#     # Polygon setup
#     roi_tuple = list(dectparam.keys())[0]
#     poly_points = [(p[0], p[1]) for p in roi_tuple]
#     poly = Polygon(poly_points)

#     # Collect all person detections with metadata
#     all_detections = []
#     for i in range(3):
#         dets = dets_list[i]
#         if len(dets) == 0:
#             print(f"No detections in frame {i}")
#             continue
#         boxes = dets[:, :4].cpu().numpy().astype(np.int32)
#         confs = dets[:, 4].cpu().numpy()
#         clss = dets[:, 5].cpu().numpy().astype(int)

#         # Filter for persons (class 0)
#         person_mask = clss == 0
#         boxes = boxes[person_mask]
#         confs = confs[person_mask]
#         cam_index = np.full(len(boxes), i)

#         for j in range(len(boxes)):
#             x1, y1, x2, y2 = boxes[j]
#             conf = confs[j]
#             # print(f"Cam {i}, Det {j}: Box {x1},{y1},{x2},{y2}, Conf {conf:.2f}")

#             # Head crop for cap detection
#             head_height = (y2 - y1) // 2
#             is_cap = False
#             if head_height > 20:
#                 head_y_end = y1 + head_height
#                 hx1 = max(0, x1)
#                 hx2 = min(640, x2)
#                 head_crop = safe_frames[i][y1:head_y_end, hx1:hx2].copy()
#                 if head_crop.shape[0] > 0 and head_crop.shape[1] > 0:
#                     _, head_counts = process_image(head_crop)
#                     red_counts = [red_count for _, _, _, _, red_count in head_counts]
#                     # print(f"Cam {i}, Det {j}: Red pixel counts = {red_counts}")
#                     if any(red_count > 5 for red_count in red_counts):  # Reduced threshold
#                         is_cap = True
#                         print(f"Cam {i}, Det {j}: Identified as Foreman with red pixel count > 5")
#                     else:
#                         print(f"Cam {i}, Det {j}: Red pixel counts = {red_counts}")
                        
#             # Store local box and metadata
#             all_detections.append({
#                 'cam': i, 'local_box': [x1, y1, x2, y2], 'conf': conf, 'is_cap': is_cap,
#                 'global_box': [x1 + i*640, y1, x2 + i*640, y2]  # Global x offset
#             })

#     if not all_detections:
#         # No persons, no offense
#         bbox_key = (91, 64, 313, 384)
#         finalresults = {bbox_key: {'count': 0, 'start_time': time.time(), 'offense': False}}
#         return safe_frames, finalresults

#     # Sort by confidence descending for greedy merging
#     all_detections.sort(key=lambda d: d['conf'], reverse=True)

#     # Group detections (greedy merge on global IoU > 0.3)
#     groups = []
#     used = set()
#     for idx, det in enumerate(all_detections):
#         if idx in used:
#             continue
#         group = {'dets': [det], 'is_foreman': det['is_cap']}
#         used.add(idx)

#         # Check for merges with subsequent detections
#         for jdx in range(idx + 1, len(all_detections)):
#             if jdx in used:
#                 continue
#             merge_det = all_detections[jdx]
#             iou = compute_iou_numpy(det['global_box'], merge_det['global_box'])
#             if iou > 0.3:  # Merge threshold for overlaps
#                 group['dets'].append(merge_det)
#                 group['is_foreman'] = group['is_foreman'] or merge_det['is_cap']
#                 used.add(jdx)
#                 # print(f"Merged dets {idx} and {jdx} (IoU {iou:.2f}), Foreman: {group['is_foreman']}")

#         groups.append(group)

#     # Process groups for offenses
#     has_offense = False
#     violation_count = 0
#     for g_id, group in enumerate(groups):
#         dets = group['dets']
#         is_foreman = group['is_foreman']
#         # print(f"Group {g_id}: {len(dets)} dets, Foreman: {is_foreman}")

#         # Representative bottom-center: average across group
#         bottom_centers_x = []
#         bottom_centers_y = []
#         for det in dets:
#             x1, y1, x2, y2 = det['local_box']
#             mid_y = (y1 + y2) / 2.0
#             bcy = (mid_y + y2) / 2.0
#             bcx_global = (det['global_box'][0] + det['global_box'][2]) / 2.0  # Global x
#             bottom_centers_x.append(bcx_global)
#             bottom_centers_y.append(bcy)
#         rep_x = np.mean(bottom_centers_x)
#         rep_y = np.mean(bottom_centers_y)
#         point = Point(rep_x, rep_y)

#         is_violation = not poly.contains(point) and not is_foreman
#         if is_violation:
#             has_offense = True
#             violation_count += 1
#             print(f"Group {g_id}: Offense detected (outside polygon, not foreman)")

#         # Assign color and label for all dets in group
#         color = (0, 255, 0)  # Green default
#         label_base = "Person"
#         if is_foreman:
#             color = (0, 0, 255)  # Red for foreman
#             label_base += " (Foreman)"
#         if is_violation:
#             color = (255, 105, 180)  # Pink for offender
#             label_base += " OFFENSE!"

#         for det in dets:
#             conf = det['conf']
#             i = det['cam']
#             x1, y1, x2, y2 = det['local_box']
#             full_label = f"{label_base} {conf:.2f}"
#             cv2.rectangle(safe_frames[i], (x1, y1), (x2, y2), color, 2)
#             cv2.putText(safe_frames[i], full_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#     # Prepare finalresults to match expected format
#     bbox_key = (91, 64, 313, 384)
#     finalresults = {bbox_key: {'count': violation_count, 'start_time': time.time(), 'offense': has_offense}}

#     return safe_frames, finalresults

# def paramreset(reset_flag):
#     # Placeholder; can be extended if needed
#     return "ok"


# import cv2
# import torch
# import numpy as np
# from queue import Queue
# import time
# import torchvision
# import base64
# from shapely.geometry import Point, Polygon
# import traceback
# from new_cap import process_image

# def mylogger(text):
#     try:
#         with open("backlogger.txt","a") as logfile:
#             logfile.write(text + "\n")
#     except:
#         pass

# class ObjectDetection:    
#     def __init__(self, weights):
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         self.model = self.load_model(weights)
#         self.model.to(self.device)
#         self.classes = self.model.names
#         print("\n\nDevice Used:", self.device)

#     def load_model(self, weights):
#         model = torch.hub.load('yolov5', 'custom', path=weights, source='local')
#         print("loaded_model")
#         return model

#     def xywh2xyxy(self, x):
#         y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
#         y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
#         y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
#         y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
#         y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
#         return y
    
#     def box_iou(self, box1, box2, eps=1e-7):
#         (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
#         inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
#         return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

#     def non_max_suppression(self,
#         prediction,
#         conf_thres=0.5,
#         iou_thres=0.45,
#         classes=None,
#         agnostic=False,
#         multi_label=False,
#         labels=(),
#         max_det=300,
#         nm=0
#         ):

#         assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
#         assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
#         if isinstance(prediction, (list, tuple)):
#             prediction = prediction[0]

#         device = prediction.device
#         mps = 'mps' in device.type
#         if mps:
#             prediction = prediction.cpu()
#         bs = prediction.shape[0]
#         nc = prediction.shape[2] - nm - 5
#         xc = prediction[..., 4] > conf_thres

#         max_wh = 7680
#         max_nms = 30000
#         time_limit = 0.5 + 0.05 * bs
#         redundant = True
#         multi_label &= nc > 1
#         merge = False

#         t = time.time()
#         mi = 5 + nc
#         output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
#         for xi, x in enumerate(prediction):
#             x = x[xc[xi]]

#             if labels and len(labels[xi]):
#                 lb = labels[xi]
#                 v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
#                 v[:, :4] = lb[:, 1:5]
#                 v[:, 4] = 1.0
#                 v[range(len(lb)), lb[:, 0].long() + 5] = 1.0
#                 x = torch.cat((x, v), 0)

#             if not x.shape[0]:
#                 continue

#             x[:, 5:] *= x[:, 4:5]

#             box = self.xywh2xyxy(x[:, :4])
#             mask = x[:, mi:]

#             if multi_label:
#                 i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
#                 x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
#             else:
#                 conf, j = x[:, 5:mi].max(1, keepdim=True)
#                 x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

#             if classes is not None:
#                 x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

#             n = x.shape[0]
#             if not n:
#                 continue
#             x = x[x[:, 4].argsort(descending=True)[:max_nms]]

#             c = x[:, 5:6] * (0 if agnostic else max_wh)
#             boxes, scores = x[:, :4] + c, x[:, 4]
#             i = torchvision.ops.nms(boxes, scores, iou_thres)
#             i = i[:max_det]
#             if merge and (1 < n < 3E3):
#                 iou = self.box_iou(boxes[i], boxes) > iou_thres
#                 weights = iou * scores[None]
#                 x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)
#                 if redundant:
#                     i = i[iou.sum(1) > 1]

#             output[xi] = x[i]
#             if mps:
#                 output[xi] = output[xi].to(device)
#             if (time.time() - t) > time_limit:
#                 print(f'WARNING NMS time limit {time_limit:.3f}s exceeded')
#                 break

#         return output

#     def prepro_image(self, imgs):
#         batch = []
#         for img in imgs:
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             img = torch.from_numpy(img.transpose((2, 0, 1))).float()
#             img /= 255.0
#             batch.append(img)
#         batch = torch.stack(batch, dim=0)
#         return batch.to(self.device)

# # Load Model
# detectionmodel = ObjectDetection(weights=r'./yolov5/best.pt')

# dectparam = {
#     ((149,267), (364,259), (484,264), (638,284),(639,256),(639,218),(789,214),
#      (1013,227),(1278,281),(1280,354), (1377,327),(1596,276), (1764,244), (1796,393),(1658,440),
#      (1458, 486),(1279,498),(1279,420),(1009,410),(809,386),(639,349),(639,423),(371,442),(129,433)): 2
# }
# edge_threshold = 100  # Consider 'edge' if box crosses within this
# y_overlap_threshold = 0.5  # 50% vertical overlap
# size_similarity_threshold = 0.2  # Heights within 20%
# h_dist_threshold = 50  # Max horizontal gap after offset

# def compute_iou_numpy(box1, box2):
#     x1_inter = np.maximum(box1[0], box2[0])
#     y1_inter = np.maximum(box1[1], box2[1])
#     x2_inter = np.minimum(box1[2], box2[2])
#     y2_inter = np.minimum(box1[3], box2[3])
#     inter_area = np.maximum(0, x2_inter - x1_inter) * np.maximum(0, y2_inter - y1_inter)
    
#     area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
#     area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
#     union_area = area1 + area2 - inter_area
#     return inter_area / (union_area + 1e-7)

# def propagate_cap_status(all_detections):
#     """
#     Propagate cap status across adjacent cameras for edge/partial views.
#     - Cam 0 right -> Cam 1 left
#     - Cam 1 right -> Cam 2 left
#     """
#     # Group by camera
#     cam_dets = {0: [], 1: [], 2: []}
#     for det in all_detections:
#         cam_dets[det['cam']].append(det)

#     # Define edge thresholds
   

#     # Propagate cam0 right -> cam1 left
#     propagate_between(cam_dets[0], cam_dets[1], direction='right_to_left', offset=640)

#     # Propagate cam1 right -> cam2 left
#     propagate_between(cam_dets[1], cam_dets[2], direction='right_to_left', offset=640)

# def propagate_between(cam_from, cam_to, direction, offset):
#     """
#     Propagate cap status from edge of cam_from to start of cam_to.
#     direction: 'right_to_left' for current setup (cam0 right to cam1 left).
#     offset: Horizontal offset between cameras (640px).
#     """
#     for from_det in cam_from:
#         x1, y1, x2, y2 = from_det['local_box']
#         if direction == 'right_to_left' and x2 > 640 - edge_threshold:  # Right edge of from cam
#             for to_det in cam_to:
#                 tx1, ty1, tx2, ty2 = to_det['local_box']
#                 if tx1 < edge_threshold:  # Left edge of to cam
#                     # Check vertical overlap
#                     y_overlap = max(0, min(y2, ty2) - max(y1, ty1)) / min(y2 - y1, ty2 - ty1)
#                     if y_overlap > y_overlap_threshold:
#                         # Check size similarity
#                         h_ratio = abs((y2 - y1) - (ty2 - ty1)) / max(y2 - y1, ty2 - ty1)
#                         if h_ratio < size_similarity_threshold:
#                             # Check horizontal proximity (global)
#                             from_global_right = from_det['global_box'][2]
#                             to_global_left = to_det['global_box'][0] + offset  # Adjust for continuity
#                             h_dist = abs(to_global_left - from_global_right)
#                             if h_dist < h_dist_threshold:
#                                 # Propagate cap if better in 'to' (full view)
#                                 if not from_det['is_cap'] and to_det['is_cap']:
#                                     from_det['is_cap'] = True
#                                     print(f"Propagated cap from Cam {to_det['cam']} left to Cam {from_det['cam']} right (overlap {y_overlap:.2f}, dist {h_dist}px)")

# def score_img(frame_list):
#     # Prepare safe frames
#     safe_frames = []
#     for i in range(3):
#         f = frame_list[i] if i < len(frame_list) and frame_list[i] is not None else np.zeros((640, 640, 3), np.uint8)
#         if f.shape[:2] != (640, 640):
#             f = cv2.resize(f, (640, 640))
#         safe_frames.append(f)

#     # Batch inference
#     batch_imgs = detectionmodel.prepro_image(safe_frames)
#     results = detectionmodel.model(batch_imgs)
#     dets_list = detectionmodel.non_max_suppression(results, conf_thres=0.5, iou_thres=0.65)

#     # Polygon setup
#     roi_tuple = list(dectparam.keys())[0]
#     poly_points = [(p[0], p[1]) for p in roi_tuple]
#     poly = Polygon(poly_points)

#     # Collect all person detections with metadata, filtering small ROIs
#     all_detections = []
#     for i in range(3):
#         dets = dets_list[i]
#         if len(dets) == 0:
#             print(f"No detections in frame {i}")
#             continue
#         boxes = dets[:, :4].cpu().numpy().astype(np.int32)
#         confs = dets[:, 4].cpu().numpy()
#         clss = dets[:, 5].cpu().numpy().astype(int)

#         person_mask = clss == 0
#         boxes = boxes[person_mask]
#         confs = confs[person_mask]

#         for j in range(len(boxes)):
#             x1, y1, x2, y2 = boxes[j]
#             width = x2 - x1
#             height = y2 - y1
#             if width <= 70 or height <= 120:
#                 print(f"Cam {i}, Det {j}: Skipped (width={width}, height={height} <= 70x120)")
#                 continue
#             conf = confs[j]
#             print(f"Cam {i}, Det {j}: Box {x1},{y1},{x2},{y2}, Conf {conf:.2f}")

#             # Head crop for cap detection
#             head_height = (y2 - y1) // 2
#             is_cap = False
#             if head_height > 20:
#                 head_y_end = y1 + head_height
#                 hx1 = max(0, x1)
#                 hx2 = min(640, x2)
#                 head_crop = safe_frames[i][y1:head_y_end, hx1:hx2].copy()
#                 if head_crop.shape[0] > 0 and head_crop.shape[1] > 0:
#                     _, head_counts = process_image(head_crop)
#                     red_counts = [red_count for _, _, _, _, red_count in head_counts]
#                     print(f"Cam {i}, Det {j}: Red pixel counts = {red_counts}")
#                     if any(red_count > 5 for red_count in red_counts):
#                         is_cap = True
#                         print(f"Cam {i}, Det {j}: Identified as Foreman with red pixel count > 5")

#             all_detections.append({
#                 'cam': i, 'local_box': [x1, y1, x2, y2], 'conf': conf, 'is_cap': is_cap,
#                 'global_box': [x1 + i*640, y1, x2 + i*640, y2]
#             })

#     if not all_detections:
#         bbox_key = (91, 64, 313, 384)
#         finalresults = {bbox_key: {'count': 0, 'start_time': time.time(), 'offense': False}}
#         return safe_frames, finalresults

#     # Propagate cap status across cameras
#     propagate_cap_status(all_detections)

#     # Sort by confidence descending for greedy merging
#     all_detections.sort(key=lambda d: d['conf'], reverse=True)

#     # Group detections (greedy merge on global IoU > 0.3)
#     groups = []
#     used = set()
#     for idx, det in enumerate(all_detections):
#         if idx in used:
#             continue
#         group = {'dets': [det], 'is_foreman': det['is_cap']}
#         used.add(idx)

#         for jdx in range(idx + 1, len(all_detections)):
#             if jdx in used:
#                 continue
#             merge_det = all_detections[jdx]
#             iou = compute_iou_numpy(det['global_box'], merge_det['global_box'])
#             if iou > 0.3:
#                 group['dets'].append(merge_det)
#                 group['is_foreman'] = group['is_foreman'] or merge_det['is_cap']
#                 used.add(jdx)
#                 print(f"Merged dets {idx} and {jdx} (IoU {iou:.2f}), Foreman: {group['is_foreman']}")

#         groups.append(group)

#     # Process groups for offenses
#     has_offense = False
#     violation_count = 0
#     for g_id, group in enumerate(groups):
#         dets = group['dets']
#         is_foreman = group['is_foreman']
#         print(f"Group {g_id}: {len(dets)} dets, Foreman: {is_foreman}")

#         # Representative bottom-center
#         bottom_centers_x = []
#         bottom_centers_y = []
#         for det in dets:
#             x1, y1, x2, y2 = det['local_box']
#             mid_y = (y1 + y2) / 2.0
#             bcy = (mid_y + y2) / 2.0
#             bcx_global = (det['global_box'][0] + det['global_box'][2]) / 2.0
#             bottom_centers_x.append(bcx_global)
#             bottom_centers_y.append(bcy)
#         rep_x = np.mean(bottom_centers_x)
#         rep_y = np.mean(bottom_centers_y)
#         point = Point(rep_x, rep_y)

#         is_violation = not poly.contains(point) and not is_foreman
#         if is_violation:
#             has_offense = True
#             violation_count += 1
#             print(f"Group {g_id}: Offense detected (outside polygon, not foreman)")

#         # Assign color and label
#         color = (0, 255, 0)
#         label_base = "Person"
#         if is_foreman:
#             color = (0, 0, 255)
#             label_base += " (Foreman)"
#         if is_violation:
#             color = (255, 105, 180)
#             label_base += " OFFENSE!"

#         for det in dets:
#             conf = det['conf']
#             i = det['cam']
#             x1, y1, x2, y2 = det['local_box']
#             full_label = f"{label_base} {conf:.2f}"
#             cv2.rectangle(safe_frames[i], (x1, y1), (x2, y2), color, 2)
#             cv2.putText(safe_frames[i], full_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#     # Final results
#     bbox_key = (91, 64, 313, 384)
#     finalresults = {bbox_key: {'count': violation_count, 'start_time': time.time(), 'offense': has_offense}}

#     return safe_frames, finalresults

# def paramreset(reset_flag):
#     return "ok"


import cv2
import torch
import numpy as np
from queue import Queue
import time
import torchvision
import base64
import traceback
from new_cap import process_image

def mylogger(text):
    try:
        with open("backlogger.txt","a") as logfile:
            logfile.write(text + "\n")
    except:
        pass

class ObjectDetection:    
    def __init__(self, weights):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.load_model(weights)
        self.model.to(self.device)
        self.classes = self.model.names
        print("\n\nDevice Used:", self.device)

    def load_model(self, weights):
        model = torch.hub.load('yolov5', 'custom', path=weights, source='local')
        print("loaded_model")
        return model

    def xywh2xyxy(self, x):
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
        y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
        y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
        y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
        return y
    
    def box_iou(self, box1, box2, eps=1e-7):
        (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
        inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
        return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

    def non_max_suppression(self,
        prediction,
        conf_thres=0.5,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nm=0
        ):

        assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
        assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
        if isinstance(prediction, (list, tuple)):
            prediction = prediction[0]

        device = prediction.device
        mps = 'mps' in device.type
        if mps:
            prediction = prediction.cpu()
        bs = prediction.shape[0]
        nc = prediction.shape[2] - nm - 5
        xc = prediction[..., 4] > conf_thres

        max_wh = 7680
        max_nms = 30000
        time_limit = 0.5 + 0.05 * bs
        redundant = True
        multi_label &= nc > 1
        merge = False

        t = time.time()
        mi = 5 + nc
        output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
        for xi, x in enumerate(prediction):
            x = x[xc[xi]]

            if labels and len(labels[xi]):
                lb = labels[xi]
                v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
                v[:, :4] = lb[:, 1:5]
                v[:, 4] = 1.0
                v[range(len(lb)), lb[:, 0].long() + 5] = 1.0
                x = torch.cat((x, v), 0)

            if not x.shape[0]:
                continue

            x[:, 5:] *= x[:, 4:5]

            box = self.xywh2xyxy(x[:, :4])
            mask = x[:, mi:]

            if multi_label:
                i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
            else:
                conf, j = x[:, 5:mi].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

            n = x.shape[0]
            if not n:
                continue
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

            c = x[:, 5:6] * (0 if agnostic else max_wh)
            boxes, scores = x[:, :4] + c, x[:, 4]
            i = torchvision.ops.nms(boxes, scores, iou_thres)
            i = i[:max_det]
            if merge and (1 < n < 3E3):
                iou = self.box_iou(boxes[i], boxes) > iou_thres
                weights = iou * scores[None]
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)
                if redundant:
                    i = i[iou.sum(1) > 1]

            output[xi] = x[i]
            if mps:
                output[xi] = output[xi].to(device)
            if (time.time() - t) > time_limit:
                print(f'WARNING NMS time limit {time_limit:.3f}s exceeded')
                break

        return output

    def prepro_image(self, imgs):
        batch = []
        for img in imgs:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(img.transpose((2, 0, 1))).float()
            img /= 255.0
            batch.append(img)
        batch = torch.stack(batch, dim=0)
        return batch.to(self.device)

# Load Model
detectionmodel = ObjectDetection(weights=r'./yolov5/best.pt')

# dectparam = {
#     ((1764,244), (1796,393),(1658,440),
#      (1458, 486),(1279,498),(1279,420),(1009,410),(809,386),(639,349),(639,423),(371,442),(129,433)): 2
# }


dectparam = {
    ((1815,244), (1870,360),(1796,393),(1658,440),
     (1458, 486),(1279,498),(1279,420),(1009,410),(809,386),(639,349),(639,423),(371,442),(60,428)): 2
}
edge_threshold = 100  # Consider 'edge' if box crosses within this
y_overlap_threshold = 0.5  # 50% vertical overlap
size_similarity_threshold = 0.2  # Heights within 20%
h_dist_threshold = 50  # Max horizontal gap after offset

def compute_iou_numpy(box1, box2):
    x1_inter = np.maximum(box1[0], box2[0])
    y1_inter = np.maximum(box1[1], box2[1])
    x2_inter = np.minimum(box1[2], box2[2])
    y2_inter = np.minimum(box1[3], box2[3])
    inter_area = np.maximum(0, x2_inter - x1_inter) * np.maximum(0, y2_inter - y1_inter)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = area1 + area2 - inter_area
    return inter_area / (union_area + 1e-7)

def get_line_y_at_x(x_query, line_points):
    """
    Interpolate y on the polyline at given x_query.
    Returns interpolated y or None if x_query outside line range.
    """
    if len(line_points) < 2:
        return None
    
    xs = [p[0] for p in line_points]
    ys = [p[1] for p in line_points]
    
    if x_query < min(xs) or x_query > max(xs):
        return None  # Extrapolation not supported; assume safe if outside
    
    # Find segment
    for k in range(len(xs) - 1):
        if xs[k] <= x_query <= xs[k+1] or xs[k+1] <= x_query <= xs[k]:
            # Linear interpolation
            x1, y1 = xs[k], ys[k]
            x2, y2 = xs[k+1], ys[k+1]
            if x1 == x2:
                return y1  # Vertical segment
            t = (x_query - x1) / (x2 - x1)
            return y1 + t * (y2 - y1)
    
    return None

def propagate_cap_status(all_detections):
    """
    Propagate cap status across adjacent cameras for edge/partial views.
    - Cam 0 right -> Cam 1 left
    - Cam 1 right -> Cam 2 left
    """
    # Group by camera
    cam_dets = {0: [], 1: [], 2: []}
    for det in all_detections:
        cam_dets[det['cam']].append(det)

    # Propagate cam0 right -> cam1 left
    propagate_between(cam_dets[0], cam_dets[1], direction='right_to_left', offset=640)

    # Propagate cam1 right -> cam2 left
    propagate_between(cam_dets[1], cam_dets[2], direction='right_to_left', offset=640)

def propagate_between(cam_from, cam_to, direction, offset):
    """
    Propagate cap status from edge of cam_from to start of cam_to.
    direction: 'right_to_left' for current setup (cam0 right to cam1 left).
    offset: Horizontal offset between cameras (640px).
    """
    for from_det in cam_from:
        x1, y1, x2, y2 = from_det['local_box']
        if direction == 'right_to_left' and x2 > 640 - edge_threshold:  # Right edge of from cam
            for to_det in cam_to:
                tx1, ty1, tx2, ty2 = to_det['local_box']
                if tx1 < edge_threshold:  # Left edge of to cam
                    # Check vertical overlap
                    y_overlap = max(0, min(y2, ty2) - max(y1, ty1)) / min(y2 - y1, ty2 - ty1)
                    if y_overlap > y_overlap_threshold:
                        # Check size similarity
                        h_ratio = abs((y2 - y1) - (ty2 - ty1)) / max(y2 - y1, ty2 - ty1)
                        if h_ratio < size_similarity_threshold:
                            # Check horizontal proximity (global)
                            from_global_right = from_det['global_box'][2]
                            to_global_left = to_det['global_box'][0] + offset  # Adjust for continuity
                            h_dist = abs(to_global_left - from_global_right)
                            if h_dist < h_dist_threshold:
                                # Propagate cap if better in 'to' (full view)
                                if not from_det['is_cap'] and to_det['is_cap']:
                                    from_det['is_cap'] = True
                                    print(f"Propagated cap from Cam {to_det['cam']} left to Cam {from_det['cam']} right (overlap {y_overlap:.2f}, dist {h_dist}px)")

def score_img(frame_list, top_only=False):
    # Prepare safe frames
    safe_frames = []
    frame_height = 640  # Assuming 640x640 frames
    top_roi_threshold = frame_height // 2  # Center to top: filter if bottom y > 320

    for i in range(3):
        f = frame_list[i] if i < len(frame_list) and frame_list[i] is not None else np.zeros((640, 640, 3), np.uint8)
        if f.shape[:2] != (640, 640):
            f = cv2.resize(f, (640, 640))
        safe_frames.append(f)

    # Batch inference
    batch_imgs = detectionmodel.prepro_image(safe_frames)
    results = detectionmodel.model(batch_imgs)
    dets_list = detectionmodel.non_max_suppression(results, conf_thres=0.2, iou_thres=0.45)

    # Baseline polyline setup (not closed polygon)
    roi_tuple = list(dectparam.keys())[0]
    baseline_points = [(p[0], p[1]) for p in roi_tuple]  # List of (x, y) points for the line

    # Collect all person detections with metadata, filtering small ROIs and top_only if enabled
    all_detections = []
    for i in range(3):
        dets = dets_list[i]
        if len(dets) == 0:
            print(f"No detections in frame {i}")
            continue
        boxes = dets[:, :4].cpu().numpy().astype(np.int32)
        confs = dets[:, 4].cpu().numpy()
        clss = dets[:, 5].cpu().numpy().astype(int)

        person_mask = clss == 0
        boxes = boxes[person_mask]
        confs = confs[person_mask]

        for j in range(len(boxes)):
            x1, y1, x2, y2 = boxes[j]
            width = x2 - x1
            height = y2 - y1
            print(f"(width={width}, height={height}")
            if width <= 70 or height <=130:
                print(f"Cam {i}, Det {j}: Skipped (width={width}, height={height} <= 70x120)")
                continue

            # Filter for top ROI only if enabled
            if top_only and y2 > top_roi_threshold:
                # print(f"Cam {i}, Det {j}: Skipped (bottom y={y2} > {top_roi_threshold}, top_only mode)")
                continue

            conf = confs[j]
            # print(f"Cam {i}, Det {j}: Box {x1},{y1},{x2},{y2}, Conf {conf:.2f}")

            # Head crop for cap detection
            head_height = (y2 - y1) // 2
            is_cap = False
            if head_height > 20:
                head_y_end = y1 + head_height
                hx1 = max(0, x1)
                hx2 = min(640, x2)
                head_crop = safe_frames[i][y1:head_y_end, hx1:hx2].copy()
                if head_crop.shape[0] > 0 and head_crop.shape[1] > 0:
                    _, head_counts = process_image(head_crop)
                    red_counts = [red_count for _, _, _, _, red_count in head_counts]
                    print(f"Red pixel counts = {red_counts}")
                    if any(red_count >5 for red_count in red_counts):
                        is_cap = True
                        print(f"Cam {i}, Det {j}: Identified as Foreman with red pixel count > 5")

            all_detections.append({
                'cam': i, 'local_box': [x1, y1, x2, y2], 'conf': conf, 'is_cap': is_cap,
                'global_box': [x1 + i*640, y1, x2 + i*640, y2]
            })

    if not all_detections:
        bbox_key = (91, 64, 313, 384)
        finalresults = {bbox_key: {'count': 0, 'start_time': time.time(), 'offense': False}}
        return safe_frames, finalresults

    # Propagate cap status across cameras
    propagate_cap_status(all_detections)

    # Sort by confidence descending for greedy merging
    all_detections.sort(key=lambda d: d['conf'], reverse=True)

    # Group detections (greedy merge on global IoU > 0.3)
    groups = []
    used = set()
    for idx, det in enumerate(all_detections):
        if idx in used:
            continue
        group = {'dets': [det], 'is_foreman': det['is_cap']}
        used.add(idx)

        for jdx in range(idx + 1, len(all_detections)):
            if jdx in used:
                continue
            merge_det = all_detections[jdx]
            iou = compute_iou_numpy(det['global_box'], merge_det['global_box'])
            if iou > 0.3:
                group['dets'].append(merge_det)
                group['is_foreman'] = group['is_foreman'] or merge_det['is_cap']
                used.add(jdx)
                # print(f"Merged dets {idx} and {jdx} (IoU {iou:.2f}), Foreman: {group['is_foreman']}")

        groups.append(group)

    # Process groups for offenses
    has_offense = False
    violation_count = 0
    for g_id, group in enumerate(groups):
        dets = group['dets']
        is_foreman = group['is_foreman']
        # print(f"Group {g_id}: {len(dets)} dets, Foreman: {is_foreman}")

        # Representative foot coordinate: average bottom-center across group
        bottom_centers_x = []
        bottom_centers_y = []
        for det in dets:
            x1, y1, x2, y2 = det['local_box']
            bcx_global = (det['global_box'][0] + det['global_box'][2]) / 2.0
            bcy = y2  # Foot y is bottom of box
            bottom_centers_x.append(bcx_global)
            bottom_centers_y.append(bcy)
        rep_x = np.mean(bottom_centers_x)
        rep_y = np.mean(bottom_centers_y)

        # Get line y at rep_x
        line_y = get_line_y_at_x(rep_x, baseline_points)
        if line_y is None:
            # If x outside line range, assume no violation (safe)
            is_violation = False
            # print(f"Group {g_id}: x={rep_x} outside baseline range, no violation")
        else:
            is_violation = rep_y > line_y and not is_foreman  # Below line (higher y) and not foreman
            # print(f"Group {g_id}: Foot at ({rep_x:.1f}, {rep_y:.1f}), line y={line_y:.1f}, violation={is_violation}")

        if is_violation:
            has_offense = True
            violation_count += 1
            print(f"Group {g_id}: Offense detected (below baseline, not foreman)")

        # Assign color and label
        color = (0, 255, 0)
        label_base = "Person"
        if is_foreman:
            color = (0, 0, 255)
            label_base += " (Foreman)"
        if is_violation:
            color = (255, 105, 180)
            label_base += " OFFENSE!"
        for det in dets:
            conf = det['conf']
            i = det['cam']
            x1, y1, x2, y2 = det['local_box']
            full_label = f"{label_base} {conf:.2f}"
            cv2.rectangle(safe_frames[i], (x1, y1), (x2, y2), color, 2)
            cv2.putText(safe_frames[i], full_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Final results
    bbox_key = (91, 64, 313, 384)
    finalresults = {bbox_key: {'count': violation_count, 'start_time': time.time(), 'offense': has_offense}}

    return safe_frames, finalresults

def paramreset(reset_flag):
    return "ok"