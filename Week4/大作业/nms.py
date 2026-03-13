import numpy as np

def compute_iou(box, boxes):
    """
    box: [x1, y1, x2, y2]
    boxes: shape (N, 4)
    """
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0, x2 - x1)
    inter_h = np.maximum(0, y2 - y1)
    inter_area = inter_w * inter_h

    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    union_area = box_area + boxes_area - inter_area
    iou = inter_area / (union_area + 1e-6)
    return iou


def nms(boxes, scores, iou_threshold=0.5):
    """
    boxes: shape (N, 4), 每行为 [x1, y1, x2, y2]
    scores: shape (N,)
    return: 保留框的索引
    """
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)

    order = scores.argsort()[::-1]  # 从大到小排序
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        current_box = boxes[i]
        remaining_boxes = boxes[order[1:]]

        ious = compute_iou(current_box, remaining_boxes)

        inds = np.where(ious <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


if __name__ == "__main__":
    boxes = [
        [100, 100, 210, 210],
        [105, 105, 215, 215],
        [150, 150, 260, 260],
        [300, 300, 400, 400]
    ]
    scores = [0.95, 0.90, 0.75, 0.80]

    keep_indices = nms(boxes, scores, iou_threshold=0.5)
    print("保留框索引:", keep_indices)
    print("保留框坐标:", [boxes[i] for i in keep_indices])