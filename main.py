import cv2
import numpy as np

model_weights = "./model/yolov3.weights"
model_config = "./yolov3.cfg"

net = cv2.dnn.readNetFromDarknet(model_config, model_weights)

class_names = []
with open("./coco.names", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

conf_threshold = 0.8
nms_threshold = 0.4


def run_recognize(img_path):
    image = cv2.imread(img_path)
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    net.setInput(blob)
    output_layers = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(output_layers)

    height, width = image.shape[:2]
    boxes = []
    class_ids = []
    confidences = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > conf_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                bbox_width = int(detection[2] * width)
                bbox_height = int(detection[3] * height)
                x = int(center_x - bbox_width / 2)
                y = int(center_y - bbox_height / 2)

                boxes.append([x, y, bbox_width, bbox_height])
                class_ids.append(class_id)
                confidences.append(float(confidence))
        # Áp dụng Non-maximum suppression để loại bỏ các khu vực trùng lắp
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # Vẽ khung bao quanh vùng quan tâm và ghi nhãn
    for i in indices:
        x, y, w, h = boxes[i]
        label = class_names[class_ids[i]]
        confidence = confidences[i]

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"{label}: {confidence:.2f}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
    output_image_path = (
        "outputs/image_23.jpg"  # Thay đổi đường dẫn và tên tệp tin theo ý muốn
    )

    cv2.imwrite(output_image_path, image)
    print("Đã xuất ra tấm hình kết quả.")


run_recognize("./images/1_mxzb.jpg")
