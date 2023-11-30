import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import ImageTk, Image
import cv2
import numpy as np

# Model config
model_weights = "./model/yolov3.weights"
model_config = "./yolov3.cfg"
net = cv2.dnn.readNetFromDarknet(model_config, model_weights)
class_names = []
with open("./coco.names", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# global val
global_image_path = ""
global_output = ""
global_save_folder = ""
conf_threshold = 0.5
nms_threshold = 0.4


def run_recognize(img_path):
    image = cv2.imread(img_path)
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    conf_threshold = 0.5
    nms_threshold = 0.4

    if conf_threshold_entry.get() and nms_threshold_entry.get():
        conf_threshold = float(conf_threshold_entry.get())
        nms_threshold = float(nms_threshold_entry.get())
        print(conf_threshold, nms_threshold)

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
    return image


def load_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png")]
    )
    global global_image_path
    global_image_path = file_path
    if file_path:
        image = Image.open(file_path)

        # Lấy kích thước của image_label
        width = image_label.winfo_width()
        height = image_label.winfo_height()

        # Tính toán tỷ lệ giữa kích thước hình ảnh và image_label
        aspect_ratio = min(width / image.width, height / image.height)

        # Tính toán kích thước mới của hình ảnh
        new_width = int(image.width * aspect_ratio)
        new_height = int(image.height * aspect_ratio)

        # Thay đổi kích thước hình ảnh
        image = image.resize((new_width, new_height), Image.LANCZOS)

        photo = ImageTk.PhotoImage(image)
        image_label.configure(image=photo)
        image_label.image = photo


def save_image():
    output_image_path = (
        global_save_folder
        + "/"
        + global_image_path.split("/")[
            -1
        ]  # Thay đổi đường dẫn và tên tệp tin theo ý muốn
    )
    cv2.imwrite(output_image_path, global_output)
    messagebox.showinfo("Thông báo", "Lưu ảnh thành công!")


def select_output_folder():
    folder_path = filedialog.askdirectory()
    if folder_path:
        output_folder_entry.delete(0, tk.END)  # Xóa nội dung trước đó (nếu có)
        output_folder_entry.insert(0, folder_path)
        global global_save_folder
        global_save_folder = folder_path


def on_slider_change(val):
    print(val)
    return


def process_image():
    # Hàm xử lý hình ảnh
    # output_image_path = (
    #     "output_image_23.jpg"  # Thay đổi đường dẫn và tên tệp tin theo ý muốn
    # )
    if global_image_path:
        image_output = run_recognize(global_image_path)
        global global_output
        global_output = image_output

        # cv2.imwrite(output_image_path, image_output)
        image_pil = Image.fromarray(image_output)

        # Lấy kích thước của image_label
        width = image_label_output.winfo_width()
        height = image_label_output.winfo_height()

        # Tính toán tỷ lệ giữa kích thước hình ảnh và image_label
        aspect_ratio = min(width / image_pil.width, height / image_pil.height)

        # Tính toán kích thước mới của hình ảnh
        new_width = int(image_pil.width * aspect_ratio)
        new_height = int(image_pil.height * aspect_ratio)

        # Thay đổi kích thước hình ảnh
        image_pil = image_pil.resize((new_width, new_height))

        photo = ImageTk.PhotoImage(image_pil)
        image_label_output.configure(image=photo)
        image_label_output.image = photo
        messagebox.showinfo("Thông báo", "Nhận diện thành công!")


root = tk.Tk()


# make window size
def relative_size():
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    window_width = int(screen_width * 0.8)  # 80% chiều rộng màn hình
    window_height = int(screen_height * 0.8)  # 80% chiều cao màn hình
    window_size = f"{window_width}x{window_height}"
    root.geometry(window_size)


root.geometry("800x500")

# Chia giao diện làm đôi
left_frame = tk.Frame(root, width=350, height=400)
left_frame.place(x=0, y=10)

middle_frame = tk.Frame(root, width=80, height=400)
middle_frame.place(x=350, y=10)

right_frame = tk.Frame(root, width=350, height=400)
right_frame.place(x=430, y=10)

bottom_frame = tk.Frame(root, width=800, height=100)
bottom_frame.place(x=0, y=400)

# Phần bên trái
load_button = tk.Button(
    left_frame, text="Tải ảnh", width=10, height=1, command=load_image
)
load_button.place(x=10, y=0)

image_label = tk.Label(left_frame, borderwidth=1, relief="solid")
image_label.place(x=10, y=30, width=300, height=300)

# Phần giữa
process_button = tk.Button(middle_frame, text="Process", command=process_image)
process_button.place(x=0, y=0)

# Phần bên phải
save_button = tk.Button(
    right_frame, text="Lưu ảnh", width=10, height=1, command=save_image
)
save_button.place(x=10, y=0)


image_label_output = tk.Label(right_frame, borderwidth=1, relief="solid")
image_label_output.place(x=10, y=30, width=300, height=300)

output_folder_label = tk.Label(right_frame, text="Thư mục đầu ra:")
output_folder_label.place(x=0, y=320)

output_folder_entry = tk.Entry(right_frame, width=40)
output_folder_entry.place(x=0, y=350)
# output_folder_entry.grid(row=0, column=1, padx=10)
select_folder_button = tk.Button(
    right_frame, text="Chọn thư mục", command=select_output_folder
)
select_folder_button.place(x=200, y=350)

# Phần dưới
conf_threshold_label = tk.Label(bottom_frame, text="Conf_threshold : ")
conf_threshold_label.place(x=0, y=20)

conf_threshold_entry = tk.Entry(bottom_frame, width=40)
conf_threshold_entry.place(x=120, y=20)

nms_threshold_label = tk.Label(bottom_frame, text="Nms_threshold : ")
nms_threshold_label.place(x=380, y=20)

nms_threshold_entry = tk.Entry(bottom_frame, width=40)
nms_threshold_entry.place(x=480, y=20)


root.mainloop()
