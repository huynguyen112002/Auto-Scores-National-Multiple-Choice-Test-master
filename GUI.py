import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import imutils
import numpy as np

# Các biến toàn cục để lưu đường dẫn ảnh
input_image_path = ""
output_image_path = "data/preprocessed_image.jpg"
width, height = 2000, 3000

# Đáp án đúng
ANSWER_KEY = {
    1: 'A', 2: 'A', 3: 'B', 4: 'C', 5: 'D',
    6: 'A', 7: 'B', 8: 'A', 9: 'C', 10: 'A'
}

# Hàm tải ảnh
def upload_image():
    global input_image_path
    input_image_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png")]
    )
    if input_image_path:
        image = cv2.imread(input_image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image.thumbnail((500, 500))  # Giảm kích thước ảnh để hiển thị
        img_tk = ImageTk.PhotoImage(image)
        panel_original.config(image=img_tk)
        panel_original.image = img_tk
        
# Hàm xử lý ảnh
def process_image():
    if input_image_path:
        preprocess_image(input_image_path, output_image_path)
        crop_image(output_image_path)
        messagebox.showinfo("Success", "Ảnh đã được xử lý thành công!")
    else:
        messagebox.showwarning("Warning", "Vui lòng tải ảnh trước khi xử lý.")

# Hàm tiền xử lý ảnh (xoay và cắt phối cảnh)
def preprocess_image(input_path, output_path):
    image = cv2.imread(input_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    if len(approx) == 4:
        points = np.array(sorted(approx[:, 0], key=lambda x: (x[1], x[0])))
        top_left, top_right = points[:2]
        bottom_left, bottom_right = points[2:]
        
        if top_left[0] > top_right[0]:
            top_left, top_right = top_right, top_left
        if bottom_left[0] > bottom_right[0]:
            bottom_left, bottom_right = bottom_right, bottom_left
        
        points = np.array([top_left, top_right, bottom_right, bottom_left])
        dst_points = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype="float32")
        
        M = cv2.getPerspectiveTransform(points.astype("float32"), dst_points)
        warped = cv2.warpPerspective(image, M, (width, height))
        cv2.imwrite(output_path, warped)
        
    else:
        messagebox.showwarning("Warning", "Không tìm thấy đủ 4 góc để xử lý phối cảnh.")

# Hàm cắt ảnh và hiển thị kết quả
def crop_image(img):
    image = cv2.imread(img)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    INS_NUM_AREA = [width * 0.7, height * 0.1, width * 0.8, height * 0.25]
    ANSWERS_AREA = [width * 0.1, height * 0.35, width * 0.9, height * 0.7]

    MIN_O = 30
    MAX_O = 50

    # Duyệt qua các contours
    question_cnts = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if w >= MIN_O and h >= MIN_O and w <= MAX_O and h <= MAX_O:
            if x >= ANSWERS_AREA[0] and x <= ANSWERS_AREA[2] and y >= ANSWERS_AREA[1] and y <= ANSWERS_AREA[3]:
                question_cnts.append((c, (x, y, w, h)))

    # Sắp xếp contours theo hàng và cột
    question_cnts = sorted(question_cnts, key=lambda x: (x[1][1], x[1][0]))
    
    # Chấm điểm
    current_question = 1
    correct = 0
    for c, (x, y, w, h) in question_cnts:

        # Đáp án của câu hỏi hiện tại
        correct_answer = ANSWER_KEY.get(current_question, None)

        if correct_answer:
            # Xác định đáp án được chọn
            answer_index = (x - 90) // 100
            selected_answer = chr(64 + answer_index)

            # Tô màu đáp án đúng hoặc sai
            if selected_answer == correct_answer:
                color = (0, 255, 0)
                correct += 1
            else:
                color = (0, 0, 255)

            cv2.drawContours(image, [c], -1, color, 3)

        current_question += 1

    # Hiển thị ảnh kết quả
    score = (correct / 10.0) * 100
    print("score: {:.2f}%".format(score))
    cv2.putText(image, "{:.2f}%".format(score), (600, 900), cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 255, 0), 10)
    
    # Hiển thị ảnh đã cắt trên Tkinter
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image.thumbnail((500, 500))
    img_tk = ImageTk.PhotoImage(image)
    panel_processed.config(image=img_tk)
    panel_processed.image = img_tk

# Tạo giao diện chính
root = tk.Tk()
root.title("Ứng dụng xử lý ảnh trắc nghiệm")
root.geometry("900x700")

# Tạo nút tải ảnh
btn_upload = tk.Button(root, text="Tải ảnh", command=upload_image)
btn_upload.pack(pady=20)

# Khung hiển thị ảnh gốc và ảnh xử lý
frame = tk.Frame(root)
frame.pack()

panel_original = tk.Label(frame)
panel_original.pack(side="left", padx=20, pady=20)

panel_processed = tk.Label(frame)
panel_processed.pack(side="right", padx=20, pady=20)

# Tạo nút xử lý ảnh
btn_process = tk.Button(root, text="Xử lý ảnh", command=process_image)
btn_process.pack(pady=20)

# Chạy giao diện
root.mainloop()
