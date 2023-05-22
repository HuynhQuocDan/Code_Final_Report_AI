import tkinter
from PIL import ImageTk
from PIL import Image,ImageOps
from tkinter import filedialog
import cv2 as cv
from keras.models import load_model
import numpy as np

np.set_printoptions(suppress=True)

# Load the model
model = load_model(r'converted_keras\keras_model.h5', compile=False)

#Giao diện bìa
def run():
    global anhbia
    anhbia.destroy()
#Tạo cửa sổ giao diện
anhbia = tkinter.Tk() #Đặt cửa sổ giao diện tên anhbia
anhbia.geometry("1280x720")
anhbia.title("While Blood Cell Detection")
# Mô hình ảnh bìa
anh=Image.open("anhbia.jpg")
# Chỉnh kích thước ảnh
resizeimage=anh.resize((1280, 720))
a = ImageTk.PhotoImage(resizeimage)
img=tkinter.Label(image=a)
img.grid(column=0,row=0)
#Nút nhấn START
Btn=tkinter.Button(anhbia,text="START",font=("Constantia",20,'bold'),bg= 'green',fg='black',command= run )
Btn.place(x=200,y=620)
anhbia.mainloop() #Lặp lại câu lệnh của anhbia để hiện cửa sổ liên tục = giữ cửa sổ hiển thị
#Set up cái đối tượng trong giao diện chính (trang tiếp theo)
class Main:
    #Thiết lập ban đầu
    xAxis = 0 #Vị trí x của winframe
    yAxis = 0 #Vị trí y của winframe
    MainWindow = 0 #Cửa sổ hiển thị và làm việc hiển thị hình và phím nhấn View
    MainObj = 0 #Đối tượng chính(hình ảnh)
    winFrame = object() #Khung viền
    btnClose = object() #nút nhấn Close
    image = object() #Hình ảnh WBC
    callingObj = object() #Biến giá trị khi đã Browse ảnh
    labelImg = 0 #Hiển thị ảnh WBC
    
    #Hàm con 
    def __init__(self, mainObj, MainWin, wWidth, wHeight, Object, xAxis=10, yAxis=10):
        self.xAxis = xAxis #gán x=10
        self.yAxis = yAxis #gán y=10
        self.MainWindow = MainWin 
        self.MainObj = mainObj
        self.MainWindow.title("White Blood Cell Detection") #Tiêu đề cửa sổ
        if (self.callingObj != 0): #gán giá trị cho biến callingObj khi đã Browser ảnh
            self.callingObj = Object

        global winFrame #cài đặt các giá trị của khung
        self.winFrame = tkinter.Frame(self.MainWindow, width=wWidth, height=wHeight) #nằm trong cửa sổ, chiều dài, chiều rộng
        self.winFrame.place(x=xAxis, y=yAxis) #Vị trí hiện khung

        #Cài đặt nút nhất Close
        self.btnClose = tkinter.Button(self.winFrame, text="CLOSE", font=("Cambria", 20,'bold'),bg='red',fg='white',
                                      command=lambda: self.quitProgram(self.MainWindow)) #Chức năng nút nhấn
        self.btnClose.place(x=100, y=580)

    #Hàm con đóng cửa sổ làm việc
    def quitProgram(self, window):
        global MainWindow #tạo Mainwindow là biến toàn cục
        self.MainWindow.destroy()

    #Hàm con 
    def getFrames(self):
        global winFrame #tạo winFrame là biến toàn cục
        return self.winFrame

    #Xóa bộ nhớ đã cấp phát cho nút nhấn Close
    def removeComponent(self):
        self.btnClose.destroy()

    #Hàm con đọc ảnh
    def readImage(self, img):
        self.image = img

    #Hàm con hiện ảnh WBC
    def displayImage(self):
        imgTk = self.image.resize((480, 340), Image.Resampling.LANCZOS) #Chỉnh lại kích thước ảnh
        imgTk = ImageTk.PhotoImage(image=imgTk) #mở ảnh sau khi xử lý
        self.image = imgTk #cho thuộc tính ảnh bằng ảnh imgTk
        self.labelImg = tkinter.Label(self.winFrame, image=self.image) #Ảnh hiện thị trong khung winFrame và ảnh hiển thị là ảnh sau khi xử lý
        self.labelImg.place(x=700, y=120) #Chọn vị trí hiển thị ảnh


# Giao diện chính 
class Gui:
    #Thiết lập ban đầu
    MainWindow = 0
    listOfWinFrame = list()
    FirstFrame = object()
    val = 0
    fileName = 0
    DT = object()
    wHeight = 720
    wWidth = 1280
    # Cài đặt giao diện
    def __init__(self):
        
        #Global(toàn cục)
        global MainWindow
        #Tạo một cửa sổ bằng lệnh TK()
        MainWindow = tkinter.Tk()
        #Kích thước
        MainWindow.geometry('1280x720')
        MainWindow.resizable(width=False, height=False)
        
        #Lấy dữ liệu từ Main vào Firstframe
        self.FirstFrame = Main(self, MainWindow, self.wWidth, self.wHeight, 0, 0)
        
        #Thêm FisrtFrame vào listOfWinFrame
        self.listOfWinFrame.append(self.FirstFrame)
        
        #Tạo label tiêu đề cho giao diện chính
        WindowLabel = tkinter.Label(self.FirstFrame.getFrames(), text="WHITE BLOOD DETECTION", height=1, width=25)
        WindowLabel.place(x=320, y=30)
        WindowLabel.configure(background="purple", font=("Cambria", 30, "bold"),fg='white')
        
        # Ảnh trong giao diện 
        anh=Image.open("wbc.jpg")
        # Chỉnh kích thước và vị trí
        resizeimage=anh.resize((400, 360))
        a = ImageTk.PhotoImage(resizeimage)
        img=tkinter.Label(image=a)
        img.place(x=100,y=120)
        
        # Nút nhấn Browse
        browsebtn = tkinter.Button(self.FirstFrame.getFrames(), text="Browse", width=15, font=("Cambria", 20,'bold'),bg='white',fg='black', command=self.browseWindow)
        browsebtn.place(x=470, y=500)
        
        # Nút nhấn Detect
        dtbtn = tkinter.Button(self.FirstFrame.getFrames(), text="Detect WBC", width=15, font=("Cambria", 20,'bold'),bg='black',fg='white', command=self.check)
        dtbtn.place(x=470, y=580)

        MainWindow.mainloop()

    def getListOfWinFrame(self):
        return self.listOfWinFrame
        
    # Cài đặt browse ảnh
    def browseWindow(self):
        global wbcImage                     
        FILEOPENOPTIONS = dict(defaultextension=('*.jpeg'),
                               filetypes=[('jpeg', '*.jpeg'), ('jpg', '*.jpg'), ('png', '*.png'),  ('All Files', '*.*')])
        self.fileName = filedialog.askopenfilename(**FILEOPENOPTIONS)
        image = Image.open(self.fileName)
        imageName = str(self.fileName)
        wbcImage = cv.imread(imageName, 1)
        self.listOfWinFrame[0].readImage(image)
        self.listOfWinFrame[0].displayImage()

    #Cài đặt nút detect
    def check(self):
     # Load labels
     class_names = open(r'converted_keras\labels.txt', 'r').readlines()
    
     # Tạo 1 mảng của hình ban đầu để phù hợp keras model
     data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

     #Đưa ảnh vào
     image=Image.open(self.fileName).convert('RGB')

     #Resize ảnh 
     size = (224, 224)
     image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

     #Chuyển về dạng mảng
     image_array = np.asarray(image)

     # Chuẩn hóa ảnh
     normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

     # Load ảnh để chuẩn hóa xuống dạng mảng
     data[0] = normalized_image_array

     # Chạy giao diện
     prediction = model.predict(data)
     index = np.argmax(prediction)
     class_name = class_names[index]
     confidence_score = prediction[0][index]
     self.listOfWinFrame = 0
     self.listOfWinFrame = list()
     self.listOfWinFrame.append(self.FirstFrame) 

     # Label tên bạch cầu
     wbcLabel = tkinter.Label(self.FirstFrame.getFrames(), text= (class_name), width=24)
     wbcLabel.configure(background="pink", font=("Cambria", 20, "bold"), fg="black")
     wbcLabel.place(x=770,y=470) 

     #Label độ tin cậy
     accLabel = tkinter.Label(self.FirstFrame.getFrames(), 
                              text= ("Confidence", "Score:", round(confidence_score*100,3),'%'), width=24)
     accLabel.configure(background="black", font=("Cambria", 20, "bold"), fg="pink")
     accLabel.place(x=770,y=525)   
     return prediction

mainObj = Gui()