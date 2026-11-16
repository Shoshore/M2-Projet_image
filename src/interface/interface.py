from tkinter import * 
from tkinter.filedialog import *
from PIL import Image, ImageTk



def resizeImage(img, newWidth, newHeight):
    oldWidth = img.width()
    oldHeight = img.height()
    if(oldWidth > oldHeight):
        newHeight = int(newHeight*oldHeight/oldWidth)
    else:
        newWidth = int(newWidth*oldWidth/oldHeight)
    newPhotoImage = PhotoImage(width=newWidth, height=newHeight)
    for x in range(newWidth):
        for y in range(newHeight):
            xOld = int(x*oldWidth/newWidth)
            yOld = int(y*oldHeight/newHeight)
            rgb = '#%02x%02x%02x' % img.get(xOld, yOld)
            newPhotoImage.put(rgb, (x, y))
    return newPhotoImage


class customCanvas:
    def __init__(self, main_window, pos_x, pos_y):
        self.main = main_window

        self.canvas = Canvas(self.main.main_canvas, width=50, height=50, background="white")
        self.canvas.pack()
        self.canvas.place(relx=pos_x, rely=pos_y, anchor=CENTER)

    def resizing(self):
        self.canvas_width = self.main.window.winfo_width() - 20
        self.canvas_height = self.main.window.winfo_height() - 20

        self.canvas.config(width=self.canvas_width//2, height=self.canvas_height//2)


class openCanvas(customCanvas):
    def __init__(self, main_window, button_name, pos_x, pos_y):
        super().__init__(main_window, pos_x, pos_y)
        self.img = None
        self.filepath = None

        self.txt = self.canvas.create_text(0, 60, text="")
        self.button=Button(self.canvas, text=button_name, command=self.openimage, anchor=W)
        self.button_window = self.canvas.create_window(5, 5, anchor=NW, window=self.button)
    
    def create_image(self):
        image_size = min(self.main.canvas_width//2, self.main.canvas_height//2-self.main.button_height*2)
        self.resized_image = self.raw_image.resize((int(image_size), int(image_size)))
        self.img = ImageTk.PhotoImage(self.resized_image)
        self.canvas.create_image(self.main.canvas_width/4, self.main.button_height+self.main.canvas_height/4, anchor=CENTER, image=self.img)
        self.canvas.delete(str(self.txt))
        self.txt = self.canvas.create_text(self.canvas_width/2-10, self.main.button_height, text=self.filepath, anchor=E)

    def resizing(self):
        super().resizing()
        if self.img != None:
            self.create_image()

    def openimage(self):
        self.filepath = askopenfilename(title="Ouvrir une image",filetypes=[('png files','.png'),('all files','.*')])
        self.raw_image = Image.open(self.filepath)
        self.create_image()


class optionCanvas(customCanvas):
    def __init__(self, main_window, pos_x, pos_y):
        super().__init__(main_window, pos_x, pos_y)
        self.img = None
        self.filepath = None

        options = ["Algo1", "Algo2"]  
        opt = StringVar(value="Algo1")  
        self.optionMenu = OptionMenu(self.canvas, opt, *options)
        self.options_window = self.canvas.create_window(5, 5, anchor=NW, window=self.optionMenu)

        self.txt = self.canvas.create_text(0, 60, text="")
        self.button=Button(self.canvas, text="process", command=self.main.generate_results, anchor=W)
        self.button_window = self.canvas.create_window(5, 40, anchor=NW, window=self.button)

    def resizing(self):
        super().resizing()
        if self.img != None:
            self.create_image()


class resultCanvas(customCanvas):
    def __init__(self, main_window, button_name, pos_x, pos_y):
        super().__init__(main_window, pos_x, pos_y)
        self.img = None
        self.filepath = None

        self.txt = self.canvas.create_text(0, 60, text="")
        self.button=Button(self.canvas, text=button_name, command=self.openimage, anchor=W)
        self.button_window = self.canvas.create_window(5, 5, anchor=NW, window=self.button)
    
    def create_image(self):
        image_size = min(self.main.canvas_width//2, self.main.canvas_height//2-self.main.button_height*2)
        self.resized_image = self.raw_image.resize((int(image_size), int(image_size)))
        self.img = ImageTk.PhotoImage(self.resized_image)
        self.canvas.create_image(self.main.canvas_width/4, self.main.button_height+self.main.canvas_height/4, anchor=CENTER, image=self.img)
        self.canvas.delete(str(self.txt))
        self.txt = self.canvas.create_text(self.canvas_width/2-10, self.main.button_height, text=self.filepath, anchor=E)

    def resizing(self):
        super().resizing()
        if self.img != None:
            self.create_image()

    def openimage(self):
        self.filepath = askopenfilename(title="Ouvrir une image",filetypes=[('png files','.png'),('all files','.*')])
        self.raw_image = Image.open(self.filepath)
        self.create_image()


class mainWindow:
    def __init__(self):
        self.img1 = None
        self.img2 = None
        self.button_height = 20
        self.filepath1 = None
        self.filepath2 = None


        self.window = Tk()
        self.window.title('Projet Image')
        self.window.geometry(str(1200)+'x'+str(900)+'+50+50')



        self.main_canvas = Canvas(self.window, width=200, height=200)
        self.main_canvas.pack()
        self.main_canvas.place(relx=0.5, rely=0.5, anchor=CENTER)

        self.canvas1 = openCanvas(self, "get hidden image", 0.25, 0.25)
        self.canvas2 = openCanvas(self, "get visible image", 0.25, 0.75)
        self.canvas3 = optionCanvas(self, 0.75, 0.25)
        self.canvas4 = resultCanvas(self, "save image", 0.75, 0.75)

        self.window.bind("<Configure>", self.resizing)
        self.window.mainloop()

    def generate_results(self):
        print("bon ben voila")

    def resizing(self, event):
            if event.widget == self.window:
                if getattr(self, "_after_id", None):
                    self.window.after_cancel(self._after_id)

                
                self.canvas_width = self.window.winfo_width() - 20
                self.canvas_height = self.window.winfo_height() - 20
                self.main_canvas.config(width=self.canvas_width, height=self.canvas_height)
                self.canvas1.resizing()
                self.canvas2.resizing()
                self.canvas3.resizing()
                self.canvas4.resizing()



window = mainWindow()
