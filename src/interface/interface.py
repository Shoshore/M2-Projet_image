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


class mainWindow:

    def __init__(self):
        self.img = None
        self.text_offset = 10
        self.filepath = None


        self.window = Tk()
        self.window.title('Projet Image')
        self.window.geometry(str(800)+'x'+str(500)+'+50+50')



        self.main_canvas = Canvas(self.window, width=200, height=200, background="yellow")
        self.main_canvas.pack()
        self.main_canvas.place(relx=0.5, rely=0.5, anchor=CENTER)

        self.bouton1=Button(self.main_canvas, text="get hidden image", command=self.openimage)

        self.canvas1 = Canvas(self.main_canvas, width=50, height=50, background="white")
        self.txt1 = self.canvas1.create_text(0, 60, text="")
        self.canvas1.pack()
        self.canvas1.place(relx=0.25, rely=0.25, anchor=CENTER)

        self.bouton2=Button(self.main_canvas, text="get visible image", command=self.openimage)

        self.canvas2 = Canvas(self.main_canvas, width=50, height=50, background="white")
        self.txt2 = self.canvas2.create_text(0, 60, text="")
        self.canvas2.pack()
        self.canvas2.place(relx=0.25, rely=0.75, anchor=CENTER)

        self.window.bind("<Configure>", self.resizing)
        self.window.mainloop()

    def create_image(self):
        image_size = min(self.canvas_width, self.canvas_height-self.text_offset*2)
        self.resized_image = self.raw_image.resize((int(image_size), int(image_size)))
        self.img = ImageTk.PhotoImage(self.resized_image)
        self.canvas1.create_image(self.canvas_width/2, self.text_offset+self.canvas_height/2, anchor=CENTER, image=self.img)
        self.canvas1.delete(str(self.txt1))
        self.txt1 = self.canvas1.create_text(self.canvas_width/2, self.text_offset, text=self.filepath)

    def resizing(self, event):
            if event.widget == self.window:
                if getattr(self, "_after_id", None):
                    self.window.after_cancel(self._after_id)

                
                self.canvas_width = self.window.winfo_width() - 50
                self.canvas_height = self.window.winfo_height() - 50
                self.main_canvas.config(width=self.canvas_width, height=self.canvas_height)

                self.canvas1.config(width=self.canvas_width//2, height=self.canvas_height//2-50)
                self.canvas2.config(width=self.canvas_width//2, height=self.canvas_height//2-50)
                #if self.img != None:
                #    self.create_image()

    
    def openimage(self):
        self.filepath = askopenfilename(title="Ouvrir une image",filetypes=[('png files','.png'),('all files','.*')])
        self.raw_image = Image.open(self.filepath)
        self.create_image()


window = mainWindow()
