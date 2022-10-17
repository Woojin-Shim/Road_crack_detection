import tkinter.ttk as ttk
from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import cv2
import os
import detector
from threading import * 
from time import gmtime, strftime

class RepeatTimer(Timer):
    def run(self):
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)


class MyFrame(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)
        self.thread = True
        self.master = master
        self.master.title("Elite")
        self.pack(fill=BOTH, expand=True)
        self.images = {}
        self.play = False
        self.op1 = IntVar()
        self.op1.set(2)
        self.t = RepeatTimer(1, self.live_detect)
        self.t.daemon = True
        self.t.start()
        self.br = True
        self.sensitivity = {'Low':0.3,'Mid':0.5,'High':0.7}
        self.rec = ['',0,0,0]
        self.list = {'In':None,'Ex':None}
        style=ttk.Style()
        style.theme_use('clam')
        self.img_list = []
        self.img_idx = 0

        self.init_file = os.path.dirname(os.path.abspath(__file__))
        file_frame = Frame(self,background='#D3D3D3')
        file_frame.pack(fill="both") # 간격 띄우기
        self.lbl_width = Label(file_frame, text="Elite Crack Detector", font=("Arial",30),background='#D3D3D3')
        self.lbl_width.pack(side="left", padx=20,pady=10)
        btn_add_file = Button(file_frame, padx=0, pady=0, width=2,height=0, text="X", command=self.quit ,background='#D3D3D3',  activebackground='#D3D3D3')
        btn_add_file.pack(side='right', ipadx=0,ipady=0,padx=10, pady=10)


        self.whole_frame = Frame(self,background='#D3D3D3')
        self.whole_frame.pack(fill='both')

        image_frame = LabelFrame(self.whole_frame, text = 'Image',background='#D3D3D3')
        image_frame["borderwidth"] = 1
        image_frame["relief"] = "solid"
        image_frame.pack(side ='left', fill="both", padx=5, pady=10,ipady=10)

        img = (Image.open("default_img.png")) 
        resized_image= img.resize((630,360))
        self.i_img= ImageTk.PhotoImage(resized_image)
        self.canvas1 = Label(image_frame, width= 630, height=360, image=self.i_img,background='#D3D3D3')
        self.canvas1.pack(fill='both',anchor=NW, padx=5, pady=5)
        self.canvas2 = Label(image_frame, width= 630, height=360, image=self.i_img,background='#D3D3D3')
        self.canvas2.pack(fill='both',anchor=NW, padx=5, pady=5)

        
        list_frame = Frame(self.whole_frame,background='#D3D3D3')
        list_frame.pack(fill="x", padx=5, pady=5, ipady=2)
        self.list_file = Listbox(list_frame, selectmode="single",height=42, background='#D3D3D3', activestyle='none', exportselection=False)#, yscrollcommand=self.scrollbar.set,background='#D3D3D3', borderwidth=0,highlightthickness=0)
        self.list_file.pack(side="left", fill="both",pady=10,expand=True)
        self.list_file.bind('<Double-1>', self.load_img)


        frame_option = LabelFrame(self.whole_frame, text="Option", background='#D3D3D3')
        frame_option.pack(fill='x',padx=5, pady=5)
        frame_option["borderwidth"] = 1
        frame_option["relief"] = "solid"

        radio_frame = Frame(frame_option,background='#D3D3D3')
        radio_frame.pack(side='left', padx=5)

        radio1 = Radiobutton(radio_frame,text='In', value=1, variable=self.op1, command=self.radcall1,background='#D3D3D3', activebackground='#D3D3D3')
        radio1.pack(anchor='w')
        radio2 = Radiobutton(radio_frame,text='Ex', value=2, variable=self.op1, command=self.radcall1,background='#D3D3D3', activebackground='#D3D3D3')
        radio2.pack(anchor='w', side='left')


        self.set1 = Label(frame_option, text="Detection Sensitivity", width=20,background='#D3D3D3')
        self.set1.pack(side="left", padx=0, pady=5)
        opt1 = ["Low", "Mid", "High"]
        self.cmb1 = ttk.Combobox(frame_option, state="readonly", values=opt1, width=10,background='#D3D3D3')
        self.cmb1.current(1)
        self.cmb1.pack(side="left", padx=5, pady=5)

        self.set2 = Label(frame_option, text="Category", width=8,background='#D3D3D3')
        self.set2.pack(side="left", padx=0, pady=5)
        opt2 = ["Highway", "Mainroads", "Frontback","Kidzone","Industrialroads"]
        self.cmb2 = ttk.Combobox(frame_option, state="readonly", values=opt2, width=10,background='#D3D3D3')
        self.cmb2.current(0)
        self.cmb2.pack(side="left", padx=5, pady=5)

        self.set3 = Label(frame_option, text="Device idx", width=8,background='#D3D3D3')
        self.set3.pack(side="left", padx=0, pady=5)
        self.cmb3 = ttk.Combobox(frame_option, state="readonly", values=self.get_device(), width=10,background='#D3D3D3')
        self.cmb3.current(0)
        self.cmb3.pack(side="left", padx=5, pady=5)

        self.btn_run = Button(frame_option, padx=15, pady=3, width=5, text="Run", command=lambda:self.run_video(True),background='#D3D3D3', activebackground='#D3D3D3')
        self.btn_run.pack(side="right",padx=5, pady=5)
        self.btn_stop = Button(frame_option, padx=15, pady=3, width=5, text="Stop", command=lambda:self.run_video(False),background='#D3D3D3', activebackground='#D3D3D3')
        self.btn_stop.pack(side="right",padx=5, pady=5)
        self.btn_clear_list = Button(frame_option, padx=15, pady=3, width=5, text="Clear", command=self.clr_list ,background='#D3D3D3',  activebackground='#D3D3D3')
        self.btn_clear_list.pack(side="right",padx=5, pady=5)


        self.path_frame = LabelFrame(self.whole_frame, text="Target Path", background='#D3D3D3')
        # self.path_frame.pack(fill='x', padx=5, pady=10)
        self.path_frame["borderwidth"] = 1
        self.path_frame["relief"] = "solid"
        self.txt_dest_file = Label(self.path_frame,text = 'Please select dir...',width=100,anchor='w',background='#D3D3D3')
        self.txt_dest_file.pack(side='left', padx=10, pady=5, ipady=4, fill='x')
        self.btn_dest_file = Button(self.path_frame,text="Select", width=6, command=self.select_file,background='#D3D3D3', activebackground='#D3D3D3')
        self.btn_dest_file.pack(side="right", padx=5, pady=10, fill='both')

    def clr_list(self):
        self.list_file.delete(0,'end')
        self.canvas2.config(image=self.i_img)
        self.canvas1.config(image=self.i_img)
        pass

    def get_device(self):
        index = 0
        arr = []
        while True:
            cap = cv2.VideoCapture(index)
            if not cap.read()[0]:
                break
            else:
                arr.append(index)
            cap.release()
            index += 1
        return arr

    def select_file(self):
        dir = filedialog.askdirectory(title="폴더를 선택하세요")
        self.list_file.delete(0,'end')
        if dir is not None:
            self.img_list = []
            self.txt_dest_file.config(text=dir)
            for root, dirs, filenames in os.walk(dir):
                for filename in filenames:
                    if '.jpg' in filename or '.png' in filename:
                        self.img_list.append(os.path.join(root,filename))
        if len(self.img_list) > 0:
            self.btn_run.config(state='normal')
            self.btn_stop.config(state='normal')
            self.t2 = RepeatTimer(0.001, self.detect_s_file)
            self.t2.daemon = True
            self.t2.start()


    def change_img(self,where,ext=True):
        filename = where.replace('\\','/').split('/')[-1]
        src = cv2.imread(where)
        src = cv2.resize(src,(630,360))
        img = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        self.img = ImageTk.PhotoImage(image=img)
        self.canvas1.config(image=self.img)
        src, p,c= detector.r_img(src,float(self.sensitivity[self.cmb1.get()]),filename,set=(0.45,3,2))
        img2 = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        img2 = Image.fromarray(img2)
        self.img2 = ImageTk.PhotoImage(image=img2)
        self.canvas2.config(image=self.img2)
        if ext:
            if self.contain(where):
                self.list_file.insert('end' ,where.replace('\\','/') + '        Pothole : ' + str(p) + '     Crack : ' + str(c))
                self.list_file.see('end')
        

    def contain(self,item):
        for i in self.list_file.get(0,'end'):
            if item in i:
                return False
        return True
       

    def load_img(self,event):
        if self.play: 
            pass
        elif self.op1.get() == 2:
            cs = self.list_file.curselection()
            txt=self.list_file.get(cs)
            src = cv2.imread('saved/' + txt.split('  ')[1] + '/' + txt.split('  ')[0].replace('/','_').replace(':','_') + '.jpg')
            src = cv2.resize(src,(630,360))
            img = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            self.img = ImageTk.PhotoImage(image=img)
            self.canvas1.config(image=self.img)

            src2,_,_ = detector.r_img(src,float(self.sensitivity[self.cmb1.get()]),txt.split('  ')[0].replace('/','_').replace(':','_') + '.jpg')
            img2 = cv2.cvtColor(src2, cv2.COLOR_BGR2RGB)
            img2 = Image.fromarray(img2)
            self.img2 = ImageTk.PhotoImage(image=img2)
            self.canvas2.config(image=self.img2)
        elif self.op1.get() == 1:
            cs = self.list_file.curselection()
            txt=self.list_file.get(cs)
            src = cv2.imread(txt.split('        ')[0])
            src = cv2.resize(src,(630,360))
            img = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            self.img = ImageTk.PhotoImage(image=img)
            self.canvas1.config(image=self.img)
            src2,_,_ = detector.r_img(src,float(self.sensitivity[self.cmb1.get()]),txt.split('  ')[0].split('/')[-1],set=(0.45,3,2))
            img2 = cv2.cvtColor(src2, cv2.COLOR_BGR2RGB)
            img2 = Image.fromarray(img2)
            self.img2 = ImageTk.PhotoImage(image=img2)
            self.canvas2.config(image=self.img2)
        pass

    def detect_s_file(self):
        self.btn_dest_file.config(state='disabled')
        for i in self.img_list:
            if self.br:
                self.change_img(i)
            else:
                self.list_file.delete(0,'end')
                self.canvas1.config(image=self.i_img)
                self.canvas2.config(image=self.i_img)
        self.t2.cancel()
        self.btn_dest_file.config(state='normal')

    def live_detect(self):
        if self.play:
            cap = cv2.VideoCapture(int(self.cmb3.get()))
            ret, src = cap.read()
            src = cv2.resize(src,(630,360))
            src2 = src.copy()
            now = strftime("%Y/%m/%d/%H:%M:%S", gmtime())
            src2 ,p,c= detector.r_img(src2,float(self.sensitivity[self.cmb1.get()]),str=now)
            if p + c > 0:
                self.list_file.insert(0,now+'  '+self.cmb2.get()+ '  Pothole : ' + str(p) + '  Crack : ' + str(c))
                cv2.imwrite("saved/" + self.cmb2.get() + '/' +now.replace('/','_').replace(':','_') + '.jpg',src)
            img1 = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(src2, cv2.COLOR_BGR2RGB)
            img1 = Image.fromarray(img1)
            img2 = Image.fromarray(img2)
            self.img1 = ImageTk.PhotoImage(image=img1)
            self.img2 = ImageTk.PhotoImage(image=img2)
            self.canvas1.config(image=self.img1)
            self.canvas2.config(image=self.img2)
            cap.release()
            cv2.destroyAllWindows()
        else:
            pass


    def run_video(self,play):
        if self.op1.get() == 2:
            self.play = play
            if play:
                self.lbl_width.config(foreground='red')
            else:
                self.lbl_width.config(foreground='black')
        else:
            cs = self.g_img_idx()
            self.list_file.selection_clear(cs)
            if play:
                self.img_idx = (cs + 1) % self.list_file.size()
            else:
                self.img_idx = (cs + self.list_file.size() - 1) % self.list_file.size()
            self.list_file.selection_set(self.img_idx % self.list_file.size())
            self.list_file.see(self.img_idx % self.list_file.size())
            self.change_img(self.img_list[self.img_idx],ext=False)





    def g_img_idx(self):
        idx = self.list_file.curselection()
        if len(idx) == 0:
            return 0
        else:
            return idx[-1]

    def radcall1(self):
        v = self.op1.get()
        if v == 1: #Inner
            self.cmb3.config(state='disabled')
            self.path_frame.pack(fill='x', padx=5, pady=10)
            self.list_file.config(height=37)
            self.play = False
            self.list['Ex'] = self.list_file.get(0,last=self.list_file.size()-1)
            self.list_file.delete(0,last=self.list_file.size()-1)
            self.btn_stop.config(text='<')
            self.btn_run.config(text='>')
            if self.list['In'] is not None:
                for v in self.list['In']:
                    self.list_file.insert('end',v)
            if len(self.img_list) == 0:
                self.btn_run.config(state='disabled')
                self.btn_stop.config(state='disabled')
            self.set2.pack_forget()
            self.cmb2.pack_forget()
            self.set3.pack_forget()
            self.cmb3.pack_forget()
            self.lbl_width.config(foreground='black')
        elif v == 2: #External
            self.br = False
            self.cmb3.config(state='readonly')
            self.path_frame.pack_forget()
            self.list_file.config(height=42)
            self.list['In'] = self.list_file.get(0,last=self.list_file.size()-1)
            self.list_file.delete(0,last=self.list_file.size()-1)
            self.btn_stop.config(text='Stop')
            self.btn_run.config(text='Run')
            if self.list['Ex'] is not None:
                for v in self.list['Ex']:
                    self.list_file.insert('end',v)
            self.btn_run.config(state='normal')
            self.btn_stop.config(state='normal')
            self.set2.pack(side="left", padx=5, pady=5)
            self.cmb2.pack(side="left", padx=5, pady=5)
            self.set3.pack(side="left", padx=5, pady=5)
            self.cmb3.pack(side="left", padx=5, pady=5)
            self.t2.cancel()

        self.canvas2.config(image=self.i_img)
        self.canvas1.config(image=self.i_img)


    def convert_to_tkimage(src):
        img = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        return imgtk

def main():
    root = Tk()
    app = MyFrame(root)
    root.resizable(False, False)
    root.attributes('-topmost',True)
    root.attributes('-fullscreen',True)
    root.iconbitmap('elite.ico')
    root.mainloop()
if __name__ == '__main__':
    main()