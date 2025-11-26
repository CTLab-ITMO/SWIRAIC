import numpy as np
import scipy as sp
import re
from PIL import Image
from scipy import signal
from scipy.linalg import hadamard
from matplotlib import pyplot as plt
import matplotlib.image
import winsound

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure

import cv2
import time
from scipy.signal import find_peaks
from scipy.signal import envelope

import logging

from lcomp.device import e140, e154, e440, e2010, l791
from lcomp.ioctl import (L_ASYNC, L_DEVICE, L_EVENT, L_PARAM, L_STREAM, L_USER_BASE,
                         WASYNC_PAR, WDAQ_PAR)
from lcomp.lcomp import LCOMP
import winsound

logging.basicConfig(level=logging.INFO)

# DMD libs
from DMD_UPOLabs.dmd_upolabs import DMD_UPOLabs

#cmd comands
import subprocess

# Python program to create a basic form 
# GUI application using the customtkinter module
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog

import os
import sys

# Sets the appearance of the window
# Supported modes : Light, Dark, System
# "System" sets the appearance mode to 
# the appearance mode of the system
ctk.set_appearance_mode("System")   

# Sets the color of the widgets in the window
# Supported themes : green, dark-blue, blue    
ctk.set_default_color_theme("green")    

# Dimensions of the window
appWidth, appHeight = 1500, 1200

#Global vars
value_connect = False
IP_dmd = '192.168.1.20'
Port_dmd = 6003
IP_Computer = '192.168.1.2'
Port_Computer = 6002
fid = 32768
DMD_freq = 10000
ADC_freq = 120000
B = []
Img_to_save = []
H = np.transpose(np.float64(hadamard(128*128)>0))

# App Class
class App(ctk.CTk):
    # The layout of the window will be written
    # in the init function itself

    def Close_Window(self):
        if value_connect:
            self.CloseDMD()
        self.withdraw()
        self.quit()

    def make_topmost(self):
        """Makes this window the topmost window"""
        self.lift()
        self.attributes("-topmost", 1)
        self.attributes("-topmost", 0)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Sets the title of the window to "App"
        self.title("GUI Application")   
        # Sets the dimensions of the window to appWidth, appHeight
        self.geometry(f"{appWidth}x{appHeight}")
        self.make_topmost()  

        # Host_IP Label
        self.Host_IpLabel = ctk.CTkLabel(self,
                                text="Host IP")
        self.Host_IpLabel.grid(row=0, column=0,
                            padx=20, pady=20,
                            sticky="ew")

        # Host_IP Entry Field
        self.Host_IpEntry = ctk.CTkEntry(self,
                          placeholder_text="192.168.1.2")
        self.Host_IpEntry.insert(ctk.END, IP_Computer) 
        self.Host_IpEntry.grid(row=0, column=1,
                            columnspan=1, padx=20,
                            pady=20, sticky="ew")
        
        # Host Port label
        self.Host_PortLabel = ctk.CTkLabel(self,
                                           text = 'Host Port')
        self.Host_PortLabel.grid(row=0, column=2,
                            padx=20, pady=20,
                            sticky="ew")       
        
        # Host Port label
        self.Host_PortEntry = ctk.CTkEntry(self,
                          placeholder_text="6002")
        self.Host_PortEntry.insert(ctk.END, Port_Computer) 
        self.Host_PortEntry.grid(row=0, column=3,
                            columnspan=1, padx=20,
                            pady=20, sticky="ew") 
        # Desp_IP Label
        self.Dest_IpLabel = ctk.CTkLabel(self,
                                text="Desp IP")
        self.Dest_IpLabel.grid(row=1, column=0,
                            padx=20, pady=20,
                            sticky="ew")

        # Desp_IP Entry Field
        self.Dest_IpEntry = ctk.CTkEntry(self,
                          placeholder_text="192.168.1.20")
        self.Dest_IpEntry.insert(ctk.END, IP_dmd) 
        self.Dest_IpEntry.grid(row=1, column=1,
                            columnspan=1, padx=20,
                            pady=20, sticky="ew")
        
        # Dest Port label
        self.Dest_PortLabel = ctk.CTkLabel(self,
                            text = 'Dest Port')
        self.Dest_PortLabel.grid(row=1, column=2,
                            padx=20, pady=20,
                            sticky="ew") 
           
        # Host Port label
        self.Dest_PortEntry = ctk.CTkEntry(self,
                            placeholder_text="6003")
        self.Dest_PortEntry.insert(ctk.END, Port_dmd) 
        self.Dest_PortEntry.grid(row=1, column=3,
                            columnspan=1, padx=20,
                            pady=20, sticky="ew")
        
        self.Connect_button = ctk.CTkButton(self,
                            text = 'Connect',
                            command=self.Connect_DMD)
        self.Connect_button.grid(row = 2, column = 0,
                                 padx = 20, columnspan=2,
                                 pady = 20, sticky = "ew")
        
        
        self.CloseDMD_button = ctk.CTkButton(self,
                            text = 'Close DMD',
                            command=self.CloseDMD)
        self.CloseDMD_button.grid(row = 3, column = 0,
                                padx = 20, columnspan=2,
                                pady = 20, sticky = "ew")
        
        self.OpenPort_button = ctk.CTkButton(self,
                            text = 'Open port',
                            command=self.OpenPort)
        self.OpenPort_button.grid(row = 4, column = 0,
                                padx = 20, columnspan=2,
                                pady = 20, sticky = "ew")

        # Text Box
        self.displayBox = ctk.CTkTextbox(self,
                                         width=200,
                                         height=100)
        self.displayBox.grid(row=5, column=0,
                             columnspan=2, padx=20,
                             pady=20, sticky="nsew")
        
        
        self.PicFilePathButton = ctk.CTkButton(self,
                                text = 'Pics Filder...',
                                command = self.GetFolder)
        self.PicFilePathButton.grid(row=6, column=0,
                            padx=20, pady=20,
                            sticky="ew") 
        
        self.PicFilePathEntry = ctk.CTkEntry(self)
        self.PicFilePathEntry.grid(row=6, column=1,
                            columnspan=1, padx=20,
                            pady=20, sticky="ew")
        self.PicFilePathEntry.insert(ctk.END, "D:/forexp128had_128_128") 
        
        self.FilesCounter = ctk.CTkEntry(self)
        self.FilesCounter.grid(row=6, column=2,
                            columnspan=1, padx=20,
                            pady=20, sticky="ew")
        self.FilesCounter.insert(ctk.END, str(fid))
        
        self.SendPicsBtn = ctk.CTkButton(self,
                            text = 'Send Pictures',
                            command = self.SendPics)
        self.SendPicsBtn.grid(row=7, column=0,
                            padx=20, pady=20,
                            sticky="ew") 
        
        self.DMD_FreqLabel = ctk.CTkLabel(self, 
                                    text = 'DMD Freq')
        self.DMD_FreqLabel.grid(row = 8, column = 0,
                            padx = 20, pady =20,
                            sticky = "ew")
        
        self.DMD_FreqEntry = ctk.CTkEntry(self)
        self.DMD_FreqEntry.grid(row = 8, column = 1,
                            padx = 20, pady =20,
                            sticky = "ew")
        self.DMD_FreqEntry.insert(ctk.END, DMD_freq)

        self.ADC_FreqLabel = ctk.CTkLabel(self, 
                                    text = 'ADC Freq')
        self.ADC_FreqLabel.grid(row = 8, column = 2,
                            padx = 20, pady =20,
                            sticky = "ew")

        self.ADC_FreqEntry = ctk.CTkEntry(self)
        self.ADC_FreqEntry.grid(row = 8, column = 3,
                            padx = 20, pady =20,
                            sticky = "ew")
        self.ADC_FreqEntry.insert(ctk.END, ADC_freq)
        
        self.ShowGarbageBtn = ctk.CTkButton(self,
                            text = 'Show Garbage',                            
                            command = self.ShowGarbage)
        self.ShowGarbageBtn.grid(row=9, column=0,
                            padx=20, pady=20,
                            sticky="ew") 
        
        self.ShowGarbageBtn.configure(state="normal")
        
        self.StopGarbageBtn = ctk.CTkButton(self,
                            text = 'Stop Showing Garbage',                            
                            command = self.StopShowing)
        self.StopGarbageBtn.grid(row=9, column=1,
                            padx=20, pady=20,
                            sticky="ew") 
        
        self.StopGarbageBtn.configure(state="disabled")

        self.StartMeasureBtn = ctk.CTkButton(self,
                            text = 'Start Measure',                            
                            command = self.StartMeasure)
        self.StartMeasureBtn.grid(row=10, column=0,
                            padx=20, pady=20,
                            sticky="ew") 
        
        self.NumForLsq = ctk.CTkEntry(self)
        self.NumForLsq.grid(row=10, column=1,
                            columnspan=1, padx=20,
                            pady=20, sticky="ew")
        self.NumForLsq.insert(ctk.END, "16384")

        self.radio_var = tk.IntVar(value=0)
        self.radiobutton_1 = ctk.CTkRadioButton(self, text="WL = 941 nm",
                                                    command=self.radiobutton_event, variable= self.radio_var, value=941)
        self.radiobutton_2 = ctk.CTkRadioButton(self, text="1064 nm",
                                                    command=self.radiobutton_event, variable= self.radio_var, value=1065)
        self.radiobutton_3 = ctk.CTkRadioButton(self, text="1550 nm",
                                                    command=self.radiobutton_event, variable= self.radio_var, value=1550)
        self.radiobutton_1.grid(row =11,column =0,
                            padx=20, pady=20,
                            sticky="ew")
        self.radiobutton_2.grid(row =11,column =1,
                            padx=20, pady=20,
                            sticky="ew")
        self.radiobutton_3.grid(row =11,column =2,
                            padx=20, pady=20,
                            sticky="ew")   

        self.radio_var_method = tk.IntVar(value=0)
        self.radiobutton_method_1 = ctk.CTkRadioButton(self, text="Method = Corr_func",
                                                    command=self.radiobutton_event_2, variable= self.radio_var_method, value=1)
        self.radiobutton_method_2 = ctk.CTkRadioButton(self, text="lsqminnorm",
                                                    command=self.radiobutton_event_2, variable= self.radio_var_method, value=2)
        self.radiobutton_method_1.grid(row =12,column =0,
                            padx=20, pady=20,
                            sticky="ew")
        self.radiobutton_method_2.grid(row =12,column =1,
                            padx=20, pady=20,
                            sticky="ew")

        self.spike_th_Entry = ctk.CTkEntry(self)
        self.spike_th_Entry.grid(row=11, column=3,
                            padx=20, pady=20,
                            sticky="ew")
        self.spike_th_Entry.insert(ctk.END, 100)      

        self.StartRecBtn = ctk.CTkButton(self,
                            text = 'Start Reconstruction',                            
                            command = self.StartRec)
        self.StartRecBtn.grid(row=12, column=2,
                            padx=20, pady=20,
                            sticky="ew") 
        self.StartSaveBtn = ctk.CTkButton(self,
                            text = 'Save Image',                            
                            command = self.Save_Img)
        self.StartSaveBtn.grid(row=12, column=3,
                            padx=20, pady=20,
                            sticky="ew") 

        
        self.StartMeasureBtn.configure(state="normal")
        self.protocol("WM_DELETE_WINDOW", self.Close_Window)

        self.fig,self.ax = plt.subplots()

        self.canvas = FigureCanvasTkAgg(self.fig, self)
        self.canvas.get_tk_widget().grid(row=0, column=4,
                                    rowspan=6)
        self.canvas.draw()
        
        self.fig2,self.ax2 = plt.subplots()

        self.canvas2 = FigureCanvasTkAgg(self.fig2, self)
        self.canvas2.get_tk_widget().grid(row=7, column=4,
                                    rowspan=6)
        self.canvas2.draw()
        #self.Connect_DMD()

    def Save_Img(self):
        #t_n =time.ctime()
        #t_n = t_n.replace(' ','_')
        #t_n = t_n.replace(':','_')
        path_to_save = 'C:/Users/QPM_Lab/Downloads/DMDNetWork/DMD_Matlab/example/For_dataset/'
        name = 'wl_'+str(self.radio_var.get())+'_'
        #name = '.log'
        #np.savetxt('.log',np.around(x_all,decimals=5),fmt="%.5f")
        np.savetxt('B_measured_'+name+'log',np.around(B,decimals=5),fmt="%.5f")

        next_number = max([int(f.split('_')[0]) for f in os.listdir(path_to_save) if f.endswith(f'_wl_{self.radio_var.get()}_.png') and f.split('_')[0].isdigit()] + [0]) + 1 
        
        matplotlib.image.imsave(path_to_save+str(next_number)+'_'+name+'.png', self.Img_to_save, cmap = 'gray')

    def StartRec(self):

        B = self.B        
        size_h = 128
        N_new = int(self.NumForLsq.get())

        if self.radio_var_method.get() == 1:
            h = H
            sBI = np.zeros(len(h[:][0]))
            sI = sBI
            sB = 0
            B = np.array(B)
            sBI = h[0:N_new][:].T @ B[0:N_new]
            sI = np.sum(h,axis=1)
            sB = np.sum(B)            
            G = np.divide(sBI,N_new) - np.divide(sI,N_new)*(sB/N_new) 
            G[0] = 0
            G[G<0] = 0
            while np.max(G) > np.median(G)*int(self.spike_th_Entry.get()):
                G[G==np.max(G)] = np.median(G)                              
            
            
            G = np.reshape(G,[size_h,size_h])
        if self.radio_var_method.get() == 2:
            N_f_lsq = int(self.NumForLsq.get())
            #h = np.loadtxt("D:/Patterns_DMD/ALL_pink_patterns128.txt", delimiter=',', dtype=int)
            h =np.transpose(np.int8(hadamard(size_h*size_h)>0))
            x = np.linalg.lstsq(h[:][0:N_f_lsq], B[0:N_f_lsq], rcond=None)                    
            G = np.reshape(x[0],[size_h,size_h])
        G = G[:][:]
        G = G-np.min(G)
        G = np.round(256*G/np.max(G))
        self.Img_to_save = G

        #np.savetxt('lo_d_2_'+name,np.around(lo_d_2,decimals=5),fmt="%.5f")
        self.ax2.clear()
        self.ax2.imshow(G, cmap = 'gray')
        #self.ax.axis('off')
        self.canvas2.draw()    
        
        n_frequency = 2500  # Set Frequency To 2500 Hertz
        n_duration = 1000  # Set Duration To 1000 ms == 1 second
        winsound.Beep(n_frequency, n_duration)

        

    def radiobutton_event(self):
        self.displayBox.delete("0.0", "200.0")
        self.displayBox.insert("0.0", "current WL: "+str(self.radio_var.get()))
    
    def radiobutton_event_2(self):
        self.displayBox.delete("0.0", "200.0")
        self.displayBox.insert("0.0", "Method: "+str(self.radio_var_method.get()))

    def SendPics(self):

        IP_dmd = self.Dest_IpEntry.get()
        DMD_freq = int(self.DMD_FreqEntry.get())
        fid = int(self.FilesCounter.get())
        picNum = fid + 32
        Filepath = self.PicFilePathEntry.get()

        DMD.DMD_SetParam1(IP_dmd, 1, DMD_freq, picNum)
        DMD.DMD_SetParam2(IP_dmd, False, False, False, 0, 0, 0, 0, 0)

        DMD.DMD_SendPath_Init(IP_dmd, 1)


        B_W_folder = "C:/Users/QPM_Lab/Downloads/DMDNetWork/test2/B&W/"
        value_sendpath_1bit_cache = {}
        for i in range(16):
            value_sendpath_1bit_cache = DMD.DMD_SendPath_1bit_cache(IP_dmd, B_W_folder + f"0(1).bmp")
            value_sendpath_1bit_cache = DMD.DMD_SendPath_1bit_cache(IP_dmd, B_W_folder + f"0(2).bmp")  
        
        self.displayBox.insert(ctk.END, "\n" f"{(i+1)*2} garbage 1bit images sent：{value_sendpath_1bit_cache}")


                        
        for i in range(fid//2):
            value_sendpath_1bit_cache = DMD.DMD_SendPath_1bit_cache(IP_dmd, Filepath + f"/{ i + 1 }.bmp")
            value_sendpath_1bit_cache = DMD.DMD_SendPath_1bit_cache(IP_dmd, Filepath + f"/{ i + 1 }_.bmp")
            progress = (i / (fid//2))
            arrow = '=' * int(round(progress * 50) - 1)
            spaces = ' ' * (50 - len(arrow))
            sys.stdout.write(f'\rProgress: [{arrow + spaces}] {int(progress * 100)}%')
            sys.stdout.flush()
        
        self.displayBox.insert(ctk.END, "\n" f"{(i+1)*2} main 1bit images sent：{value_sendpath_1bit_cache}")
        sys.stdout.write(f'\rProgress: [{arrow + spaces}] {int(100)}% Done')

        


    def GetFolder(self):
        Filepath = filedialog.askdirectory()
        self.PicFilePathEntry.delete(0, ctk.END)
        self.PicFilePathEntry.insert(ctk.END,Filepath) 
        fid = 0
        for path in os.listdir(Filepath):
            if os.path.isfile(os.path.join(Filepath, path)):
                fid += 1
        self.FilesCounter.delete(0, ctk.END)
        self.FilesCounter.insert(ctk.END,str(fid)) 


    def Connect_DMD(self):
        IP_Computer = self.Host_IpEntry.get()
        IP_dmd = self.Dest_IpEntry.get()
        Port_Computer = int(self.Host_PortEntry.get())
        Port_dmd = int(self.Dest_PortEntry.get())
        value_connect = str(DMD.DMD_Connect(IP_Computer, Port_Computer, IP_dmd, Port_dmd))
        DMD.DMD_Stop(IP_dmd)
        DMD.DMD_Reset(IP_dmd)
        self.displayBox.delete("0.0", "200.0")
        self.displayBox.insert("0.0", "DMD connect: "+value_connect)
        return value_connect
    
    def CloseDMD(self):
        DMD.DMD_Stop(IP_dmd)
        DMD.DMD_Reset(IP_dmd)
        value_close = str(DMD.DMD_Close(IP_dmd))
        self.displayBox.delete("0.0", "200.0")
        self.displayBox.insert("0.0", "DMD close: "+ value_close)
        return value_close
    
    def OpenPort(self):
        command = "netstat -aon|findstr \"6002\""
        ret = subprocess.run(command, stdout=subprocess.PIPE, shell=True)
        a = ret.stdout.decode('utf-8')[-6:-2]
        if len(a) > 0:
            command = "tasklist|findstr \""+a+"\""
            ret2 = subprocess.run(command, stdout=subprocess.PIPE, shell=True)
            a2 = ret2.stdout[0:20].decode('utf-8')
            a3 = a2[0:a2.find(".exe")+4]
            command = "taskkill /f /t /im " + a3
            ret3 = subprocess.run(command, stdout=subprocess.PIPE, shell=True)
            a4 = ret3.stdout.decode('utf-8')
            self.displayBox.delete("0.0", "200.0")
            self.displayBox.insert("0.0", "Port opened")
        else:
            self.displayBox.delete("0.0", "200.0")
            self.displayBox.insert("0.0", "Port has been opened earlier")
    
    def ShowGarbage(self):

        DMD_freq = int(self.DMD_FreqEntry.get())
        picNum = fid + 32
        
        DMD.DMD_Stop(IP_dmd)
        DMD.DMD_Reset(IP_dmd)

        DMD.DMD_SetParam1(IP_dmd, 1, DMD_freq, picNum)
        DMD.DMD_SetParam2(IP_dmd, False, False, False, 0, 0, 0, 0, 0)

        DMD.DMD_Play(IP_dmd, 1, 16, 1)
        self.ShowGarbageBtn.configure(state="disabled")
        self.StopGarbageBtn.configure(state="normal")
        self.StartMeasureBtn.configure(state="disabled")


    def StopShowing(self):

        DMD.DMD_Stop(IP_dmd)
        DMD.DMD_Reset(IP_dmd)
        self.StopGarbageBtn.configure(state="disabled")
        self.ShowGarbageBtn.configure(state="normal")
        self.StartMeasureBtn.configure(state="normal")

    def StartMeasure(self):
        DMD_freq = int(self.DMD_FreqEntry.get())
        ADC_freq = int(self.ADC_FreqEntry.get())
        #self.ShowGarbageBtn.configure(state="disabled")
        #self.StartMeasureBtn.configure(state="disabled")
        
        #self.ax.clear()


        fid = int(self.FilesCounter.get())
        picNumPlay = fid + 32
        picNum = 8192
        
        pauseTime = picNum/DMD_freq

        DMD.DMD_SetParam1(IP_dmd, 1, DMD_freq, picNumPlay)
        DMD.DMD_SetParam2(IP_dmd, False, False, False, 0, 0, 0, 0, 0)
        
        DMD.DMD_Stop(IP_dmd)
        DMD.DMD_Reset(IP_dmd)
        with LCOMP(slot=0) as ldev:
                slpar = ldev.GetSlotParam()
                descr = ldev.ReadPlataDescr()

                buffer_size = ldev.RequestBufferStream(size=131072, stream_id=L_STREAM.ADC)        

                adcpar = WDAQ_PAR()

                adcpar.t3.s_Type = L_PARAM.ADC
                adcpar.t3.FIFO = 4096
                adcpar.t3.IrqStep = 4096
                adcpar.t3.Pages = 32
                adcpar.t3.AutoInit = 0                              
                adcpar.t3.dRate = ADC_freq//1000 #Частота в КГц                            
                adcpar.t3.dKadr = 0.01                              
                adcpar.t3.SynchroType = e154.NO_SYNC               
                adcpar.t3.SynchroSensitivity = e154.A_SYNC_LEVEL    
                adcpar.t3.SynchroMode = e154.A_SYNC_UP_EDGE         
                adcpar.t3.AdChannel = 0
                adcpar.t3.AdPorog = 0
                adcpar.t3.NCh = 1

                adcpar.t3.Chn[1] = e154.CH_1 | e154.V1600         
                # adcpar.t3.Chn[2] = e140.CH_2 | e140.V0625         # e440.CH_2 | e440.V0625    e154.CH_2 | e154.V0500
                # adcpar.t3.Chn[3] = e140.CH_3 | e140.V0156         # e440.CH_3 | e440.V0156    e154.CH_3 | e154.V0160
                adcpar.t3.IrqEna = 1
                adcpar.t3.AdcEna = 1

                ldev.FillDAQparameters(adcpar.t3)
                data_ptr, syncd = ldev.SetParametersStream(adcpar.t3, buffer_size)

                b_h=[]
                if picNumPlay <= picNum:
                    counts = range(1)
                else:
                    counts = range(int(np.ceil(picNumPlay/picNum)))
                x_all = np.zeros((int(np.ceil(picNumPlay/picNum)),buffer_size))
                for i in counts:
                    if abs((picNumPlay-picNum*(i)))>=picNum:
                        end_p = picNum
                    else:
                        end_p = abs((picNumPlay-picNum*(i)))
                    ldev.InitStartLDevice()
                    ldev.StartLDevice()
                    time.sleep(0.1)
                    DMD.DMD_Play(IP_dmd, picNum*(i)+1, end_p, 0)
                    print(f"Start")
                    time.sleep(pauseTime+1)

                    DMD.DMD_Stop(IP_dmd)
                    DMD.DMD_Reset(IP_dmd)
                    ldev.StopLDevice()
                    print(f"\n Done")
                    time.sleep(0.1)
                    x = e154.GetDataADC(adcpar.t3, descr, data_ptr, buffer_size)
                    x2 = x[0]
                    x2 = x2 - np.median(x2[0:400])
                    x_all[i][:] = np.array(x2)
                
                n_frequency = 1000  # Set Frequency To 2500 Hertz
                n_duration = 100  # Set Duration To 1000 ms == 1 second
                winsound.Beep(n_frequency, n_duration)
                #x_all = x_all - np.mean(x_all[0:1500])
                #x_all = x_all/np.max(x_all)
                dist = int(ADC_freq // DMD_freq) 
                #BG = sum(x_all[0:1500])/1500
                #t_h = (max(x_all)-BG)/10   
                all_pos=[]
                all_val=[]
                for i in counts:
                    pos_all,val = find_peaks(x_all[i][:],height=max(x_all[i][:]/3),distance=dist+1,width=2)
                    pos_start = pos_all[0]
                    pos_end = pos_all[-1]
                    self.ax.clear()
                    self.ax.plot(x_all[i][pos_start-dist:pos_end+dist])
                    self.canvas.draw()

                    x_f = x_all[i][pos_start-3*dist:pos_end+3*dist]
                    x_f = x_f-min(x_f)
                    x_f = x_f/max(x_f)
                    t_h=0.3
                    pos,val = find_peaks(x_f,height=t_h,distance=dist,width=(-1)+dist//2)
                    pos_skip = np.where(np.diff(pos)>(dist+1)*3)
                    pos_skip = pos_skip[0][:]
                    pos_2 = pos
                    if len(pos_skip)>0:
                        for ji in range(len(pos_skip)):
                            first = pos[pos_skip[ji]]
                            last = pos[pos_skip[ji]]+dist*2*np.round((pos[pos_skip[ji]+1]-pos[pos_skip[ji]])/(dist*2))
                            step = dist*2
                            imp = np.arange(first, last, step)
                            pos_n = np.where(pos_2==pos[pos_skip[ji]])
                            pos_n = pos_n[0][0] + 1
                            pos_2 = np.insert(pos_2,pos_n,imp[1:len(imp)])
                    pos_2 = np.int32(pos_2)
                    val = x_all[i][pos_2+pos_start-3*dist]
                    all_pos = np.append(all_pos,pos_2)
                    all_val = np.append(all_val,val)

                #np.savetxt('peaks_pos.txt',pos)
                #pos = np.int32(np.loadtxt('peaks_pos.txt'))
                #val = val['peak_heights']
                #val = x_all[pos+pos_start[0]]
                self.B = all_val[16:]
        self.ax.plot(all_pos,all_val,'*') 
        self.ax.text(0.01,max(all_val/2),'Peaks_number = '+str(len(all_val)),ha='left', va='top')       
        self.canvas.draw()
        
        self.ShowGarbageBtn.configure(state="normal")
        self.StartMeasureBtn.configure(state="normal")

        if len(self.B) == 128*128:
            self.StartRec()

    
    
    



if __name__ == "__main__":
    DMD = DMD_UPOLabs(libDir="C:/Users/QPM_Lab/Downloads/DMDNetWork/DMDNetwork_SDK/DevPkg/bin/")
    app = App()

    # Used to run the application
    app.mainloop()
    
