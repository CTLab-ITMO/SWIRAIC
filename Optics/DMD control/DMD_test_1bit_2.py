import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from DMD_UPOLabs.dmd_upolabs import DMD_UPOLabs

import numpy as np
from matplotlib import pyplot as plt
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


# main function --------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    
    DMD = DMD_UPOLabs(libDir="C:/Users/QPM_Lab/Downloads/DMDNetWork/DMDNetwork_SDK/DevPkg/bin/")

    IP_Computer = "192.168.1.2"
    IP_dmd = "192.168.1.20"
    Port_Computer = 6002
    Port_dmd = 6003

    
    value_connect = DMD.DMD_Connect(IP_Computer, Port_Computer, IP_dmd, Port_dmd)
    H_treshold = 0.125
        
    prefix_str = input('Prefix: ')
    wl = input('Wavelength: 1. 800 nm, 2. 1050 nm, 3. 1550 nm 4. end 5. show 3rd pattern: ')
    duration = 1000  # milliseconds
    freq_s = 1000  # Hz

    while wl != "4" or wl != "4":

        if wl == "5":
            freq = 5
            picNum = 8224
            
            value_setparam1 = DMD.DMD_SetParam1(IP_dmd, 1, freq, picNum)
            print(f"设置参数一成功？：{value_setparam1}")

            # 设置参数二，触发分频设置
            #IP, Mirror?, Reverse?, ...
            value_setparam2 = DMD.DMD_SetParam2(IP_dmd, False, False, False, 0, 0, 0, 0, 0)
            print(f"设置参数二成功？：{value_setparam2}")
            DMD.DMD_Stop(IP_dmd)
            DMD.DMD_Reset(IP_dmd)
            picNumPlay = 8224
            pauseTime = 1 #picNumPlay/freq

            DMD.DMD_Play(IP_dmd, 1, picNumPlay, 0)

            time.sleep(pauseTime)
            DMD.DMD_Stop(IP_dmd)

            wl = input('Wavelength: 1. 800 nm, 2. 1050 nm, 3. 1550 nm 4. end 5. show 3rd pattern: ')
            
            DMD.DMD_Reset(IP_dmd)
            
        else:
            freq = 200   
            rsl = 64
            picNum = 8224

            picNumPlay = (rsl*rsl)*2+32*(rsl*rsl//(64*64))

            pauseTime = picNum/freq

            value_setparam1 = DMD.DMD_SetParam1(IP_dmd, 1, freq, picNum)
            print(f"设置参数一成功？：{value_setparam1}")
                #IP, Mirror?, Reverse?, ...
            value_setparam2 = DMD.DMD_SetParam2(IP_dmd, False, False, False, 0, 0, 0, 0, 0)
            print(f"设置参数二成功？：{value_setparam2}")        
            DMD.DMD_Stop(IP_dmd)
            DMD.DMD_Reset(IP_dmd)
            with LCOMP(slot=1) as ldev:
                slpar = ldev.GetSlotParam()
                descr = ldev.ReadPlataDescr()

                buffer_size = ldev.RequestBufferStream(size=131072, stream_id=L_STREAM.ADC)        

                adcpar = WDAQ_PAR()

                adcpar.t3.s_Type = L_PARAM.ADC
                adcpar.t3.FIFO = 4096
                adcpar.t3.IrqStep = 4096
                adcpar.t3.Pages = 32
                adcpar.t3.AutoInit = 0                              
                adcpar.t3.dRate = 2.5 #Частота в КГц                            
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
                    end_p = picNumPlay
                    counts = range(1)
                else:
                    counts = range((picNumPlay//picNum))
                    end_p = picNum
                
                for i in counts:
                    ldev.InitStartLDevice()
                    ldev.StartLDevice()
                    time.sleep(0.1)
                    DMD.DMD_Play(IP_dmd, picNum*(i)+1, end_p, 0)
                    print(f"Start")                
                    time.sleep(pauseTime+1)
                        # DMD停止播放
                    DMD.DMD_Stop(IP_dmd)
                    DMD.DMD_Reset(IP_dmd)
                    ldev.StopLDevice()
                    print(f"Done")
                    x = e154.GetDataADC(adcpar.t3, descr, data_ptr, 106000)
                    x2 = x[0]

                    if i == 0:
                        x_all = np.array(x2)
                    else:
                        x_all = np.concatenate([x_all,x2])
                    
                    #plt.plot(x_all)
                    #plt.show()
                    #x2 = x2 - min(x2)
                    #x2 = x2/max(x2)
                    #x2 = x2[x2>0\.01]
                    #plt.plot(x2)
                    #plt.show()
                    #n, n_out = x2.size, x2.size//13
                    #T = pauseTime/n
                    #t = np.arange(n) * T

                    #bp_in= (int(4 * (n*T)), None)
                    #x_env, x_res = envelope(x, bp_in, n_out=n_out)
                    #t_out = np.arange(n_out) * (n / n_out) * T
                    #t_out= np.transpose(np.expand_dims(t_out,1))

                    #locs,_ = find_peaks(x2,height=H_treshold,threshold=None,distance=10,width=None)
                    #plt.plot(t, x2)
                    #plt.plot(t_out, x_res-x_env, '.-', alpha=0.5, label=None)
                    #plt.plot(locs,x2[locs],'X')
                    #plt.show()
                    
                #关闭电脑和DMD的连接
            
            timestr = time.strftime("%Y_%m_%d-%H_%M_%S_")
            
            match wl:
                case '1':
                    wl_str = '800_nm'
                case '2':
                    wl_str = '1050 nm'
                case '3':
                    wl_str = '1550 nm'

            name = prefix_str +'_' + wl_str +'.log'
            #np.savetxt(name,np.around(b_h,decimals=3),fmt="%.3f")
            np.savetxt(name,np.around(x_all,decimals=3),fmt="%.3f")
            winsound.Beep(freq_s, duration)
            #print(f"b_size：{len(b_h)}")
            
            wl = input('Wavelength: 1. 800 nm, 2. 1050 nm, 3. 1550 nm 4. end 5. show 3rd pattern: ')


value_close = DMD.DMD_Close(IP_dmd)
print(f"关闭成功？：{value_close}")

