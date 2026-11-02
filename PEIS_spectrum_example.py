# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 23:51:43 2025

@author: Herman
"""



import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
import matplotlib
#Locale settings
import locale
# Set to German locale to get comma decimal separater
# locale.setlocale(locale.LC_NUMERIC, "cs_CZ")

from matplotlib.ticker import FuncFormatter
def custom_format(x, pos):
    return r"${:,}$".format(int(x)).replace(',', r'\,')


plt.rcParams['text.usetex'] = True
plt.rcParams['axes.formatter.use_locale'] = True

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": "Times New Roman"
})


s028 = {'name':'028', "loading" : "GDE", "position" : "1-1-?H"} # hotovo
s029 = {'name':'029', "loading" : "116 nm Pt", "position" : "1-1-?H"} # hotovo
s041 = {'name':'041', "loading" : "58 nm Pt +  58 nm C", "position" : "1-1-PS"} # hotovo
s042 = {'name':'042', "loading" : "29 nm Pt +  89 nm C", "position" : "1-1-PS"} # hotovo
s044 = {'name':'044', "loading" : "Pt + C cosputtering", "position" : "1-2-LS"}
s109 = {'name':'109', "loading" : "7 nm Pt on 5 nm Fe"}
s047 = {'name':'047', "loading" : "14 nm Pt"} # hotovo
s046 = {'name':'046', "loading" : "14 nm Pt + C"}   # hotovo
s049 = {'name':'049', "loading" : "7 nm Pt +  110 nm C"} # hotovo
s050 = {'name':'050', "loading" : "4 nm Pt +  108 nm C", "position" : "2-1-PS"} # hotovo, možná by se dal dofitovat 1. cyklus
s055 = {'name':'055', "loading" : "7 nm Pt +  58 nm C"} # hotovo
s065 = {'name':'065', "loading" : "7 nm Pt +  29 nm C"} # hotovo
s076 = {'name':'076', "loading" : "7 nm Pt +  14 nm C", "position" : "2-2-PS"}
s079 = {'name':'079', "loading" : "cosputtering Pt 5 W C 300 W 60 min", "position" : "2-2-LS"}
s111 = {'name':'111', "loading" : "7 nm Pt on 5 nm Fe, 30 nm C", "position" : "3-1-PH"} #hotovo
s092 = {'name':'092', "loading" : "4 nm Pt + 29 nm C", "position" : "3-??"}
s094 = {'name':'094', "loading" : "7 nm Pt + 7 nm C", "position" : "3-??"} # hotovo
s096 = {'name':'096', "loading" : "60 nm PtO", "position" : "3-??"}
s097 = {'name':'097', "loading" : "15 nm PtO", "position" : "3-??"}
s098 = {'name':'098', "loading" : "7 nm Pt", "position" : "2-2-PS"} # hotovo
s120 = {'name':'120', "loading" : "7 nm Pt + 15 nm C", "position" : "4-1-?"}
s143 = {'name':'143', "loading" : "7 nm Pt + x nm Fe", "position" : "4-1-?"}


MEA = [s049]
for mea in MEA:
    mea['files'] = glob(os.path.join('C:/Users/Herman/Desktop/MFF/Diplomka/Data z PEMWE/'+mea['name']+'*/','PEIS_time24*'))
    mea['files'].sort()
    

        #0.5 Hz až do 10000 Hz
plt.close()

# colors = ['orange','darkorange','red','darkred']


fig, ax = plt.subplots()

for i,file in enumerate(s049['files']):
    print(file)
    d = np.loadtxt(file,skiprows = 2,usecols = (0,1))
    Zre = 1000*d[:,0]
    Zim = 1000*d[:,1]
    # cmap = plt.get_cmap('rainbow')
    # colors = cmap(np.linspace(0,1,len(Zre))) 
    # for bod in range(0,len(Zre)):
        # plt.plot(Zre[bod],Zim[bod],'.',c = colors[bod])
    im = ax.scatter(x=Zre,y=Zim,marker = 'x',c = np.linspace(1,0,len(Zre)),cmap = 'rainbow')


plt.xlabel(r'$Z_{\mathrm{Re}} ~ \mathrm{\left( m\Omega \right) }$',size=16)
plt.ylabel(r'$-Z_{\mathrm{Im}} ~ \mathrm{\left( m\Omega \right) }$',size=16,labelpad = -5)

# cbar = fig.colorbar(sc, orientation='horizontal', ticks=[], shrink=0.5,aspect=15)
# cbar.ax.set_position([10, 30, 12, 30])
# cbar.ax.set_xlabel('frequency', fontsize=18,labelpad=10)


# plt.yticks([0,1],size=14)
# plt.xticks([0,1,2,3],size=14)
# plt.xticks([0, 500,1000,1500,2000],size=14)
plt.xticks(size = 14)
plt.yticks(size = 14)
plt.axhline(0,ls='--',c="black",lw=0.5)
plt.xlim(18,81)
plt.gca().set_aspect('equal')
plt.text(19,-6,"Ohmic \n resistance",size = 15)
plt.text(36,-10,"Inductive part",size = 15)
plt.text(70,12,"Anodic \n reaction",size = 15)
plt.text(26,6,"Cathodic \n reaction",size = 15)
# plt.ylim(1.38,2.02)
# ax1.set_xticks([0,1000,2000])
# ax2.set_xticks([0,1000,2000])

# ax1.tick_params(axis='both', labelsize=14)
# ax2.tick_params(axis='both', labelsize=14)

# plt.gca()..set_major_formatter(FuncFormatter(custom_format))
# ax1.xaxis.set_major_formatter(FuncFormatter(custom_format))
cbar_ax = fig.add_axes([0.64, 0.27, 0.2, 0.05])
# cbar_ax = (0.125,0.125;0.775x0.775)
# print(type(cbar_ax))
cbar = fig.colorbar(im, orientation="horizontal", ticks=[0,1],cax = cbar_ax) 
# cbar.ax.tick_params(labelsize=15)
cbar.ax.set_title("Frequency", fontsize=15) 
cbar.set_ticks([0, 1])       # Specify tick positions
cbar.ax.set_xticklabels(["10 kHz","500 mHz"], fontsize=12)  # Custom labels

xticks = cbar.ax.get_xticks()
xticklabels = cbar.ax.get_xticklabels()

for label, color in zip(xticklabels, ['indigo', 'red']):
    label.set_color(color)




# cbar.ax.invert_xaxis()
plt.savefig('C:/Users/Herman/Desktop/MFF/Diplomka/Grafy/PEIS_spectrum_examplex.png',dpi = 1000, format = 'png', bbox_inches = 'tight')

plt.show()
# fig, ax = plt.subplots(1, 1, sharex=True)
# j=0        
# print(soubory)
# order = []
# for i,soubor in enumerate(soubory):
#     try:
#         soubor=soubor[0]
#     except:
#         continue
#     d = np.loadtxt(str(soubor),skiprows=1) 
#     I = d[:,0] 
#     U = d[:,1] 

#     print(soubor)
            
#     ax.plot(I/4.62,U,'.',c=colors[j], label=labels[i] ) 
#     j=j+1
#     g = lambda x: np.log(x + 1)
        
#     def forward(y):
#         return 1/y
        
#     def inverse(y):
#         return y
#     print(order)
# ax.set_ylabel('$U$ [${\mathrm{V}}$]',size=17)
# ax.set_xlabel('$j  \mathrm{\\left[ mA ~ cm^{-2} \\right] }$',size=17)


# # MEA GDE, 224 a 224 nelept
# # ax.set_xlim([0, 15000])
# ax.set_xlim([0, 20000/4.62])

# ax.set_ylim([1.3,2.05])
# ax.xaxis.set_tick_params( labelsize=16)
# ax.yaxis.set_tick_params( labelsize=16)
# ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: locale.format_string("%.1f", x)))


# # MEA GDE, 224 a 224 nelept
# # plt.title('Aktivita \emph{MEA GDE}, \emph{MEA 224 nm} a \emph{MEA 224 nm nelept.',size=16)
# # plt.title('Aktivita \emph{MEA 112 nm}, \emph{MEA 224 nm} a \emph{MEA GDE}',size=16)
# # plt.title('Aktivita \emph{MEA 56 nm} a \emph{MEA 56 nm nelept.}',size=16)
# ax2=ax.twinx()
# ax2.set_ylim([148.1/2.05,148.1/1.3])
# ax2.set_yscale('function', functions=(forward, inverse))
# ax2.set_ylabel('$\eta_U$ [\%]',size=17)
# ax2.yaxis.set_tick_params( labelsize=16)
# ax.legend(loc='lower right',handletextpad=0.2,fontsize=17,edgecolor='0',framealpha=1).get_frame().set_facecolor('none')

# fig.set_size_inches(8, 6)

# plt.savefig('GDL56.svg',dpi = 1000, format = 'svg', bbox_inches = 'tight')
# plt.savefig('GDE+224.svg',dpi = 1000, format = 'svg', bbox_inches = 'tight')
# plt.savefig('GDE+480+240.png',dpi = 500, format = 'png', bbox_inches = 'tight')

        
# plt.show()

