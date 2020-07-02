########################
#IMPORTS AND ARGUMENTS
########################
import sys
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
import time
import datetime
import plot_invariant_functions as pif

ap = argparse.ArgumentParser()
ap.add_argument("-fs", "--file_signal", required=True, help="Path to the file to plot")
ap.add_argument("-fe", "--file_events", required=True, help="Events file name")
ap.add_argument("-sa", "--save", required=False, default='0', help="Save to file and the name")
ap.add_argument("-st", "--start", required=False, default=0, help="Start second")
ap.add_argument("-en", "--end", required=False, default='None', help="End second")
ap.add_argument("-fr", "--frecuency", required=False, default=10000, help="Sampling freq of -fs")
ap.add_argument("-mo", "--mode", required=False, default=2, help="Mode '1' points appear with the signal. Mode '2' points always on display")
ap.add_argument("-cu", "--current", required=False, default=0, help="'1' Display current injected. '0' ignores current")
ap.add_argument("-fps", "--fps", required=False, default=24, help="Frames per second")
ap.add_argument("-dpi", "--dpi", required=False, default=100, help="Dots per inch")
args = vars(ap.parse_args())
save = args['save']

################
#GLOBAL
################

#### Configuración del programa
segundos_ventana = 10   # Tiempo que se verá en el plot, con 2 funciona otros nu se xD

#### No se recomienda tocar, valores internos
modo  = int(args['mode'])                 # Modo de visualización, va por parámetro, el 2 da mejor rendimiento
freq  = int(args['frecuency'])            # Frecuencia de muestreo, va por parámetro
fps   = int(args['fps'])                  # Frames por segundo
dpi   = int(args['dpi'])                  # DPI. Puntos por pulgada

curr = bool(args['current'])

size_plot = 5                             # Variable que controla el tamaño del plot
ini_ventana=0                             # Inicialización origen de la ventana
pts_ventana = int(segundos_ventana*freq)  # Ventana de datos a tener en cuenta
interval = 1000/fps                       # Cada cuantos milisegundos se refresca la pantalla
index_event = 0                           # Variable de uso interno para controlar el evento mostrado
pts_avance = int(freq/fps)                # Puntos que se deben avanzar en cada refresco

################
#EVENT HANDLE
################
def onKey(event):
    global pts_avance
    v_max=1200
    if abs(pts_avance) >= v_max:
        print("Max Speed") 
    if event.key == ' ':
        if pts_avance!=0:
            pts_avance = 0
        else:
            pts_avance = 150
    elif event.key == 'escape':
        exit(0)
    elif event.key == 'up' and pts_avance<v_max:
        pts_avance = pts_avance+150 
    elif event.key == 'down' and pts_avance>-v_max:
        pts_avance = pts_avance-150 
    elif event.key == 'right' and pts_avance<v_max:
        pts_avance = pts_avance+50   
    elif event.key == 'left' and pts_avance>-v_max:
        pts_avance = pts_avance-50 
       

################
#ARCHIVOS
################
#signal = pif.read_data_manuroy_v1(args["file_signal"], args["start"], args["end"], freq)
# signal = pif.read_data_ceres(args["file_signal"], args["start"], args["end"], freq)
signal = pif.read_data_lymnaea(args["file_signal"], args["start"], args["end"], freq,curr)
events = pif.read_events_lymnaea(args["file_events"], args["start"], args["end"], freq)

################
#PLOT CREATION
################

########General
fig = plt.figure(figsize=(size_plot*3, size_plot), dpi=dpi)
#print('fig size: {0} DPI, size in inches {1}'.format(fig.get_dpi(), fig.get_size_inches()))
fig.canvas.mpl_connect('key_press_event', onKey)

if signal.c_on == True:
    ax_i = plt.subplot2grid((3, 3), (0, 2), rowspan=3)
    ax_v = plt.subplot2grid((3, 3), (0, 0), colspan=2,rowspan=2)
    ax_c = plt.subplot2grid((3, 3), (2, 0), colspan=2, sharex=ax_v)
    ax_v.xaxis.set_visible(False)
    #ax_v.grid(True)
    #ax_v.set_axisbelow(True)
    #ax_v.xaxis.grid(color='gray', linestyle='dashed')
else:
    ax_i = plt.subplot2grid((1, 3), (0, 2))
    ax_v = plt.subplot2grid((1, 3), (0, 0), colspan=2)

########Voltage
if signal.c_on == True:
    ax_v_n1, = ax_v.plot([], [], label="N1 neuron" , linewidth=0.4)
    ax_v_n2, = ax_v.plot([], [], label="N2 neuron", linewidth=0.4)
    ax_v_n3, = ax_v.plot([], [], label="N3 neuron", linewidth=0.4)
else: 
    ax_v_n1, = ax_v.plot([], [], label="N1 neuron" , linewidth=0.4)
    ax_v_n2, = ax_v.plot([], [], label="N2 neuron", linewidth=0.4)
    ax_v_n3, = ax_v.plot([], [], label="N3 neuron", linewidth=0.4)
ax_v.set_title("Voltage time series")
ax_v.legend(loc=2)
ax_v.set_yticklabels([])
ax_v.set_ylabel("Voltage (10mV/div)")
ax_v.set_ylim(signal.v_min, signal.v_max)
ax_v.xaxis.set_animated(True)

########Current
if signal.c_on == True:
    ax_c_l, = ax_c.plot([], [], label="Injected current" , linewidth=0.4)
    # ax_c_m, = ax_c.plot([], [], label="Current to cell", linewidth=0.4)
    ax_c_m, = ax_c.plot([], [], linewidth=0.4)

    ax_c.legend(loc=2)
    ax_c.set_xlabel("Time (s)")
    ax_c.set_ylabel("Current")
    ax_c.set_ylim(signal.c_min-0.5, signal.c_max+0.5)
    ax_c.xaxis.set_animated(True)
else:
    ax_v.set_xlabel("Time (s)")

# ########Invariant

#Último dato se pintará resaltado en estos ax
ax_i_red_last,  = ax_i.plot([], [], 'go', markersize='10.0')
ax_i_blue_last, = ax_i.plot([], [], 'go', markersize='10.0')

if modo==1:
    #Los datos se añaden progresivamente, declaramos las listas a las cuales se añadira y los ax
    list_inv_red_x, list_inv_red_y, list_inv_blue_x, list_inv_blue_y  = [], [], [], []
    ax_i_blue_progress, = ax_i.plot([], [], 'bo', markersize='1.0')
    ax_i_red_progress,  = ax_i.plot([], [], 'ro', markersize='1.0')

if modo==2:
    #Todos los datos desde el principio y coloreados, no hace falta ni guardar la scatter
    ax_i.scatter(events.fN1_fN1, events.lN2_fN1, marker='o', c=events.firstN1, cmap='Reds',  s=1)
    ax_i.scatter(events.fN1_fN1, events.fN2_fN1, marker='o', c=events.firstN1, cmap='Blues', s=1)

#Barras en el plot del voltage
#Los valores y son definitivos (pos), para que no de error meto valores x dummies [0,0]
list_barra_blue_x, list_barra_red_x, list_barra_black_x = [], [], []
ax_v_event_blue,  = ax_v.plot( [0,0], [signal.pos1, signal.pos1], 'b', marker=6, linestyle='-')
ax_v_event_red,   = ax_v.plot( [0,0], [signal.pos2, signal.pos2], 'r', marker=6, linestyle='-')
ax_v_event_black, = ax_v.plot( [0,0], [signal.pos3, signal.pos3], 'k', marker=6, linestyle='-')
#ax_v_event_green, = ax_v.plot( [0,0], [signal.pos1, signal.pos1], 'y', marker=6, linestyle='-')

##### Fit Blue
fit    = np.polyfit( events.fN1_fN1, events.fN2_fN1, 1)
fit_fn = np.poly1d(fit) 
m      = fit[0] #pendiente
yhat   = fit_fn( events.fN1_fN1)                         
ybar   = np.sum( events.fN2_fN1) / len(events.fN2_fN1)    
ssreg  = np.sum( (yhat-ybar)**2)   
sstot  = np.sum( (events.fN2_fN1 - ybar)**2)   
R2     = ssreg / sstot
#ax_i.plot(events.fN1_fN1, fit_fn(events.fN1_fN1), linewidth= 0.5, c="midnightblue", label="R2={0:.4f}\tm={1:.4f}".format(R2, m).expandtabs())
ax_i.plot(events.fN1_fN1, fit_fn(events.fN1_fN1), linewidth= 0.5, c="midnightblue", label="N2N1 interval\tR2={0:.4f}".format(R2).expandtabs())

##### Fit Red
fit    = np.polyfit( events.fN1_fN1, events.lN2_fN1, 1)
fit_fn = np.poly1d(fit) 
m      = fit[0] #pendiente
yhat   = fit_fn( events.fN1_fN1)                         
ybar   = np.sum( events.lN2_fN1) / len(events.lN2_fN1)    
ssreg  = np.sum( (yhat-ybar)**2)   
sstot  = np.sum( (events.lN2_fN1 - ybar)**2)   
R2     = ssreg / sstot
#ax_i.plot(events.fN1_fN1, fit_fn(events.fN1_fN1), linewidth= 0.5, c="firebrick", label="R2={0:.4f}\tm={1:.4f}".format(R2, m).expandtabs())
ax_i.plot(events.fN1_fN1, fit_fn(events.fN1_fN1), linewidth= 0.5, c="firebrick", label="N2N1 delay\tR2={0:.4f}".format(R2).expandtabs())

##### Estetica plot fit
ax_i.legend()
ax_i.set_title  ("Dynamical invariant")
ax_i.set_xlabel ("Period (s)")
ax_i.set_ylabel ("N2N1 interval, N2N1 Delay (s)")

plt.tight_layout()

########
#UN1ATE
########

def _blit_draw(self, artists, bg_cache):
    # Handles blitted drawing, which renders only the artists given instead
    # of the entire figure.
    updated_ax = []
    for a in artists:
        # If we haven't cached the background for this axes object, do
        # so now. This might not always be reliable, but it's an attempt
        # to automate the process.
        if a.axes not in bg_cache:
            #bg_cache[a.axes] = a.figure.canvas.copy_from_bbox(a.axes.bbox)
            # change here
            bg_cache[a.axes] = a.figure.canvas.copy_from_bbox(a.axes.figure.bbox)
        a.axes.draw_artist(a)
        updated_ax.append(a.axes)

    # After rendering all the needed artists, blit each axes individually.
    for ax in set(updated_ax):
        # and here
        #ax.figure.canvas.blit(ax.bbox)
        ax.figure.canvas.blit(ax.figure.bbox)

# MONKEY PATCH!!
matplotlib.animation.Animation._blit_draw = _blit_draw

def init():
    # Aqui se pone todo lo que implique crear
    ax_v_n1.set_data( [], [] )
    ax_v_n2.set_data( [], [] )
    ax_v_n3.set_data( [], [] )
    if signal.c_on == True:
        ax_c_m.set_data( [], [] )
        ax_c_l.set_data( [], [] )

    ax_i_red_last.set_data ( [], [] )
    ax_i_blue_last.set_data( [], [] )

    ax_v_event_blue.set_data  ( [0,0], [signal.pos1, signal.pos1] )
    ax_v_event_red.set_data   ( [0,0], [signal.pos2, signal.pos2] )
    ax_v_event_black.set_data ( [0,0], [signal.pos3, signal.pos3] )
    #ax_v_event_green.set_data ( [0,0], [signal.pos1, signal.pos1] )

    if modo==1:
        ax_i_red_progress.set_data  ( [], [] )
        ax_i_blue_progress.set_data ( [], [] )
        if signal.c_on == True:
            return (ax_v.xaxis, ax_c.xaxis, ax_v_n1, ax_v_n2, ax_v_n3, ax_c_m, ax_c_l, ax_i_red_last, ax_i_blue_last, ax_v_event_blue, ax_v_event_red, ax_v_event_black, ax_i_red_progress, ax_i_blue_progress)
        else:
            return (ax_v.xaxis, ax_v_n1, ax_v_n2, ax_v_n3, ax_i_red_last, ax_i_blue_last, ax_v_event_blue, ax_v_event_red, ax_v_event_black, ax_i_red_progress, ax_i_blue_progress)
    else:
        if signal.c_on == True:
            return (ax_v.xaxis, ax_c.xaxis, ax_v_n1, ax_v_n2, ax_v_n3, ax_c_m, ax_c_l, ax_i_red_last, ax_i_blue_last, ax_v_event_blue, ax_v_event_red, ax_v_event_black)
        else:
            return (ax_v.xaxis, ax_v_n1, ax_v_n2, ax_v_n3, ax_i_red_last, ax_i_blue_last, ax_v_event_blue, ax_v_event_red, ax_v_event_black)


def update(i):
    global ini_ventana, pts_avance, index_event, pts_ventana
    ini_ventana += pts_avance
    fin =  ini_ventana+pts_ventana

    # Limites de tiempo alcanzados o velocidad parada
    if ini_ventana<0:
        pts_avance = 0
        ini_ventana = 0
    elif fin > signal.num_points or (index_event)>=(len(events.firstN2)-2):
        pts_avance = 0
        ini_ventana = signal.num_points - pts_ventana -1
        fin = ini_ventana + pts_ventana
    elif pts_avance == 0:
        if modo == 1:
            if signal.c_on == True:
                return (ax_v.xaxis, ax_c.xaxis, ax_v_n1, ax_v_n2, ax_v_n3, ax_c_m, ax_c_l, ax_i_red_last, ax_i_blue_last, ax_v_event_blue, ax_v_event_red, ax_v_event_black, ax_i_red_progress, ax_i_blue_progress)
            else:
                return (ax_v.xaxis, ax_v_n1, ax_v_n2, ax_v_n3, ax_i_red_last, ax_i_blue_last, ax_v_event_blue, ax_v_event_red, ax_v_event_black, ax_i_red_progress, ax_i_blue_progress)
        else:
            if signal.c_on == True:
                return (ax_v.xaxis, ax_c.xaxis, ax_v_n1, ax_v_n2, ax_v_n3, ax_c_m, ax_c_l, ax_i_red_last, ax_i_blue_last, ax_v_event_blue, ax_v_event_red, ax_v_event_black)
            else:
                return (ax_v.xaxis, ax_v_n1, ax_v_n2, ax_v_n3, ax_i_red_last, ax_i_blue_last, ax_v_event_blue, ax_v_event_red, ax_v_event_black)

    # Actualizamos las señales V y C a los rangos nuevos
    ax_v_n1.set_data( signal.t[ini_ventana:fin], signal.v_n1[ini_ventana:fin] )
    ax_v_n2.set_data( signal.t[ini_ventana:fin], signal.v_n2[ini_ventana:fin] )
    ax_v_n3.set_data( signal.t[ini_ventana:fin], signal.v_n3[ini_ventana:fin] )
    ax_v.set_xlim( signal.t[ini_ventana], signal.t[fin-1] )
    if signal.c_on == True:
        # ax_c_m.set_data( signal.t[ini_ventana:fin], signal.c_m[ini_ventana:fin] )
        ax_c_l.set_data( signal.t[ini_ventana:fin], signal.c[ini_ventana:fin] )
        ax_c.set_xlim( signal.t[ini_ventana], signal.t[fin-1] )

    #### EVENTOS
    ref_ini = signal.t[ini_ventana]
    ref_fin = signal.t[ini_ventana+pts_ventana-1]
    ref_event = events.firstN2[index_event+1] #El evento de referencia

    # Avanzando hacia delante AND eventos pendientes
    if pts_avance>0 and index_event<(events.num_events-1) and ref_event <= ref_fin:
        update_events(index_event)
        index_event+=1
            
    # Avanzando hacia atras AND se han recorrido eventos
    elif pts_avance<0 and index_event!=0 and ref_event>=ref_fin:
        index_event-=1
        update_events(index_event)

    if modo==1:
        if signal.c_on == True:
            return (ax_v.xaxis, ax_c.xaxis, ax_v_n1, ax_v_n2, ax_v_n3, ax_c_m, ax_c_l, ax_i_red_last, ax_i_blue_last, ax_v_event_blue, ax_v_event_red, ax_v_event_black, ax_i_red_progress, ax_i_blue_progress)
        else: 
            return (ax_v.xaxis, ax_v_n1, ax_v_n2, ax_v_n3, ax_i_red_last, ax_i_blue_last, ax_v_event_blue, ax_v_event_red, ax_v_event_black, ax_i_red_progress, ax_i_blue_progress)
    else:
        if signal.c_on == True:
            return (ax_v.xaxis, ax_c.xaxis, ax_v_n1, ax_v_n2, ax_v_n3, ax_c_m, ax_c_l, ax_i_red_last, ax_i_blue_last, ax_v_event_blue, ax_v_event_red, ax_v_event_black)
        else:
            return (ax_v.xaxis, ax_v_n1, ax_v_n2, ax_v_n3, ax_i_red_last, ax_i_blue_last, ax_v_event_blue, ax_v_event_red, ax_v_event_black)

def update_events(index_event):
    #### Funcion que actualiza los eventos
    #print(events.firstN2 [index_event+1] - events.lastN1 [index_event]) 

    # #### Dato resaltado en verde en los puntos del invariante
    ax_i_red_last.set_data  ( events.fN1_fN1[index_event], events.lN2_fN1[index_event] )
    ax_i_blue_last.set_data ( events.fN1_fN1[index_event], events.fN2_fN1[index_event] )

    #### Actualización de las tres barras sobre el voltaje
    ax_v_event_blue.set_xdata  ( [ events.firstN2 [index_event+1], events.firstN1 [index_event+2] ] )
    ax_v_event_red.set_xdata   ( [ events.lastN2  [index_event+1], events.firstN1 [index_event+2] ] )
    ax_v_event_black.set_xdata ( [ events.firstN1 [index_event+1], events.firstN1 [index_event+2] ] )
    #ax_v_event_green.set_xdata ( [ events.lastN1 [index_event], events.firstN2 [index_event+1] ] )
    
    if modo==1:
        # En este modo los puntos del invariante nuevos no estan

        # Añadimos valores a las listas
        list_inv_red_x.append ( events.fN1_fN1 [index_event] )
        list_inv_red_y.append ( events.lN2_fN1 [index_event] )
        list_inv_blue_x.append( events.fN1_fN1 [index_event] )
        list_inv_blue_y.append( events.fN2_fN1 [index_event] )

        # Las listas al objeto axe correspondiente
        ax_i_red_progress.set_data  ( list_inv_red_x,  list_inv_red_y  )
        ax_i_blue_progress.set_data ( list_inv_blue_x, list_inv_blue_y )

########
#MAIN
########
print ("Duración señal   = ", int(signal.num_points/freq), "s")
print("DPI              = ", dpi)
print("Resolucion       = ", dpi*size_plot*3, "x", dpi*size_plot)
print("FPS              = ", fps)

if save == '0':
    anim = FuncAnimation(fig, update, interval=interval, repeat=False, blit=True, init_func=init)
    plt.show()

else:
    frames = (signal.num_points - pts_ventana) / pts_avance # Calculo de cuantos avances hay que producir para llegar al final de la señal
    print("Frames           = ", int(frames))
    print("Est. t dpi100    = ", int(frames*150/42349)+1, "min")
    print("Est. t dpi300    = ", int(frames*230/42349)+1, "min")
    anim = FuncAnimation(fig, update, interval=interval, repeat=False, blit=True, init_func=init, frames=int(frames))
    print("Inicio           =  " + '{:%H:%M:%S}'.format(datetime.datetime.now()))
    anim.save(save+".mp4", dpi=dpi, writer='ffmpeg', bitrate=-1)
    print("Fin              =  " + '{:%H:%M:%S}'.format(datetime.datetime.now()))