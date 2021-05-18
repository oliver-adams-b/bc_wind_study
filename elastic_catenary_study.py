import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize
from functools import partial

class catenary_approximation():
    bc_data = pd.read_csv("bc_webbing_stretch.csv")
    
    bc_mpuls = {"Green":0.053, 
                "Jybn":0.051,
                "Feather PRO":0.051,
                "Blue":0.055,
                "Spider Silk MK3":0.038, 
                "Lift 2be":0.048, 
                "Jelly Webbing":0.045}
    
    bc_widths = {"Green":0.0254, 
                 "Jybn":0.0254, 
                 "Feather PRO":0.0254, 
                 "Blue":0.0254, 
                 "Spider Silk MK3":0.0254, 
                 "Lift 2be": 0.0254, 
                 "Jelly Webbing": 0.0254}
    
    bc_wlls = {"Green":5200,
               "Jybn":5200, 
               "Feather PRO":5800, 
               "Blue":5000, 
               "Spider Silk MK3":7600, 
               "Lift 2be":4300, 
               "Jelly Webbing":4000}
    
    def __init__(self, webbing_type = "Green"):
        self.webbing_type = webbing_type
        bc_type = self.bc_data[["Force", webbing_type]]
        bc_type[webbing_type] = [float(x.replace("%", ""))/100 for x in bc_type[webbing_type]]
        bc_type["Force"] = [float(x.replace(" kN", ""))*1000 for x in bc_type["Force"]]
        bc_type["k"] = bc_type["Force"]/bc_type[webbing_type]
        
        self.k = np.mean(bc_type["k"]) # approximate elastic modulus
        self.lam = 2*self.bc_mpuls[webbing_type] # mass per unit length
        self.baseline_tension = 2000
        self.wll = self.bc_wlls[webbing_type]
        self.width = 2*self.bc_widths[webbing_type]  #ASSUMPTION HERE
        
        
    def g_from_v(self, v, rho = 1.225): #ASSUMPTION HERE
        return (self.width*rho)/(self.lam) * v**2
    
    def cat_f(self, #unused but it's still here
              x, 
              g = 1):
        
        a  = self.baseline_tension / (self.lam*g)
        sinh = np.sinh((1/a)*(x-(self.baseline_tension/self.k)))
        return a*np.power(1+sinh**2, 1/2) + (self.baseline_tension*sinh)/(2*self.k)

    def T(self, 
          x, #the point at which you want to compute the tension
          g = 1, 
          T_0 = 1):
        
        a  = T_0 / (self.lam*g)
        sinh = np.sinh((1/a)*(x-(T_0/self.k)))
        
        return T_0*np.power(1+sinh**2, 1/2)
    
    def plot_ws_ll_danger_zone(self, 
                               length_range = [0, 1000], 
                               wind_speed_range = [0, 35], 
                               show = True):
         #plot wind-speed line-length danger zone
         t_vals = []
         wf_ll_vals = []
        
         for wf in np.linspace(wind_speed_range[0], wind_speed_range[1], 200):
             for ll in np.linspace(length_range[0], length_range[1], 200):
                
                 g = self.g_from_v(wf)
                
                 wf_ll_vals.append([wf, ll])
                
                 t = self.T(ll, 
                             g = g, 
                             T_0 = self.baseline_tension)
                
                 t_vals.append(np.log10(t))
        
         wf_ll_vals = np.array(wf_ll_vals)
         t_vals = np.array(t_vals)
        
         
         levels = wf_ll_vals[t_vals.astype(int) >= np.log10(self.wll)]
        
         if show:
             fig, ax = plt.subplots(1, 2, figsize = (10, 10))
             plt.subplots_adjust(left = 0, right = 1)
             s0 = ax[0].scatter(wf_ll_vals[:, 0], 
                        wf_ll_vals[:, 1],
                        c = t_vals, 
                        s = 10, 
                        cmap = "tab20c")
            
             plt.colorbar(s0, ax = ax[0], 
                          label = "$log$(Tension [N])", 
                          pad = 0.1, orientation="horizontal")
            
             ax[0].scatter(levels[:, 0], 
                        levels[:, 1], 
                        c = "r", s = 10, 
                        alpha = .1, 
                        marker = "s",
                        label = "No Go Zone, Tension Exceeds\n WLL: {}".format(self.wll))
            
             ax[0].legend(bbox_to_anchor=(1.4, -0.35), loc='lower right', fancybox = True)
             ax[0].set_ylabel("Line length $[m]$")
             ax[0].set_xlabel("Approximate Wind Speed $[m/s^2]$")
             ax[0].set_title("$log$(Tension) as a Function of \n Line Length and Wind Speed")
             
             s1 = ax[1].scatter(wf_ll_vals[:, 0], 
                        wf_ll_vals[:, 1],
                        c = np.exp(t_vals), 
                        s = 10, 
                        cmap = "tab20c")
             
             plt.colorbar(s1, ax = ax[1],
                          label = "Tension [N]", 
                          pad = 0.1, orientation="horizontal")
         
         levels = wf_ll_vals[10**t_vals >= self.wll]
         
         if show:
             ax[1].scatter(levels[:, 0], 
                           levels[:, 1], 
                           c = "r", s = 10, 
                           alpha = .3, 
                           marker = "s")
             
             ax[1].set_ylabel("Line length $[m]$")
             ax[1].set_xlabel("Approximate Wind Speed $[m/s^2]$")
             ax[1].set_title("Tension as a Function of \n Line Length and Wind Speed")
             
             fig.suptitle("Approximate Wind-Speed Line-Length Safety Boundaries for BC {}".format(self.webbing_type), fontsize="x-large")
             plt.show()
         
         tol = 20
         t_vals = 10**t_vals
         self.wl_boundary = wf_ll_vals[(t_vals.astype(int) <= self.wll + tol) & (t_vals.astype(int) >= self.wll - tol)]
         
         def exp(x, a, b, c):
             return np.exp(-a*x-b)+c
         
         fit = optimize.curve_fit(exp, self.wl_boundary[:, 0], self.wl_boundary[:, 1])
         self.fit = partial(exp, a = fit[0][0], b = fit[0][1], c = fit[0][2])
         
"""
Estimating the theoretical average danger boundary for a selection of BC webbing types:
"""  
green = catenary_approximation("Green")
names = list(green.bc_mpuls.keys())
fits = [] #will be a list of functions
for name in names:
    bc_name = catenary_approximation(name)
    bc_name.plot_ws_ll_danger_zone(show = False)
    fits.append(bc_name.fit)
 
    
wind_speed_range = [5, 35]
avg_pts = []
for w in np.linspace(wind_speed_range[0], wind_speed_range[1], 300):
    temp_l = 0
    for func in fits:
        temp_l += func(w)
    avg_pts.append([w, temp_l/len(fits)])

avg_pts = np.asarray(avg_pts)
plt.plot(avg_pts[:, 0], 
         avg_pts[:, 1], 
         c = "r")
plt.fill_between(avg_pts[:, 0], 
                 avg_pts[:, 1], 
                 2000, 
                 alpha = 0.35, 
                 facecolor = "red", 
                 label = "Modeled Danger Zone - Theoretical\n Forces From Wind Exceed WLL")

"""
Comparing real-world data to the model!
"""
wind_data = pd.read_csv("pq_wind_data.csv")
wind_data = wind_data.loc[:, ['VS Avrg Gust kph',
                              'Highline length (in m)',
                              'Failed Line',
                              'Failed Anchors',
                              'Anchor Damage',
                              'Line Damage']]

wind_data = wind_data.rename(columns = {"VS Avrg Gust kph":'wind speed'})
wind_data["wind speed"] = wind_data["wind speed"] * 0.27777 #convert kph to m/s
wind_data = wind_data[wind_data["wind speed"] > 5]
failure_types = ['Failed Line',
                'Failed Anchors',
                'Anchor Damage',
                'Line Damage']

markers = ["o", "d", "+", "x", "*", "|"]

for i, failure_type in enumerate(failure_types):
    temp_wind_data = wind_data[wind_data[failure_type] == 1]
    plt.scatter(temp_wind_data["wind speed"],
                temp_wind_data["Highline length (in m)"], 
                label = failure_type, 
                marker = markers[i], 
                alpha = 0.35,
                s = 180)


plt.legend(bbox_to_anchor=(1.7, .55 ), loc='lower right', fancybox = True)
plt.xlabel("Avg Wind Gust $[m/s]$")
plt.ylabel("Line Length $[m]$")

plt.title("""Theoretical Wind Speed / Line Length Danger Zone \n Versus Real-World Slackline Incidents Involving Wind""")
plt.show()

