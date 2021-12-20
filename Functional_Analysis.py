#!/usr/bin/env python
# coding: utf-8

# # Analysis

# This file contains functions which cleans and processes them to yield the following: 
# 
# - CSV file of X,Y co-ordinates (per unit time)
# - CSV file of Sentiments (per unit time)
# - Still image of X,Y co-ordinate (per unit time)
# - Still image of locus of X,Y co-ordinate (per user)
# - GIF of X,Y co-ordinate (animated by unit time)
# - Heat Maps divided into coarse units (per unit time)
# 
# ---
# 
# 

# #### Description
# 
# The following code takes the raw input as given, extracts the gaze co-ordinates according to time and user.
# Essentially, one is transpose of the other.

# In[1]:


import pandas as pd
import csv
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import glob


df = pd.read_csv('https://raw.githubusercontent.com/Shivani-Srivastava/GazeProject/main/report-summary1.csv')
col_name = list(df.columns)


#Blank list to record index number of relevant columns
indx = []

#Relevant here are all columns that are capturing coordinates of Eye Gaze

for i in range(len(col_name)):
    if 'Eye Gaze' in col_name[i]:
        indx.append(i)


gaze = []

for i in indx:
    gaze.append(col_name[i])

df["Multiplier_X"] = 1280 * ( 1 / (df["playerWidth"]))
df["Multiplier_Y"] = 720 * ( 1 / (df["playerHeight"]))

    #return df

#############################################################
#                  Per Second Plotter			    #
#############################################################

def sec_n(n):
	
    #Nth Second
    gaze0 = df[gaze[n]]
    gaze0_x = []
    gaze0_y = []

    arr = np.zeros((df.shape[0],2))
      
    for i in range(df.shape[0]):
        try:
            x_coord = list(gaze0)[i].split(',')[0]
            x_val = int(x_coord) * df["Multiplier_X"][i]
            gaze0_x.append(round(x_val))
            y_coord = list(gaze0)[i].split(',')[1].strip(' ')
            y_val = int(y_coord) * df["Multiplier_Y"][i]
            gaze0_y.append(round(y_val))
            arr[i][0] = x_val
            arr[i][1] = y_val
        except:
            pass
    return arr
    

PathData = pd.DataFrame()
for i in range(30):
	PathData[f"Second{i}_X"] = sec_n(i)[:,1]
	PathData[f"Second{i}_Y"] = sec_n(i)[:,0]
	print("Done for", i)
print(PathData)

PathData.to_csv('PathDataPerSecond_CB.csv',index = False, header = True)	


##########################################################
#		User Path Plotter			 #
##########################################################

def UserPath(UsrIndex):
    user1_eyegaze = []
    for i in gaze:
        user1_eyegaze.append(df.loc[UsrIndex][i])
        
    x_coord = []
    y_coord = []
    
    arr = np.zeros((len(gaze),2))
    
    for i in range(len(user1_eyegaze)):
        try:      
            x1 = user1_eyegaze[i].split(',')[0]
            y1 = user1_eyegaze[i].split(',')[1].strip(' ')
            x_coord.append(int(x1))
            y_coord.append(int(y1))
            arr[i][0] = x1
            arr[i][1] = y1
        except:
            pass
        
    
    #print(arr.sum())
    return arr

PathData2 = pd.DataFrame()

for i in range(df.shape[0]):
	PathData2[f"User_{i}_X"] = UserPath(i)[:,1]
	PathData2[f"User_{i}_Y"] = UserPath(i)[:,0]
print(PathData2)

PathData2.to_csv('PathDataXY.csv',index = False, header = True)	




# ### Sentiment Extraction

# In[2]:


#df = pd.read_csv("report-summaryCB.csv")


col_name = list(df.columns)
indx = []

for i in range(len(col_name)):
    if 'Neutral' in col_name[i]:
            indx.append(i)
            
df.iloc[:,indx]

a =[indx[0],indx[0]+1,indx[0]+2, indx[0]+3,indx[0]+4, indx[0]+5, indx[0]+6]

sec_0_sentiment = df.iloc[:,a]


Neutral = []
Happy = []
Surprised = []
Sad = []
Scared = []
Angry = []
Disgust = []

for i in range(50):
    Neutral.append(sec_0_sentiment.iloc[i,:].round(4)[0])
    Happy.append(sec_0_sentiment.iloc[i,:].round(4)[1])
    Surprised.append(sec_0_sentiment.iloc[i,:].round(4)[2])
    Sad.append(sec_0_sentiment.iloc[i,:].round(4)[3])
    Scared.append(sec_0_sentiment.iloc[i,:].round(4)[4])
    Angry.append(sec_0_sentiment.iloc[i,:].round(4)[5])
    Disgust.append(sec_0_sentiment.iloc[i,:].round(4)[6])

Neutral_Sentiment = pd.Series(Neutral)
Happy_Sentiment = pd.Series(Happy)
Surprised_Sentiment = pd.Series(Surprised)
Sad_Sentiment = pd.Series(Sad)
Scared_Sentiment = pd.Series(Scared)
Angry_Sentiment = pd.Series(Angry)
Disgust_Sentiment = pd.Series(Disgust)

frame = {'Neutral': Neutral_Sentiment, 'Happy': Happy_Sentiment, 'Surprised':Surprised_Sentiment, 'Sad': Sad_Sentiment, 'Scared':Scared_Sentiment, 'Angry':Angry_Sentiment,'Disgust':Disgust_Sentiment}

Emotions = pd.DataFrame(frame)

Emotions.to_csv('emotions.csv')


# ## Plotting Gaze points as a ScatterPlot

# In[3]:


def plot_N(n):
	gaze0 = df[gaze[n]]
	
	gaze0_x = []
	gaze0_y = []

	for i in range(len(gaze0)):
		try:
		        x_coord = list(gaze0)[i].split(',')[0]
		        x_val = int(x_coord) * df["Multiplier_X"][i]
		        gaze0_x.append(round(x_val))
		        y_coord = list(gaze0)[i].split(',')[1].strip(' ')
		        y_val = int(y_coord) * df["Multiplier_Y"][i]
		        gaze0_y.append(round(y_val))
		except:
		        pass
		
	fig = px.scatter(x=gaze0_x, y=gaze0_y)
	fig.update_traces(marker = dict(size = 12, line = dict(width = 2, color = 'DarkSlateGrey')), selector = dict(mode = 'markers'))
	#fig.update_layout({‘plot_bgcolor’: ‘rgba(0, 0, 0, 0)’,‘paper_bgcolor’: ‘rgba(0, 0, 0, 0)’,})
	fig.update_xaxes(range=[0, 1280])
	fig.update_yaxes(range=[0, 720])
	fig.show()


# In[5]:


plot_N(20)

# Scatter plot for the 20th second


# In[7]:


for i in range(len(indx)):
    plot_N(i)


# ## Heatmaps / Grid-ification
# 
# This code will transform the XY coordfile on a per second basis, to a scaled down grid version of 9 * 4.
# 
# 
# 
# 

# In[8]:


# Functions to turn co-ordinates into grid address

def grid_x(x):
    return round(x/(1280/9))

def grid_y(y):
    return round(y/(720/4))


df = pd.read_csv('PathDataPerSecond_CB.csv')

col_name = list(df.columns)


# Blank list to record index number of relevant columns

indx_x = []
indx_y = []

# Relevant here are all columns that are capturing coordinates of Eye Gaze

for i in range(len(col_name)):
    if '_X' in col_name[i]:
        indx_x.append(i)
    elif '_Y' in col_name[i]:
        indx_y.append(i)

colname_X = []
colname_Y = []
        
for i in indx_x:
    colname_X.append(col_name[i])

for i in indx_y:
    colname_Y.append(col_name[i])
    
# Transforming Co-ords into Grids

for i in colname_X:
    df[i] = df[i].apply(grid_x)

for i in colname_Y:
    df[i] = df[i].apply(grid_y)
    

df.head()

    


# This code will convert the scatterplots into heatplots.

# In[16]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage.filters import gaussian_filter


def myplot(x, y, s, bins=1000):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
    heatmap = gaussian_filter(heatmap, sigma=s)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent




def plot_heatmap(x,y):
        
    fig, axs = plt.subplots(2, 2)

    sigmas = [0, 16, 32, 64]
    
    for ax, s in zip(axs.flatten(), sigmas):
        if s == 0:
            ax.plot(x, y, 'k.', markersize=5)
            ax.set_title("Scatter plot")
        else:
            img, extent = myplot(x, y, s)
            ax.imshow(img, extent=extent, origin='lower', cmap=cm.jet)
            ax.set_title("Smoothing with  $\sigma$ = %d" % s)
    return plt.show()


# In[17]:


plot_heatmap(df['Second0_X'],df['Second0_Y'])


# In[20]:


size = df.shape

for i in range(int(size[1]/2 - 1)):
    plot_heatmap(df[f'Second{i}_X'],df[f'Second{i}_Y'])


# In[ ]:




