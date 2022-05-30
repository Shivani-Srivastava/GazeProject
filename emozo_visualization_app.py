import streamlit as st
import numpy as np
import pandas as pd
import json
import csv
import glob
from statistics import mean


st.title('Emozo Data')

datasets = st.selectbox('Pick one', ['Student Data', 'Subtitle Data', 'Demo Data'])

if datasets == 'Student Data':
	for file in glob.glob('/home/shivani/Documents/Emozo/Student/**.csv', recursive = True):
		df = pd.read_csv(file)
		st.write('Sample size for this file',len(df['sections']))
		st.write('File Name')
		st.write(file)
	
		df_dist = []
		df_mean_dist = []
		df3 = pd.DataFrame()
		
		
		
		for i in range(len(df['sections'])):
			a = df['sections'][i]
			a0 = json.loads(a)
		
			dist = 0
			loc_dist=[]
			
			for j in range(len(a0[0]['assets'][0]['gazedata'])-1):
				t00 = a0[0]['assets'][0]['gazedata'][j]
				t01 = a0[0]['assets'][0]['gazedata'][j+1]
			
				point1 = np.array((int(t00['x']), int(t00['y'])))
				point2 = np.array((int(t01['x']), int(t01['y'])))
			
				sum_sq = np.sum(np.square(point1 - point2))
				dist0 = np.sqrt(sum_sq)
				dist += dist0
				loc_dist.append(round(dist0,3))
			#s = pd.Series(loc_dist)
		#print(s.describe())
		#print('For user',i,' in test',file)
		#print(dist)
			
			
	
			
			df_dist.append(dist)
		#print(mean(loc_dist))
			df_mean_dist.append(mean(loc_dist))
		st.write(mean(df_dist)," is the mean distance travelled by all users.")
		st.write(mean(df_mean_dist)," is the mean distance travelled per unit time by all users.")
		
	

if datasets == 'Subtitle Data':
	for file in glob.glob('/home/shivani/Documents/Emozo/Dummy/**.csv', recursive = True):
		df = pd.read_csv(file)
		st.write('Sample size for this file',len(df['assets']))
		st.write('File Name')
		st.write(file)
	
		df_dist = []
		df_mean_dist = [0]
		df3 = pd.DataFrame()
		
		for i in range(len(df['assets'])):
			a = df['assets'][i]
			a0 = json.loads(a)
			
			dist = 0
			loc_dist=[0]
		
			for j in range(len(a0[0]['gazedata'])-1):
				t00 = a0[0]['gazedata'][j]
				t01 = a0[0]['gazedata'][j+1]
				point1 = np.array((int(t00['x']), int(t00['y'])))
				point2 = np.array((int(t01['x']), int(t01['y'])))
				sum_sq = np.sum(np.square(point1 - point2))
				dist0 = np.sqrt(sum_sq)
			#print(dist0)
			
				dist += dist0
			
				loc_dist.append(round(dist0,3))
		#s = pd.Series(loc_dist)
		#print(s.describe())
		#print('For user',i,' in test',file)
		#print(dist)
			df_dist.append(dist)
		#print(mean(loc_dist))
			df_mean_dist.append(mean(loc_dist))
			

			
		st.write(mean(df_dist)," is the mean distance travelled by all users.")
		st.write(mean(df_mean_dist)," is the mean distance travelled per unit time by all users.")
		
if datasets == 'Demo Data':
	for file in glob.glob('/home/shivani/Documents/Emozo/DemoData/**.csv', recursive = True):
		df = pd.read_csv(file)
		st.write('Sample size for this file',len(df['assets']))
		st.write('File Name')
		st.write(file)
	
		df_dist = []
		df_mean_dist = [0]
	
		for i in range(len(df['assets'])):
			a = df['assets'][i]
			a0 = json.loads(a)
			
			dist = 0
			loc_dist=[0]
		
			for j in range(len(a0[0]['gazedata'])-1):
				t00 = a0[0]['gazedata'][j]
				t01 = a0[0]['gazedata'][j+1]
				point1 = np.array((int(t00['x']), int(t00['y'])))
				point2 = np.array((int(t01['x']), int(t01['y'])))
				sum_sq = np.sum(np.square(point1 - point2))
				dist0 = np.sqrt(sum_sq)
			#print(dist0)
			
				dist += dist0
				loc_dist.append(round(dist0,3))
		#s = pd.Series(loc_dist)
		#print(s.describe())
		#print('For user',i,' in test',file)
		#print(dist)
			df_dist.append(dist)
		#print(mean(loc_dist))
			df_mean_dist.append(mean(loc_dist))
		st.write(mean(df_dist)," is the mean distance travelled by all users.")
		st.write(mean(df_mean_dist)," is the mean distance travelled per unit time by all users.")
		
			
	
