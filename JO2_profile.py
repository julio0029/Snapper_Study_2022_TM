#!/usr/bin/env python3
#-*- coding: utf-8 -*-

'''-------------------------------------------------------------------------------
Copyright© 2021 Jules Devaux & Tristan McArley. All Rights Reserved
Open Source script under Apache License 2.0
----------------------------------------------------------------------------------
'''


import os
current_path = os.path.dirname(os.path.abspath(__file__))


#======== PARAMETERS =========
MAX_kPA = 20 #kPa
GRAPHING = True
_saving = True # SAve summary file and stats

TEMPERATURES=[21,25]
GROUPS=['control1', 'control2', 'group1', 'group2', 'group3']
TISSUES=['heart', 'red_muscle']
DISPLAY='mean_fitted_normalised'
Y_axis_RANGE=(0,150)
X_axis_RANGE=(0,MAX_kPA)
#=============================


OUTLIERS={}

import sys, pickle
import numpy as np
import pandas as pd
import pingouin as pg
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt


X=np.arange(0, MAX_kPA, 0.01)


def graph(_dict):

	'''
	Expect dict with len first orer being rows, second order being columns
	'''
	rows=len(_dict)
	cols=len(_dict[0])

	fig, axes= plt.subplots(rows,cols)

	for ridx in range(rows):
		for cidx in range(cols):
			axes[ridx,cidx].plot(_dict[ridx][cidx])

			axes[ridx,cidx].set_ylim(Y_axis_RANGE)
			axes[ridx,cidx].set_xlim(X_axis_RANGE)
			axes[ridx,cidx].set_xticks(np.arange(0,MAX_kPA,5))
			axes[ridx,cidx].grid(True)
			if cidx==0:axes[ridx,cidx].set_ylabel(f"{state}\n(pmol.(s*mg)-1)")
			else:axes[ridx,cidx].yaxis.set_ticklabels([])
			if ridx==0:
				axes[ridx,cidx].set_title(f"{temp}ºC", fontsize=10, fontweight='bold')
				axes[ridx,cidx].xaxis.set_ticklabels([])

	axes[1,3].set_xlabel('PO2 (kPa)')
	plt.subplots_adjust(wspace=0,hspace=0)
	plt.show()




def JO2_profile(_return='average'):

	with open('chambers.pkl','rb') as f:
		chambers=pickle.load(f)

	#df=pd.DataFrame(index=X)
	chbs={}
	
	for state in STATES:
		for temp in TEMPERATURES:
			_id=0 # counter to loop trhough diplicates

			for chamber in chambers:
				if chamber['temperature']==temp and chamber['state']==state and chamber['filename'] not in OUTLIERS[state]:
					_name=f"{state}-{temp}-{_id}"
					if 'raw' in DISPLAY:
						chbs.update({_name:chamber['df'].rename(columns={'JO2':_name})[_name]})

					elif 'fitted' in DISPLAY:
						chbs.update({_name: chamber['predicted'].rename(columns={'JO2':_name})[_name]})
					_id+=1


	if 'raw' in DISPLAY:
		df=[v for k,v in chbs.items()]
		df=pd.concat(df, axis=1, ignore_index=False).interpolate(method='linear')

	elif 'fitted' in DISPLAY:
		#df=pd.DataFrame(chbs, index=X)
		df=pd.concat(chbs, axis=1)

	meandf=pd.DataFrame(index=X)
	for temp in TEMPERATURES:
		for state in STATES:
			mask=[c for c in df.columns if (str(temp) in c) and (state in c)]
			tdf=df.loc[:,mask]
			meandf[f"{state}-{temp}-mean"]=tdf.mean(axis=1)
			meandf[f"{state}-{temp}-semm"]=tdf.mean(axis=1)-tdf.sem(axis=1)
			meandf[f"{state}-{temp}-semM"]=tdf.mean(axis=1)+tdf.sem(axis=1)


	if _return == 'all': return df
	elif _return == 'average': return meandf



def graph_profile():

	with open('chambers.pkl','rb') as f:
		chambers=pickle.load(f)

	data={}

	df=pd.DataFrame()


	#========================= Plot for each temp and state ================================
	fig, axes= plt.subplots(len(TISSUES),len(GROUPS))

	for group in GROUPS:
		tidx=GROUPS.index(group)

		tdf={t:[] for t in GROUPS}
		for tissue in TISSUES:
			data.update({tissue:{group:[]}})
			sidx=TISSUES.index(tissue)

			tsdf=[]

			for chamber in chambers:
				if chamber['Group']==group and chamber['tissue']==tissue:

					if DISPLAY=='all_raw':
						axes[sidx,tidx].plot(chamber['df']['JO2'], label=f"{chamber['filename']}")

					elif DISPLAY=='all_fitted':
						axes[sidx,tidx].plot(X,chamber['predicted'], label=f"{chamber['filename']}")


					if 'raw' in DISPLAY:
						tsdf.append(chamber['df'].rename(columns={'JO2':f"{tissue}_{group}"}))
					elif 'fitted' in DISPLAY:
						df=pd.DataFrame(chamber['predicted'],X).rename(columns={'JO2':f"{tissue}_{group}"})
						tsdf.append(df)


			if 'mean' in DISPLAY:
				tsdf=pd.concat(tsdf, axis=1, ignore_index=False)
				print(tsdf)
				tsdf['mean']=tsdf.mean(axis=1)
				tsdf['semm']=tsdf.mean(axis=1)-tsdf.sem(axis=1)
				tsdf['semM']=tsdf.mean(axis=1)+tsdf.sem(axis=1)

				if 'scope' not in DISPLAY:
					if tissue=='heart': color=['#6C0020','#E70044']
					else: color=['#309BCD','#C3E2F0']
					axes[sidx,tidx].plot(X,tsdf['mean'], label=f"{chamber['filename']}", color=color[0])
					axes[sidx,tidx].plot(X,tsdf['semm'], label=f"{chamber['filename']}", linestyle='--',color=color[1])
					axes[sidx,tidx].plot(X,tsdf['semM'], label=f"{chamber['filename']}", linestyle='--',color=color[1])

				else:tdf[group].append(tsdf.loc[:,['mean','semm','semM']])

			# ---- Sort axis labels and so on

			#axes[sidx,tidx].legend(loc='upper right', fontsize='xx-small')
			#sidx=0
			axes[sidx,tidx].set_ylim(Y_axis_RANGE)
			axes[sidx,tidx].set_xlim(X_axis_RANGE)
			axes[sidx,tidx].set_xticks(np.arange(0,MAX_kPA,5))
			axes[sidx,tidx].grid(True)
			if tidx==0:axes[sidx,tidx].set_ylabel(f"{tissue}\n(pmol.(s*mg)-1)")
			else:axes[sidx,tidx].yaxis.set_ticklabels([])
			if sidx==0:
				axes[sidx,tidx].set_title(f"{group}", fontsize=10, fontweight='bold')
				axes[sidx,tidx].xaxis.set_ticklabels([])

	axes[1,1].set_xlabel('PO2 (kPa)')
	plt.subplots_adjust(wspace=0,hspace=0)
	plt.show()




def main():

	with open('chambers.pkl','rb') as f:
		chambers=pickle.load(f)

	#df=pd.DataFrame(index=X)
	chbs={}
	
	for tissue in TISSUES:
		for group in GROUPS:
			_id=0 # counter to loop trhough diplicates

			for chamber in chambers:
				if chamber['Group']==group and chamber['tissue']==tissue:
					_name=f"{tissue}-{group}-{_id}"
					if 'raw' in DISPLAY:
						chbs.update({_name:chamber['df'].rename(columns={'JO2':_name})[_name]})

					elif 'fitted' in DISPLAY:
						chbs.update({_name: chamber['predicted'].rename(columns={'JO2':_name})[_name]})
					_id+=1


	if 'raw' in DISPLAY:
		df=[v for k,v in chbs.items()]
		df=pd.concat(df, axis=1, ignore_index=False).interpolate(method='linear')

	elif 'fitted' in DISPLAY:
		#df=pd.DataFrame(chbs, index=X)
		df=pd.concat(chbs, axis=1)

	# Create Excel template
	fileName=f'JO2_profiles{DISPLAY}.xlsx'
	writer=pd.ExcelWriter(fileName, engine='xlsxwriter')
	# First, add data to first tab
	df.to_excel(writer, sheet_name='all_fitted') 

	meandf=pd.DataFrame(index=X)
	for group in GROUPS:
		for tissue in TISSUES:
			mask=[c for c in df.columns if (str(group) in c) and (tissue in c)]
			tdf=df.loc[:,mask]
			meandf[f"{tissue}-{group}-mean"]=tdf.mean(axis=1)
			meandf[f"{tissue}-{group}-semm"]=tdf.mean(axis=1)-tdf.sem(axis=1)
			meandf[f"{tissue}-{group}-semM"]=tdf.mean(axis=1)+tdf.sem(axis=1)

	meandf.to_excel(writer, sheet_name="mean_sem")


	writer.save()



if __name__ == '__main__':
	graph_profile()
	#main()
