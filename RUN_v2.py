#!/usr/bin/env python3
#-*- coding: utf-8 -*-

'''
-------------------------------------------------------------------------------
Copyright© 2021 Jules Devaux / Lyvia Bowering / Tristan McArley. All Rights Reserved
Open Source script under Apache License 2.0
-------------------------------------------------------------------------------
'''

import os
current_path = os.path.dirname(os.path.abspath(__file__))


#========= PARAMETERS =========
TEMPERATURES=[21,25]# 
CHAMBER_VOLUME = 2 #ml
WINDOW = 20 #Moving averages
MAX_kPA = 24 #kPa
GRAPHING = False
_saving = True # Save summary file and stats
DATA_FOLDER = f'{current_path}/CSV/' # Folder containing all csvs
SUBFOLDER_NAME=None

#=============================


import sys, pickle, datetime
import numpy as np
import pandas as pd
import pingouin as pg
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt


def hill(S, Vmax, Km, n=1):
	S=np.abs(S)
	return (Vmax*S**n)/(Km+S**n)


def fit_hill_curve(x, y):
	'''
	x_y is df with x as index, y as values
	'''
	Vmax = y.max()
	xmax, xmin = x[-1], x[0]
	dx = x[-1]-x[0]

	# parameters as [max, min, dt, ]
	param_initial = [Vmax, 0.5*dx, 1]
	param_bounds = ([Vmax-0.5*Vmax, xmin*0.1, 0.01],
					[Vmax+0.5*Vmax, xmax*10, 100])
	bounds = [(Vmax-0.5*Vmax, Vmax+0.5*Vmax),
			(xmin, xmax),
			(0.01, 100)]


	popt, _ = curve_fit(
		hill, x, y,
		p0 = param_initial,
		bounds=param_bounds)

	Vmax, pP50, _hill =popt[0], popt[1], popt[2]
	y_fit = hill(x, Vmax, pP50, _hill)


	# Goodness of fit:
	# residual sum of squares
	ss_res = np.sum((y - y_fit) ** 2)
	ss_tot = np.sum((y - np.mean(y)) ** 2)
	r2 = 1 - (ss_res / ss_tot)

	predicted=pd.Series(y_fit, name='predicted')

	return Vmax, pP50, _hill, predicted, r2


def extract_csvs():

	# All chambers saved as dict
	chambers = []

	# Get subfolders
	folders=[x[1] for x in os.walk(DATA_FOLDER)][0]
	for folder in folders:
		temperature=int(folder.split('_')[0][:-1])
		_group=folder.split('_')[1]
		csvs=[file for file in os.listdir(f"{DATA_FOLDER}/{folder}/") if '.csv' in file]
		for csv in csvs:
			try:
				print(f'Exctracting {csv}...')
				if any(('HEART' in csv, 'heart' in csv)): tissue='heart'
				else: tissue ='red_muscle'

				df=pd.read_csv(f"{DATA_FOLDER}/{folder}/{csv}", low_memory=False, encoding= 'unicode_escape')

				# Retrieve and select columns for that specific chamber
				for c in df.columns:
					if f": O2 pres" in c:
						kPa=True
						PO2_col=[c for c in df.columns if f": O2 pres" in c][0]

					elif f": O2 conc" in c:
						kPa=False
						PO2_col=[c for c in df.columns if f": O2 conc" in c][0]

				
				JO2_col=[c for c in df.columns if f": O2 flux" in c][0]
				try:DYm_col=[c for c in df.columns if f": Amp [" in c][0]
				except: DYm_col=[c for c in df.columns if f": pX [" in c][0]

			
				# Create DataFrame for that chamber
				cdf=df.loc[:,[PO2_col,JO2_col,DYm_col]].dropna()
				cdf=cdf.rename(columns={PO2_col:'PO2',
										JO2_col:'JO2',
										DYm_col:'DYm'})

				# Normalise to 0
				cdf['PO2']=cdf['PO2']-cdf['PO2'].min()
				cdf['JO2']=cdf['JO2']-cdf['JO2'].min()

				chambers.append({
					'filename':csv,
					'temperature': temperature,
					'Group': _group,
					'tissue': tissue,
					'df': cdf,
					})
			except Exception as e: print(f"{csv}: ERROR: {e}")

	return chambers



def fundamentals(chambers, _saving=True):

	_dicts=[] # Main list to create the final df
	for chamber in chambers:

		_dicts.append({
			'filename':f"{chamber['filename']}",
			'PO2_max':chamber['df']['PO2'].max(),
			'PO2_min':chamber['df']['PO2'].min(),
			'JO2_max':chamber['df']['JO2'].max(),
			'JO2_min':chamber['df']['JO2'].min(),
			'dt':len(chamber['df'])
		})

	df=pd.DataFrame(_dicts)
	summary=df.agg(['mean','sem'])
	if _saving:
		df.to_csv('fundamentals.csv')
		print('saved fundamentals')

	return df


def process_all_chambers(chambers):
	'''
	index as time, columns = JO2, PO2, DYm
	'''

		
	_dicts=[] #temprary list to avoid overwriting chambers list
	for chamber in chambers:
		try:

			print(f"Processing {chamber['filename']}")

			

			# GRAPH
			# #plt.scatter(X, predicted, label='pred')
			# plt.scatter(chamber['df'].index,chamber['df']['JO2'].values, label='expe')
			# plt.title(chamber['filename'])
			# plt.show()

			# ============= JO2 PARAMETERS AND FITTED HILL CURVE ============
			# Observed P50 from raw JO2
			try:
				JO2_max=chamber['df']['JO2'].max()
				JO2_50=JO2_max/2
				_portion=chamber['df'].loc[chamber['df']['JO2']<(JO2_50)]
				oP50=_portion['PO2'].iloc[0]
			except Exception as e:
				print(f"ERROR: observed_P50: {e}")
				oP50=float('nan')

			Kcat=JO2_max/oP50

			# need to have df with PO2 index and JO2 column
			tdf=chamber['df'].loc[:,['PO2','JO2']]
			tdf=tdf.drop_duplicates(subset='PO2', keep='first').set_index('PO2').sort_index()

			# Fit curve to experimental PO2 
			fJO2max, fP50, _hill, fitted, r2 = fit_hill_curve(tdf.index.values, tdf['JO2'].values)

			# Extend curve to 0 -> 100% PO2
			# Standardise PO2 for comaprison between samples
			X=np.arange(0, MAX_kPA, 0.01)
			predicted=hill(X, fJO2max, fP50, _hill)
			predicted=pd.DataFrame(predicted, index=X, columns=['JO2'])

			# Calculate area under the curve from Standardised
			for i in [21, 10, 5, 2.5, 1.25, 0.5]:
				_trimmed_df=predicted.loc[predicted.index<i]
				auc=np.trapz(_trimmed_df['JO2'], _trimmed_df.index, 0.1)
				chamber.update({f'area under the curve {i}kPa':auc})

			# JO2max at 100% saturation 
			kPa21_JO2max=predicted['JO2'].max()

			# Extract P50 from fitted JO2
			try:
				JO2_50=kPa21_JO2max/2
				_portion=predicted.loc[predicted['JO2']<JO2_50]
				cP50=_portion.index.values[-1]
			except Exception as e:
				print(f"ERROR: calculated_P50: {e}")
				cP50=float('nan')


			# Calculate scope and exctract some


			# ============= MEMBRANE POTENTIAL ==========
			



			# Update parameters for chamber
			chamber.update(
				{'observed_P50':oP50,
				'fitted_JO2max': fJO2max,
				'fitted_P50':fP50,
				'fitted_hill':_hill,
				'JO2max_at_21kPa':kPa21_JO2max,
				'calc_P50':cP50,
				'Kcat':Kcat,
				'goodness of fit':r2,
				'predicted':predicted
				})

			_dicts.append(chamber)

			# GRAPH
			if GRAPHING is True:
				plt.scatter(predicted.index, predicted['JO2'], label='pred', c='blue')
				plt.scatter(tdf.index, tdf['JO2'].values, label='expe', c='red')
				plt.title(f"{chamber['filename']} - {chamber['state']}")
				plt.show()

		except Exception as e:print(e)

	return _dicts


def create_summary(chambers):

	# Check outliers
	outliers=[]
	for chamber in chambers:
		if float(chamber['goodness of fit'])<=0.5:
			outliers.append({
				'filename':chamber['filename'],
				'chamber': chamber['chamber']
				})

		print("============= OUTLIERS ==============")
		for o in outliers:
			print(f"{o['filename']} - chamber {o['chamber']}")
		print("\n\n")

		# Create summary table
		summary=[]
		for i in range(len(chambers)):
			row={k:v for k,v in chambers[i].items() if k not in ['df','PO2','predicted','JO2','JO2_calc','DYm']}

			row=pd.DataFrame(row, index=[i])
			summary.append(row)
		summary=pd.concat(summary)

		# Save summary
		if _saving is True:
			summary.to_csv('summary.csv')

		return summary


def do_stats(df):
	
	# Create Excel template
	_now=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
	fileName=f'Summary_{_now}.xlsx'
	writer=pd.ExcelWriter(fileName, engine='xlsxwriter')

	# First, add data to first tab
	df.to_excel(writer, sheet_name='main') 


	between=['temperature', 'tissue', 'Group']
	parameters=[c for c in df.columns if df[c].dtypes != 'object' and c not in between]


	#===== Do Averages and SEM
	summary=df.groupby(between).agg(['mean','sem'])
	summary.to_excel(writer, sheet_name='averages') 


	#===== Do two-Way ANOVA ====
	aov_row=0 #Keep track of rows to append ANOVA tests
	hoc_row=0

	# Loop through parameter for ANOVA
	for parameter in parameters:
		aov=pg.anova(df, dv=parameter,
				between=between,
				detailed=True)
		aov['Parameter']=parameter

		# Append to Excel, spacing of 3 rows
		aov.to_excel(writer, sheet_name="ANOVAs", startrow=aov_row , startcol=0)   
		aov_row+=(len(aov.index)+3)

		# ==== POST-HOCs
		groups=[{'groups': ['temperature', 'Group'], 'between': 'tissue'},
				{'groups': ['temperature', 'tissue'], 'between': 'Group'},
				{'groups': ['tissue', 'Group'], 'between': 'temperature'}]
		for g in groups:
			for group, gdf in df.groupby(g['groups']):
				#print(gdf[parameter])
				# Do the test
				try:
					ph=pg.pairwise_tukey(data=gdf, dv=parameter, between=g['between'])
					ph['Parameter']=parameter
					ph['_group']=group
			
					# Append to Excel
					ph.to_excel(writer, sheet_name="PostHocs", startrow=hoc_row , startcol=0)   
					hoc_row+=(len(ph.index)+3)
				except Exception as e:print(e)


	writer.save()
	print(f'Saved {fileName}')


def sandbox():
	extract_csvs()


def main():
	chambers=extract_csvs()
	for chamber in chambers: print(chamber['filename'])
	_fundamentals=fundamentals(chambers)
	# chambers=cleanup(chambers, _fundamentals)
	chambers=process_all_chambers(chambers)
	with open('chambers.pkl','wb') as f:
		pickle.dump(chambers,f)
	df=create_summary(chambers)
	df=pd.read_csv('summary.csv', header=0).drop(columns='Unnamed: 0')
	do_stats(df)



if __name__=='__main__':
	main()