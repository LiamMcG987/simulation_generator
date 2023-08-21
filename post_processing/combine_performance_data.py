
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns


# Methods

def extract_data(sub_files):
	""" Extract force-displacement data. """

	key_data = []
	for sub_file in sub_files:  # Slice using df.xs('location1',level='loc',axis=1)
		temp_df = pd.read_csv(sub_file)
		key_data.append(temp_df)
	df = pd.concat(key_data, axis=1)
	header = pd.MultiIndex.from_product([
		sub_files,
		['U', 'RF']],
		names=['file', 'data'])
	df.columns = header
	return df


def extract_metric(sub_files):
	""" Extract key performance metric. """

	key_data = []
	for sub_file in sub_files:
		temp_df = pd.read_csv(sub_file)
		key_data.append(float(temp_df.columns[0]))
	df = pd.DataFrame(columns=sub_files)
	df.loc[0] = key_data
	return df


def plot_data(ax, df, sim_type, sub_file_names):
	""" Plot force-displacement data. """

	prop_cycle = plt.rcParams['axes.prop_cycle']
	colors = prop_cycle.by_key()['color']
	if not sim_type == 'bulge':
		x_data = df.xs('U', level='data', axis=1).values
		y_data = df.xs('RF', level='data', axis=1).values
		ax.plot(x_data, y_data)
		ax.set_title('{}'.format(sim_type))
		ax.set_xlabel('Displacement (mm)')
		ax.set_ylabel('Force (N)')
		ax.legend(sub_file_names, loc='lower right', frameon=False)
	else:
		x_data = sub_file_names
		y_data = df.values[0]
		for i in range(len(x_data)):
			ax.bar(x_data[i], y_data[i], color=colors[i])
		ax.set_title('{}'.format(sim_type))
		ax.set_ylabel('Deflection (mm)')
		for label in ax.get_xticklabels():
			label.set_rotation(30)
			label.set_ha('right')


# Main

def main():
	""" Extracts performance data and combines into spreadsheet and plots. """

	# Initialise Excel writer
	writer = pd.ExcelWriter('performance_data.xlsx')
	sim_types = ['top_load', 'side_load', 'squeeze', 'bulge']
	fig, axes = plt.subplots(1, len(sim_types), figsize=(20, 7))
	# Read files
	files = glob.glob('*.csv')
	with writer as f:
		for idx, sim_type in enumerate(sim_types):
			sub_files = [s for s in files if sim_type in s.lower()]
			# Clean file names
			sub_file_names = [s.lower().removesuffix('_{}_disp_force.csv'.format(sim_type)) for s in sub_files]
			if len(sub_file_names) > 0:
				# Extract data
				if not sim_type == 'bulge':
					df = extract_data(sub_files)
				else:
					df = extract_metric(sub_files)
				# Export and plot data
				df.to_excel(f, sheet_name=sim_type)
				plot_data(axes[idx], df, sim_type, sub_file_names)
		plt.tight_layout()
		plt.savefig('performance_plot.png', dpi=300)


if __name__ == "__main__":
	main()
