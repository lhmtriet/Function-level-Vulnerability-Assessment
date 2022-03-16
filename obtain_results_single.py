import pandas as pd
import glob
import numpy as np
from numpy import mean
from numpy import var
from math import sqrt
import scipy
from scipy.stats import mannwhitneyu, ttest_ind, wilcoxon

from cliffs_delta import cliffs_delta

########################################
def compute_average_results(result_csv, group_cols=['problem', 'granularity', 'scope', 'feature', 'classifier', 'parameters']):
	avg_df = result_csv.copy()

	avg_df = avg_df.groupby(group_cols).mean().reset_index()

	return avg_df

########################################
def find_best_results(avg_df, group_cols=['problem', 'granularity', 'scope', 'feature']):

	avg_df = compute_average_results(avg_df, group_cols=['problem', 'granularity', 'scope', 'feature', 'classifier', 'parameters'])

	return avg_df.loc[avg_df.groupby(group_cols)['test_mcc'].idxmax()].reset_index(drop=True)


# function to calculate Cohen's d for independent samples
def cohend(d1, d2):
	# calculate the size of samples
	n1, n2 = len(d1), len(d2)
	# calculate the variance of the samples
	s1, s2 = var(d1, ddof=1), var(d2, ddof=1)
	# calculate the pooled standard deviation
	s = sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
	# calculate the means of the samples
	u1, u2 = mean(d1), mean(d2)
	# calculate the effect size
	val = abs(u1 - u2) / s
	res = 'negligible'

	if 0.2 < val <= 0.5:
		res = 'small'
	elif 0.5 < val <= 0.8:
		res = 'medium'
	elif val > 0.8:
		res = 'large'

	return val, res


def compute_test(a, b):

	test_res = wilcoxon(a, b, alternative='greater', correction=True)
	p_value = test_res.pvalue
	z = abs(scipy.stats.norm.ppf(p_value/2))

	val = z / sqrt(len(a) + len(b))
	res = 'negligible'
	if 0.1 < val <= 0.3:
		res = 'small'
	elif 0.3 < val <= 0.5:
		res = 'medium'
	elif val > 0.5:
		res = 'large'

	return (z, p_value), (val, res)

def test_significance(df):

	df = compute_average_results(df,
								 group_cols=['problem', 'granularity', 'scope', 'classifier', 'feature'])

	print(len(df))

	problems = df['problem'].unique()
	granularities = df['granularity'].unique()
	scopes = df['scope'].unique()
	classifiers = df['classifier'].unique()
	# features = df['feature'].unique()

	# df['p_value'] = 100
	# df['effect_size'] = 100

	metric_col = 'test_mcc'
	base_scope = 'whole'

	scope_res = {}

	# for g in granularities:
	for g in ['method']:

		cur_df = df[df['granularity'] == g].copy()
		print(len(cur_df))

		for s in scopes:
			scope_res[s] = cur_df[cur_df['scope'] == s][metric_col].values

		# for i in range(len(scopes)):
		for s1 in ['lines_without_context', 'lines_with_all_context', 'lines_with_surrounding_context', 'whole']:
			for j in range(len(scopes)):
				# s1 = scopes[i]

				s2 = scopes[j]

				if s2 == s1:
					continue

				if s1 == 'lines_with_all_context':
					if s2 != 'lines_without_context' and s2 != 'slicing_only':
						continue
				elif s1 == 'lines_with_surrounding_context':
					if s2 != 'lines_without_context' and s2 != 'surrounding_only':
						continue
				elif s1 == 'lines_without_context':
					if s2 != 'non_vuln' and s2 != 'context_only' and s2 != 'random_context':
						continue
				
				if len(scope_res[s1]) > 0 and len(scope_res[s2]) > 0:

					s1_res = scope_res[s1]
					s2_res = scope_res[s2]

					p_val, effect_sz = compute_test(s1_res, s2_res)
					print(g, s1, s2, p_val, effect_sz, np.mean(s1_res), np.mean(s2_res))
					print('#' * 30)

			print('\n' * 2)


def test_significance_model(df):

	df = compute_average_results(df, group_cols=['problem', 'granularity', 'scope', 'feature', 'classifier'])

	classifiers = df['classifier'].unique()
	features = df['feature'].unique()

	metric_col = 'test_mcc'
	base_scope = 'whole'

	res = {}

	print('Testing classifiers')

	for g in ['method']:

		cur_df = df[df['granularity'] == g].copy()
		print(len(cur_df))

		for s in classifiers:
			res[s] = cur_df[cur_df['classifier'] == s][metric_col].values

		for i in range(len(classifiers)):
			for j in range(len(classifiers)):
				s1 = classifiers[i]
				s2 = classifiers[j]

				if s1 == s2:
					continue

				if len(s1) > 0 and len(s2) > 0:
					s1_res = res[s1]
					s2_res = res[s2]

					p_val, effect_sz = compute_test(s1_res, s2_res)
					
					print(g, s1, s2, p_val, effect_sz, np.mean(s1_res), np.mean(s2_res))
					print('#' * 30)

			print('\n' * 2)

	res = {}

	print('Testing features')

	for g in ['method']:

		cur_df = df[df['granularity'] == g].copy()
		print(len(cur_df))

		for s in features:
			res[s] = cur_df[cur_df['feature'] == s][metric_col].values

		for i in range(len(features)):
			for j in range(len(features)):
				s1 = features[i]
				s2 = features[j]

				if s1 == s2:
					continue

				if len(s1) > 0 and len(s2) > 0:
					s1_res = res[s1]
					s2_res = res[s2]

					p_val, effect_sz = compute_test(s1_res, s2_res)

					print(g, s1, s2, p_val, effect_sz, np.mean(s1_res), np.mean(s2_res))
					print('#' * 30)

			print('\n' * 2)

#################################

result_folder = 'Code/single/ml_results_single/'

all_results = pd.concat([pd.read_csv(file) for file in glob.glob(f'{result_folder}results_*.csv')])

avg_df = compute_average_results(all_results, group_cols=['granularity', 'scope', 'problem'])
avg_df.to_csv(f'{result_folder}avg_results_single.csv', index=False)

test_significance(all_results)
test_significance_model(all_results)

best_results = find_best_results(all_results, group_cols=['granularity', 'scope', 'problem'])
best_results.to_csv(f'{result_folder}best_results_single.csv', index=False)
