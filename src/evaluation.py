import pandas as pd
import pickle

# read in simulation data
sim_filepath = '../results/results_continuous reading.pkl'
with open(sim_filepath, 'rb') as infile:
    simulation_results = pickle.load(infile)
# read in words that have been skipped and never fixated (they are not included in fixation data,
# but should be considered for wordskip results
skipped_filepath = '../results/skipped_words_continuous reading.pkl'
with open(skipped_filepath, 'rb') as infile:
    skipped_words = pickle.load(infile)

# --------- Get data frame with all tokens ---------
# data frame with columns = keys from dict of each fixation and rows = each fixation instance
df_sim_results = pd.DataFrame.from_dict(simulation_results, orient='index')
# add skipped words
df_skipped_words = pd.DataFrame.from_records(skipped_words)
df_skipped_words['wordskip'] = True
df_all_tokens = pd.concat([df_sim_results,df_skipped_words], axis=0)
df_all_tokens.fillna({'wordskip': False, 'fixation duration': 0.0}, inplace=True)
df_all_tokens = df_all_tokens.set_index('foveal word index')
df_all_tokens = df_all_tokens.sort_index()

# Distribution

# --------- General fixation duration measures  ---------
# saccade type counts
saccade_type_counts = df_sim_results['saccade type'].value_counts()
# total viewing time for each token
sum_fixation_duration = df_all_tokens.groupby(['foveal word index'])[['fixation duration']].sum()
# first pass
first_pass_indices = []
for word, fix_hist in df_sim_results.groupby(['foveal word index']):
  for i, fix in fix_hist.iterrows():
    if 'regression' not in fix_hist['saccade type'].tolist():
      first_pass_indices.append(i)
    elif i < fix_hist['saccade type'].tolist().index('regression'):
      first_pass_indices.append(i)
first_pass = df_sim_results.filter(items=first_pass_indices,axis=0)
# first fixation
first_fixation = first_pass.loc[first_pass.groupby(['foveal word index']).apply(lambda x: x.index[0]).values,:][['foveal word','foveal word index','fixation duration']]
# refixations
refixations = first_pass.groupby(['foveal word index']).filter(lambda x:len(x)>1)
# second fixation
second_fixations = first_pass.loc[refixations.groupby(['foveal word index']).apply(lambda x: x.index[1]).values,:][['foveal word','foveal word index','fixation duration']]
# gaze duration
gaze_duration = first_pass.groupby(['foveal word index'])[['fixation duration']].sum()
