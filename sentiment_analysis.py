# Import APIs, load data
import nltk
import pandas as pd
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import time
import os
from loading import *

# Clear terminal at the start for easier reading
os.system('cls' if os.name == 'nt' else 'clear')

############################################################
# ###        Calculate Personality Trait Weights:        ###
# ### Conscientiousness, Extraversion, and Agreeableness ###
############################################################

# Import OK Cupid user data
print('Loading and cleaning data')
df = pd.read_csv('users.csv')
# Import test questions
trait_test = pd.read_csv('data-final.csv', delim_whitespace=True) # file was space-seperated
all_cols = trait_test.columns.tolist()
questions = all_cols[0:all_cols.index('OPN10_E')+1]
res_df = trait_test[questions]

# Explore questions, create lists of questions by category

opn_qs = []
csn_qs = []
ext_qs = []
agr_qs = []
est_qs = []
print("Creating traits questions list")
with open('codebook.txt') as question_text:
    for line in question_text.readlines():
        if line[0:3] == 'OPN':
            q = line[line.index('I'):len(line)].rstrip('.\n')
            opn_qs.append(q)
        elif line[0:3] == 'CSN':
            q = line[line.index('I'):len(line)].rstrip('.\n')
            csn_qs.append(q)
        elif line[0:3] == 'EXT':
            q = line[line.index('I'):len(line)].rstrip('.\n')
            ext_qs.append(q)
        elif line[0:3] == 'AGR':
            q = line[line.index('I'):len(line)].rstrip('.\n')
            agr_qs.append(q)
        elif line[0:3] == 'EST':
            q = line[line.index('I'):len(line)].rstrip('.\n')
            est_qs.append(q)

# about every other question is formated as a question of the opposite category (extrovert - introvert), 
# create lists of non category questions

non_opn_qs = []
non_csn_qs = []
non_ext_qs = []
non_agr_qs = []
non_est_qs = []


for i in range(1,5):
    non_opn_qs.append(opn_qs.pop(i))
    non_csn_qs.append(csn_qs.pop(i))
    non_ext_qs.append(ext_qs.pop(i))
non_ext_qs.append(ext_qs.pop(-1))
# the next question sets alternated on a different phase than the previous sets
non_agr_qs.append(agr_qs.pop(0))
non_est_qs.append(est_qs.pop(0))
for i in range(1,5):
    non_agr_qs.append(agr_qs.pop(i))
    non_est_qs.append(est_qs.pop(i))
# there were 3 more non-questions for emotional stability
for i in range(3):
    non_est_qs.append(est_qs.pop(-1))

##################

# Compare sentance similarity using spacy's nlp english-trained model
def sentence_comparer(q_sent, user_sent):
    q_sent = nlp(q_sent)
    try:
        user_sent = nlp(user_sent)
        return q_sent.similarity(user_sent) 
    except TypeError:
        # print("Inputted Text was not a string")
        pass

# Turn essay into list of questions & clean sentence from newline and other characters
def para_to_sents(para):
    from nltk import tokenize
    try:
        sents = tokenize.sent_tokenize(para)
        clean_sents = []
        for sent in sents:
            new_sent = sent.replace('\n',' ').replace('<br',' ').replace('/>',' ').replace('       ','').replace('  ','').replace('\\','')
            clean_sents.append(new_sent)
        return clean_sents
    except UserWarning:
        pass
    except TypeError:
        # print("Inputted text was not a string")
        pass


# Function that will create and visualize personality trait scores of inputted text
import time
# Initialize scores, total sentances to 0
opn = 0
csn = 0
ext = 0
agr = 0
est = 0
non_opn = 0
non_csn = 0
non_ext = 0
non_agr = 0
non_est = 0
total_sents = 0
start = time.time()

print("Loading spacy nlp models")
import spacy
nlp = spacy.load("en_core_web_lg")
print('Finished loading spacy nlp model')

# User Input of text
# para = input('Tell me about yourself; input a paragraph (at least 3 sentences) in order to score emergent personality traits:\n')
# para = para_to_sents(para)
# print(f'your sentences are: {para}')

ess_cols = df.columns.tolist()
user = 100
if os.path.exists('user_responses.txt'):
    os.remove('user_responses.txt')

for es in range(10):
    para = para_to_sents(df[f'essay{es}'][user])
    with open('user_responses.txt','a') as file:
        file.write(f'Essay {es+1}:\n\n')
    try:
        for user_sent in para:
            with open('user_responses.txt','a') as file:
                file.write(user_sent+'\n')
            total_sents += 1
            # Openness
            for q_sent in opn_qs:
                opn += sentence_comparer(q_sent, user_sent)
            for q_sent in non_opn_qs:
                non_opn += sentence_comparer(q_sent, user_sent)
            # Conscientiousness
            for q_sent in csn_qs:
                csn += sentence_comparer(q_sent, user_sent)
            for q_sent in non_csn_qs:
                non_csn += sentence_comparer(q_sent, user_sent)
            # Extroverted
            for q_sent in ext_qs:
                ext += sentence_comparer(q_sent, user_sent)
            for q_sent in non_ext_qs:
                non_ext += sentence_comparer(q_sent, user_sent)
            # Agreeable
            for q_sent in agr_qs:
                agr += sentence_comparer(q_sent, user_sent)
            for q_sent in non_agr_qs:
                non_agr += sentence_comparer(q_sent, user_sent)
            # Emotionally Stable
            for q_sent in est_qs:
                est += sentence_comparer(q_sent, user_sent)
            for q_sent in non_est_qs:
                non_est += sentence_comparer(q_sent, user_sent)
            
            loading(total_sents, f'sentance {total_sents} (essay {es+1}/10)')
 
    except TypeError:
        # print("Inputted text was not a string")
        pass
   
    with open('user_responses.txt','a') as file:
        file.write('\n\n\n')

###################        

# Take average sentiment score over total number of sentances analyzed
opn /= len(opn_qs) * total_sents 
csn /= len(csn_qs) * total_sents 
ext /= len(ext_qs) * total_sents
agr /= len(agr_qs) * total_sents
est /= len(est_qs) * total_sents
non_opn /= len(non_opn_qs) * total_sents
non_csn /= len(non_csn_qs) * total_sents
non_ext /= len(non_ext_qs) * total_sents
non_agr /= len(non_agr_qs) * total_sents
non_est /= len(non_est_qs) * total_sents

end = time.time()
print('Sentance analysis complete')
time.sleep(1)

# Clear terminal at the start for easier reading
os.system('cls' if os.name == 'nt' else 'clear')     
print(f"""Personality Trait Scores (+,-) (ranging from 0.0-1.0):\n
Openness: {round(opn,2)}, {round(non_opn,2)}\n
Conscienciousness: {round(csn,2)}, {round(non_csn,2)}\n
Extrovertedness: {round(ext,2)}, {round(non_ext,2)}\n
Agreeableness: {round(agr,2)}, {round(non_agr,2)}\n
Emotional Stability: {round(est,2)}, {round(non_est,2)}
""")
print('\n')

print(f'Elapsed time: {round(end-start,1)}s (for {total_sents} sentences)')
print(f'See the file \'user_responses.txt\' for the written responses that were analyzed to generate these scores.')

# Create new df with results of sentance analysis
res_df = pd.DataFrame([
['Openness',opn, "+"],['Openness',non_opn,'-'], ['Conscienciousness',csn,'+'],['Conscienciousness',non_csn,'-'], 
    ['Extrovertedness',ext,'+'],['Extrovertedness', non_ext,'-'], ['Agreeableness',agr,'+'],['Agreeableness',non_agr,'-'], 
    ['Emotional Stability',est, '+'],['Emotional Stability',non_est,'-']],columns = ['trait','score','+/-'])

# assign colors based on +/- personality traits
colors = ['blue' if res_df.iloc[row]['+/-']=='+' else 'red' for row in range(len(res_df))]
ax = sns.catplot(data=res_df, kind="bar", x='trait', y="score", hue="+/-", palette=colors)
ax.set_xticklabels(rotation = 45)
ax.set(ylim=(res_df['score'].min()-0.02, res_df['score'].max()+0.02))
ax.set(title = f'Personality Trait Scores for User {user}')
plt.show()
