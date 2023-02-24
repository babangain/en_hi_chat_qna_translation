import pandas as pd
from tqdm import tqdm
import os
def context(DATASET,speaker):
  if speaker=='user':
    DATASET[['target','source']] =DATASET[['source','target']]
  TMP = DATASET.copy()
  #print(TMP.head())


  print(TMP.iloc[1])

  source = []
  target = []
  lst = []
  id = DATASET.iloc[0]['id']
  for i in tqdm(range(1,len(DATASET))):
    if (DATASET.iloc[i]['speaker'] == DATASET.iloc[i-1]['speaker']) and (DATASET.iloc[i]['id'] == DATASET.iloc[i-1]['id']) :
      if (DATASET.iloc[i]['speaker'] ==speaker) and DATASET.iloc[i]['id'] != 'dlg-beecf8db-5947-4621-a871-d9bd9e281efa':  # Exception for conversation with only one speaker
        j = i-1
        lst = []
        while j >= 0:
          if DATASET.iloc[j]['speaker'] == DATASET.iloc[i]['speaker'] and len(lst) < 3:
            lst.append(str(DATASET.iloc[j]['source']))
            j -= 1
          else:
            break


    s = ""
        
    if len(lst) > 0:
      s = "<context> "
      s =s+ " ".join(lst[::-1])
      s = s + " <end> "
      s= s + str(DATASET.iloc[i]['source'])
      lst = []
    else:
      s= DATASET.iloc[i]['source']
    TMP.iloc[i, TMP.columns.get_loc('source')] = s
  return TMP

def process_dataset(input_filename,output_prefix,target_lang):
  data = pd.read_csv(input_filename, sep="\t", names=['id','speaker','source','target'])
    
  print("Please wait.... This may take a while....")
  data_system= context(data.copy(), 'system')
  data_user= context(data.copy(), 'user')

  data_only_system = data_system #[data_system.speaker == 'system']
  data_only_user = data_user #[data_user.speaker == 'user']

  if target_lang =='hi':
    with open(output_prefix+'.en', 'w') as f:
      f.write( data_only_system['source'].str.cat(sep='\n'))
    with open(output_prefix+'.hi', 'w') as f:
      f.write( data_only_system['target'].str.cat(sep='\n'))
  elif target_lang =='en':
    with open(output_prefix+'.en', 'w') as f:
      f.write( data_only_user['target'].str.cat(sep='\n'))
    with open(output_prefix+'.hi', 'w') as f:
      f.write( data_only_user['source'].str.cat(sep='\n'))
  else:
    print("Invalid target lang")


print("Processing English-Hindi subset")
if not os.path.isdir('data/mmd_en_hi_context'):
  os.mkdir('data/mmd_en_hi_context')
process_dataset('data/mmd_csv/test.csv','data/mmd_en_hi_context/test','hi')
process_dataset('data/mmd_csv/valid.csv','data/mmd_en_hi_context/valid','hi')
process_dataset('data/mmd_csv/train.csv','data/mmd_en_hi_context/train','hi')



print("Processing Hindi-English subset")
if not os.path.isdir('data/mmd_hi_en_context'):
  os.mkdir('data/mmd_hi_en_context')
process_dataset('data/mmd_csv/test.csv','data/mmd_hi_en_context/test','en')
process_dataset('data/mmd_csv/valid.csv','data/mmd_hi_en_context/valid','en')
process_dataset('data/mmd_csv/train.csv','data/mmd_hi_en_context/train','en')
