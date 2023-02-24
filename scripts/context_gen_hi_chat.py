import pandas as pd
import os
def context(DATASET,speaker):
  if speaker=='customer':
    DATASET[['target','source']] =DATASET[['source','target']]
  TMP = DATASET.copy()
  #print(TMP.head())


  #print(TMP.iloc[910])

  source = []
  target = []
  lst = []
  id = DATASET.iloc[0]['id']
  for i in range(1,len(DATASET)):
    if (DATASET.iloc[i]['speaker'] == DATASET.iloc[i-1]['speaker']) and (DATASET.iloc[i]['id'] == DATASET.iloc[i-1]['id']) :
      if (DATASET.iloc[i]['speaker'] ==speaker) and (DATASET.iloc[i]['id'] not in [ 'dlg-beecf8db-5947-4621-a871-d9bd9e281efa', 'dlg-47f64bcb-cd1c-47e8-9116-7f998ff5ff47']):  # Exception for conversation with only one speaker
        j = i-1
        lst = []
        while j >= 0:
          if DATASET.iloc[j]['speaker'] == DATASET.iloc[i]['speaker'] and len(lst) < 3:
            lst.append(DATASET.iloc[j]['source'])
            j -= 1
          else:
            break


    s = ""
        
    if len(lst) > 0:
      s = "<context> "
      s =s+ " ".join(lst[::-1])
      s = s + " <end> "
      s= s + DATASET.iloc[i]['source']
      lst = []
    else:
      s= DATASET.iloc[i]['source']
    TMP.iloc[i, TMP.columns.get_loc('source')] = s
  return TMP

def process_dataset(input_filename,output_prefix,target_lang):
  data = pd.read_csv(input_filename, sep="\t", names=['id','utteranceID','speaker','source','target'])
  #print(data.iloc[910])
  print("Please wait.... This may take a while....")
  data_agent= context(data.copy(), 'agent')
  data_customer= context(data.copy(), 'customer')
  print("Processed Agent subset into intermediate state ")

  data_only_agent = data_agent #[data_agent.speaker == 'agent']
  data_only_customer = data_customer # [data_customer.speaker == 'customer']
  #print(data_only_agent.iloc[910])

  if target_lang =='hi':
    with open(output_prefix+'.en', 'w') as f:
      f.write( data_only_agent['source'].str.cat(sep='\n'))
    lst = list(data_only_agent['target'])
    with open(output_prefix+'.hi', 'w') as f:
      f.write( data_only_agent['target'].str.cat(sep='\n'))
  elif target_lang =='en':
    with open(output_prefix+'.en', 'w') as f:
      f.write( data_only_customer['target'].str.cat(sep='\n'))
    with open(output_prefix+'.hi', 'w') as f:
      f.write( data_only_customer['source'].str.cat(sep='\n'))
  else:
    print("Invalid target lang")


print("Processing English-Hindi subset")
if not os.path.isdir('data/wmt20_en_hi_context'):
  os.mkdir('data/wmt20_en_hi_context')
process_dataset('data/wmt20_csv/test.csv','data/wmt20_en_hi_context/test','hi')
process_dataset('data/wmt20_csv/valid.csv','data/wmt20_en_hi_context/valid','hi')
process_dataset('data/wmt20_csv/train.csv','data/wmt20_en_hi_context/train','hi')

print("Processing Hindi-English subset")
if not os.path.isdir('data/wmt20_hi_en_context'):
  os.mkdir('data/wmt20_hi_en_context')
process_dataset('data/wmt20_csv/test.csv','data/wmt20_hi_en_context/test','en')
process_dataset('data/wmt20_csv/valid.csv','data/wmt20_hi_en_context/valid','en')
process_dataset('data/wmt20_csv/train.csv','data/wmt20_hi_en_context/train','en')