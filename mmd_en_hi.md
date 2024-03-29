## Install dependencies from https://github.com/babangain/english_hindi_translation

## English-Hindi version of MMD Data
### Lowercase and Apply BPE
```
MOSES_DIR=mosesdecoder
FASTBPE_DIR=fastBPE

DATA_FOLDER_NAME=mmd_en_hi
DATA_DIR=data/$DATA_FOLDER_NAME
mkdir -p $DATA_DIR
cp en_hi/bpecode $DATA_DIR/bpecode
cp en_hi/vocab.en $DATA_DIR/vocab.en

for SUBSET in train test valid
do
  for LANG in en hi
  do
    cat $DATA_DIR/$SUBSET.$LANG | $MOSES_DIR/scripts/tokenizer/lowercase.perl> $DATA_DIR/$SUBSET.lc.$LANG
    $FASTBPE_DIR/fast applybpe $DATA_DIR/$SUBSET.bpe.$LANG $DATA_DIR/$SUBSET.lc.$LANG $DATA_DIR/bpecode
  done
done

```

## Binarize the data for faster training
```
BINARY_DATA_DIR=data_bin/$DATA_FOLDER_NAME
mkdir -p $BINARY_DATA_DIR
fairseq-preprocess \
    --source-lang en --target-lang hi \
    --joined-dictionary \
    --srcdict $DATA_DIR/vocab.en \
    --trainpref $DATA_DIR/train.bpe \
    --validpref $DATA_DIR/valid.bpe \
    --testpref $DATA_DIR/test.bpe \
    --destdir $BINARY_DATA_DIR \
    --workers 20
```

## Training
```
MODEL_DIR=models/$DATA_FOLDER_NAME
mkdir -p $MODEL_DIR
export CUDA_VISIBLE_DEVICES=4
nohup fairseq-train $BINARY_DATA_DIR --fp16 \
    --source-lang en --target-lang hi \
    --arch transformer --log-interval  1  --log-format simple \
    --dropout 0.2 --weight-decay 0.0 \
    --share-all-embeddings \
    --ddp-backend=no_c10d \
    --lr 0.0005 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --seed 42 \
    --lr-scheduler inverse_sqrt \
    --max-tokens 4000 --update-freq 4 \
    --max-update 5000 \
    --save-interval 1 \
    --patience 5 \
    --finetune-from-model models/samanantar/checkpoint_last.pt \
    --save-dir $MODEL_DIR &
```

## Generate From Baseline model
```
OUTFILENAME=$DATA_DIR/result_baseline
fairseq-generate $BINARY_DATA_DIR --batch-size 32 --path models/samanantar/checkpoint_last.pt  --remove-bpe \
--beam 5 --source-lang en --target-lang hi --task translation >  $OUTFILENAME.txt

cat $OUTFILENAME.txt |grep ^H | sort -nr -k1.2 | cut -f3- | $MOSES_DIR/scripts/tokenizer/detokenizer.perl > $OUTFILENAME.hi 
cat $OUTFILENAME.hi | sacrebleu $DATA_DIR/test.hi  -m bleu ter

python scripts/user_divide.py $DATA_DIR test.hi test.speaker.txt $OUTFILENAME.hi system
cat $OUTFILENAME.hi.system | sacrebleu $DATA_DIR/system.hi -m bleu ter
```

## Generate From Finetuned model
```
OUTFILENAME=$DATA_DIR/result_finetune
fairseq-generate $BINARY_DATA_DIR --batch-size 32 --path $MODEL_DIR/checkpoint_best.pt  --remove-bpe \
--beam 5 --source-lang en --target-lang hi --task translation >  $OUTFILENAME.txt

cat $OUTFILENAME.txt |grep ^H | sort -nr -k1.2 | cut -f3- | $MOSES_DIR/scripts/tokenizer/detokenizer.perl > $OUTFILENAME.hi 
cat $OUTFILENAME.hi | sacrebleu $DATA_DIR/test.hi  -m bleu ter
python scripts/user_divide.py $DATA_DIR test.hi test.speaker.txt $OUTFILENAME.hi system
cat $OUTFILENAME.hi.system | sacrebleu $DATA_DIR/system.hi -m bleu ter
```

## Training with WMT20 Chat Model
```
SUFFIX=xfer_from_chat
MODEL_DIR=models/$DATA_FOLDER_NAME.$SUFFIX
mkdir -p $MODEL_DIR
export CUDA_VISIBLE_DEVICES=4
nohup fairseq-train $BINARY_DATA_DIR --fp16 \
    --source-lang en --target-lang hi \
    --arch transformer --log-interval  1  --log-format simple \
    --dropout 0.2 --weight-decay 0.0 \
    --share-all-embeddings \
    --ddp-backend=no_c10d \
    --lr 0.0005 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --seed 42 \
    --lr-scheduler inverse_sqrt \
    --max-tokens 4000 --update-freq 4 \
    --max-update 5000 \
    --save-interval 1 \
    --patience 5 \
    --finetune-from-model models/wmt20_chat_en_hi/checkpoint_best.pt \
    --save-dir $MODEL_DIR &
```

## Generate From Transfer Learning Model
```
OUTFILENAME=$DATA_DIR/result_$SUFFIX
fairseq-generate $BINARY_DATA_DIR --batch-size 32 --path $MODEL_DIR/checkpoint_best.pt  --remove-bpe \
--beam 5 --source-lang en --target-lang hi --task translation >  $OUTFILENAME.txt

cat $OUTFILENAME.txt |grep ^H | sort -nr -k1.2 | cut -f3- | $MOSES_DIR/scripts/tokenizer/detokenizer.perl > $OUTFILENAME.hi 
cat $OUTFILENAME.hi | sacrebleu $DATA_DIR/test.hi  -m bleu ter

python scripts/user_divide.py $DATA_DIR test.hi test.speaker.txt $OUTFILENAME.hi system
cat $OUTFILENAME.hi.system | sacrebleu $DATA_DIR/system.hi -m bleu ter
```
## Context Based Model
```
python scripts/context_gen_mmd.py
DATA_FOLDER_NAME=mmd_en_hi_context
DATA_DIR=data/$DATA_FOLDER_NAME
mkdir -p $DATA_DIR
cp en_hi/bpecode $DATA_DIR/bpecode
cp en_hi/vocab.en $DATA_DIR/vocab.en

for SUBSET in train test valid
do
  for LANG in en hi
  do
    cat $DATA_DIR/$SUBSET.$LANG | $MOSES_DIR/scripts/tokenizer/lowercase.perl> $DATA_DIR/$SUBSET.lc.$LANG
    $FASTBPE_DIR/fast applybpe $DATA_DIR/$SUBSET.bpe.$LANG $DATA_DIR/$SUBSET.lc.$LANG $DATA_DIR/bpecode
  done
done

```

## Binarize the data for faster training
```
BINARY_DATA_DIR=data_bin/$DATA_FOLDER_NAME
mkdir -p $BINARY_DATA_DIR
fairseq-preprocess \
    --source-lang en --target-lang hi \
    --joined-dictionary \
    --srcdict $DATA_DIR/vocab.en \
    --trainpref $DATA_DIR/train.bpe \
    --validpref $DATA_DIR/valid.bpe \
    --testpref $DATA_DIR/test.bpe \
    --destdir $BINARY_DATA_DIR \
    --workers 20
```

## Training
```
MODEL_DIR=models/$DATA_FOLDER_NAME
mkdir -p $MODEL_DIR
export CUDA_VISIBLE_DEVICES=4
nohup fairseq-train $BINARY_DATA_DIR --fp16 \
    --source-lang en --target-lang hi \
    --arch transformer --log-interval  1  --log-format simple \
    --dropout 0.2 --weight-decay 0.0 \
    --share-all-embeddings \
    --ddp-backend=no_c10d \
    --lr 0.0005 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --seed 42 \
    --lr-scheduler inverse_sqrt \
    --max-tokens 4000 --update-freq 4 \
    --max-update 5000 \
    --save-interval 1 \
    --patience 5 \
    --finetune-from-model models/samanantar/checkpoint_last.pt \
    --save-dir $MODEL_DIR &
```
## Generate
```
OUTFILENAME=$DATA_DIR/result_context
fairseq-generate $BINARY_DATA_DIR --batch-size 32 --path $MODEL_DIR/checkpoint_best.pt  --remove-bpe \
--beam 5 --source-lang en --target-lang hi --task translation >  $OUTFILENAME.txt

cat $OUTFILENAME.txt |grep ^H | sort -nr -k1.2 | cut -f3- | $MOSES_DIR/scripts/tokenizer/detokenizer.perl > $OUTFILENAME.hi 
cat $OUTFILENAME.hi | sacrebleu $DATA_DIR/test.hi  -m bleu ter
python scripts/user_divide.py $DATA_DIR test.hi test.speaker.txt $OUTFILENAME.hi system
cat $OUTFILENAME.hi.system | sacrebleu $DATA_DIR/system.hi -m bleu ter
```
## Extract users from test set
```
cat data/mmd_csv/test.csv | awk -F"\t" '{print $3}'
```
