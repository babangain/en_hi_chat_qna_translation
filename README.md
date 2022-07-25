# Chat & QnA Translation for English-Hindi

## Install dependencies from https://github.com/babangain/english_hindi_translation

## English-Hindi version of WMT20 Chat Data
### Lowercase and Apply BPE
```
DATA_FOLDER_NAME=wmt20_chat_en_hi
DATA_DIR=data/$DATA_FOLDER_NAME
cp en_hi/bpecode $DATA_DIR/bpecode
cp en_hi/vocab.en $DATA_DIR/vocab.en

for SUBSET in train test valid
do
  for LANG in en hi
  do
    cat $DATA_DIR/$SUBSET.$LANG | $MOSES_DIR/scripts/tokenizer/lowercase.perl> $DATA_DIR/$SUBSET.lc.$LANG
    $FASTBPE_DIR/fast applybpe $DATA_DIR/$SUBSET.bpe.$LANG $DATA_DIR/SUBSET.lc.$LANG $DATA_DIR/bpecode
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
    --destdir $BINARY_DATA_DIR \
    --workers 20
```

## Training
```
MODEL_DIR=models/$DATA_FOLDER_NAME
mkdir -p $MODEL_DIR
export CUDA_VISIBLE_DEVICES=5,6
nohup fairseq-train --fp16 \
    $BINARY_DATA_DIR \
    --source-lang en --target-lang hi \
    --arch transformer --log-interval  1  --log-format simple \
    --dropout 0.2 --weight-decay 0.0 \
    --share-all-embeddings \
    --ddp-backend=no_c10d \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --seed 42 \
    --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 5000 --disable-validation --valid-subset train \
    --max-tokens 4000 --update-freq 64  \
    --max-epoch 30 \
    --save-interval 10\
    --save-dir $MODEL_DIR &
```
## Generate 
```
BINARY_DATA_DIR=~/scripts/chat_en_hi/data/data_bin/data/wmt20_chat
OUTFILENAME=wmt20_baseline
fairseq-generate $BINARY_DATA_DIR --batch-size 32 --path models/samanantar/checkpoint_last.pt  --remove-bpe \
--beam 5 --source-lang en --target-lang hi --task translation >  $OUTFILENAME.txt

cat $OUTFILENAME.txt |grep ^H | sort -nr -k1.2 | cut -f3- | $MOSES_DIR/scripts/tokenizer/detokenizer.perl > $OUTFILENAME.hi 

cat $OUTFILENAME.hi | sacrebleu ~/scripts/chat_en_hi/data/data/wmt20_chat/test.hi  -m bleu ter

```

```
