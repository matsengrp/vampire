#!/bin/bash

while getopts w:i:n:g:e:b:c:s: option
do
    case "${option}" in
        w) WD_PATH=${OPTARG};;
        i) INPUT_FILE=${OPTARG};;
        n) NUM_SCENARIOS=${OPTARG};;
        g) NUM_GEN_SEQUENCES=${OPTARG};;
        e) EVAL_BATCH_NAME=${OPTARG};;
        b) GEN_BATCH_NAME=${OPTARG};;
        c) CHAIN=${OPTARG};;
        s) SPECIES=${OPTARG};;
    esac
done

# Remove files from any previous IGoR runs
rm -rf $WD_PATH/aligns
rm -rf $WD_PATH/$GEN_BATCH_NAME*
rm -rf $WD_PATH/$EVAL_BATCH_NAME*


IGOR_PREFIX="igor -set_wd $WD_PATH"

echo $IGOR_PREFIX -batch $EVAL_BATCH_NAME -read_seqs $INPUT_FILE
$IGOR_PREFIX -batch $EVAL_BATCH_NAME -read_seqs $INPUT_FILE

IGOR_PREFIX="$IGOR_PREFIX -species $SPECIES -chain $CHAIN"

echo $IGOR_PREFIX -batch $EVAL_BATCH_NAME -align --all
$IGOR_PREFIX -batch $EVAL_BATCH_NAME -align --all

echo $IGOR_PREFIX -batch $EVAL_BATCH_NAME -evaluate -output --scenarios $NUM_SCENARIOS
$IGOR_PREFIX -batch $EVAL_BATCH_NAME -evaluate -output --scenarios $NUM_SCENARIOS

echo $IGOR_PREFIX -batch $GEN_BATCH_NAME -generate $NUM_GEN_SEQUENCES --CDR3
$IGOR_PREFIX -batch $GEN_BATCH_NAME -generate $NUM_GEN_SEQUENCES --CDR3
