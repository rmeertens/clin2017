#!/bin/bash

subword_count=750

function segment_words { #size, in file, out file
  subword_count=$1
  segment_input_file=$2
  segment_output_file=$3
  PYTHONIOENCODING="UTF-8" python3 /home/roland/clinworkspace/subword-nmt/get_vocab.py < $segment_input_file  > /tmp/tmp_vocab.txt
  echo "done with part 1" 
   PYTHONIOENCODING="UTF-8" python3 /home/roland/clinworkspace/subword-nmt/segment-char-ngrams.py --vocab /tmp/tmp_vocab.txt --shortlist $subword_count < $segment_input_file > $segment_output_file 
  echo "done with part 2" 
}

name_subword_folder="subwords"$subword_count
echo $name_subword_folder
mkdir $name_subword_folder

segment_words $subword_count clin2017/lexicon.txt lexicon_subwords.txt

mv lexicon_subwords.txt $name_subword_folder
