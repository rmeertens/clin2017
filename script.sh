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

segment_words $subword_count clin2017/1637/bible.txt input.txt
segment_words $subword_count clin2017/1888/bible.txt output.txt

TRAINSIZE=30000
head -n -$TRAINSIZE input.txt > infor.train.en
head -n -$TRAINSIZE output.txt > infor.train.nl
tail -n +$TRAINSIZE input.txt | sed '1d' > infor.dev.en
tail -n +$TRAINSIZE output.txt | sed '1d' > infor.dev.nl
	
mv input.txt $name_subword_folder
mv output.txt $name_subword_folder
mv infor.* $name_subword_folder
