TRAINSIZE=33000
TESTSIZE=4235
filename_train=subworded_prefix_roland_bible_1637.txt
filename_test=subworded_prefix_roland_bible_1888.txt
amount_to_split=500
python segment_words_subwords_script.py $amount_to_split clin2017/1637/bible.txt $filename_train
python segment_words_subwords_script.py $amount_to_split clin2017/1888/bible.txt $filename_test
head -n $TRAINSIZE $filename_train> infor.train.en
head -n $TRAINSIZE $filename_test > infor.train.nl
tail -n $TESTSIZE $filename_train  > infor.dev.en
tail -n $TESTSIZE $filename_test  > infor.dev.nl
	
name_subword_folder=prefix_splitted$amount_to_split
mkdir $name_subword_folder
mv infor.* $name_subword_folder

python3 ~/tensorflow/tensorflow/models/rnn/translate/translate.py --data_dir $name_subword_folder --train_dir $name_subword_folder --en_vocab_size $amount_to_split --fr_vocab_size $amount_to_split --num_layers 2

