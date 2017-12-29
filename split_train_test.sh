TRAINSIZE=30000
filename_train=subworded_prefix_roland_bible_1637.txt
filename_test=subworded_prefix_roland_bible_1888.txt
head -n -$TRAINSIZE $filename_train> infor.train.en
head -n -$TRAINSIZE $filename_test > infor.train.nl
tail -n +$TRAINSIZE $filename_train | sed '1d' > infor.dev.en
tail -n +$TRAINSIZE $filename_test | sed '1d' > infor.dev.nl
	
name_subword_folder=prefix_splitted
mkdir $name_subword_folder
mv infor.* $name_subword_folder
