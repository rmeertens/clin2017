
# coding: utf-8

# In[1]:
import sys
import re

_WORD_SPLIT = re.compile("([.,!?\"':;)( ])")

def get_frequencies_from_word(word):
    subwords = []
    for subword_length in range(0,len(word)):
        for i in range(len(word)-subword_length):
            subwords.append(word[i:i+subword_length+1])
    return subwords

def get_subword_and_frequencies(words):
    subword_frequency = dict()
    for word in words:
        for subword in get_frequencies_from_word(word):
            if subword not in subword_frequency:
                subword_frequency[subword]=0
            subword_frequency[subword]+=1

    subwords_and_frequency = list(subword_frequency.items())
    subwords_and_frequency = [(b,a) for a,b in subwords_and_frequency]
    return subwords_and_frequency


def get_n_subwords(words,first_n):
    subwords_and_frequency = get_subword_and_frequencies(words)
    subwords_and_frequency.sort(reverse=True)
    subwords = [word for freq,word in subwords_and_frequency if len(word)==1]
    for freq,word in subwords_and_frequency:
        if len(word)>1:
            subwords.append(word)
        if len(subwords)==first_n:
            break
    
    return subwords


def get_subworded(word,subwords):
    if word=="":
        return []
    for i in range(len(word),-1,-1):
      if word[:i] in subwords:
	base = [word[:i]]
	base.extend(get_subworded(word[i:],subwords))
	return base


def get_subworded_words(words,subwords):
    subworded = []
    for complete_word in words:
        subwords_here = get_subworded(complete_word,subwords)        
        subworded.append(subwords_here)
    return subworded


def basic_tokenizer(sentence):
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(_WORD_SPLIT.split(space_separated_fragment))
  return [w for w in words if w]


def subword_file(subword_count,filename_input,filename_output):
    to_output_lines = []
    with open(filename_input) as input_file:
        input_lines = [l for l in input_file]
    old_words = []
    for l in input_lines:
        tokenized_sentence = basic_tokenizer(l)
        old_words.extend(tokenized_sentence)
    subwords = get_n_subwords(old_words, subword_count)
    subwords.sort()
    for l in input_lines:
        tokenized_sentence = basic_tokenizer(l)
        this_line = ""

        for w in tokenized_sentence:
            tokenized_word_here = get_subworded(w,subwords)
            for token in tokenized_word_here[:-1]:
                this_line+=token+"@@ "
            this_line+=tokenized_word_here[-1]+ " "
        to_output_lines.append(this_line)
    with open(filename_output,'w' ) as output_file:
        for line in to_output_lines:
            output_file.write(line)
            output_file.write("\n")

if __name__ == "__main__":
  if len(sys.argv) != 4:
    print("use: python name.py amount input.txt output.txt")
  else:
    subword_file(int(sys.argv[1]),sys.argv[2],sys.argv[3])
#subword_file("clin2017/1637/bible.txt","subworded_prefix_roland_bible_1637.txt")
#subword_file("clin2017/1888/bible.txt","subworded_prefix_roland_bible_1888.txt")

