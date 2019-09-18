python conll_split.py data/EntityMentionRelation/conll04.corp
python conll_stat.py data/EntityMentionRelation/conll04.corp
wget -O glove.6B.zip http://nlp.stanford.edu/data/glove.6B.zip
unzip -j glove.6B.zip glove.6B.50d.txt -d data/glove.6B
