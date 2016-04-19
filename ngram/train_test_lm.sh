# ========================================
# Train n-gram LM
# TODO: figure out the meaning of the smothing parameters in the lm training step


data_train=LM-train-100MW
data_test=data_devel_false_1
n=5

# 1. get vocabulary file
cat $data_train.txt | tr " " "\n" | sort | uniq > $data_train.vocab

# 2. get counts file
ngram-count -vocab $data_train.vocab -text $data_train.txt -order $n -write $data_train.$n.count -unk

# 3. estimate n-gram language model
ngram-count -vocab $data_train.vocab -read $data_train.$n.count -order $n -lm $data_train.$n.lm -gt1min 3 -gt1max 7 -gt2min 3 -gt2max 7 -gt3min 3 -gt3max 7

# 4. test 1 example against model
ngram -ppl $data_test.txt -order 4 -lm $data_train.$n.lm >> $data_test.score

# 5. get the perplexity from the scores file
grep -E -o "ppl=.{1,7}" $data_test.score| sed 's|ppl\= ||g' > $data_test.ppl



# ==========================================
