make
if [ ! -e news.2012.en.shuffled ]; then
  wget http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2012.en.shuffled.gz
  gzip -d news.2012.en.shuffled.gz -f
fi
sed -e "s/’/'/g" -e "s/′/'/g" -e "s/''/ /g" < news.2012.en.shuffled | tr -c "A-Za-z'_ \n" " " > news.2012.en.shuffled-norm0
# -e 以选项中的指定的script来处理输入的文本文件
# g: 匹配每一行有行首到行尾的所有符合字符 没有g:匹配到第一个符合的就结束,不匹配该行剩下的
# s/A/B/ 就是A替换B,　A，B正则表达式 
# -c 取补集。也就是所有不是"A-Za-z'_ \n"的元素，都用" "替代。只留下字母,_,换行，空格,单引号
time ./word2phrase -train news.2012.en.shuffled-norm0 -output news.2012.en.shuffled-norm0-phrase0 -threshold 200 -debug 2
time ./word2phrase -train news.2012.en.shuffled-norm0-phrase0 -output news.2012.en.shuffled-norm0-phrase1 -threshold 100 -debug 2
#tr 'a-z'　'A-Z' <./bash_profile1 将当前目录下的bash_profile1文件中的所有大写字母，转换成小写字母
# 得到phrase以后，才全变成小写
# < 表示的是输入重定向的意思，就是把<后面跟的文件取代键盘作为新的输入设备。
# > 输出重定向  比如输入一条命令，默认行为是将结果输出到屏幕。但有时候我们需要将输出的结果保存到文件，就可以用重定向。
tr A-Z a-z < news.2012.en.shuffled-norm0-phrase1 > news.2012.en.shuffled-norm1-phrase1
time ./word2vec -train news.2012.en.shuffled-norm1-phrase1 -output vectors-phrase.bin -cbow 1 -size 200 -window 10 -negative 25 -hs 0 -sample 1e-5 -threads 20 -binary 1 -iter 15
./distance vectors-phrase.bin
