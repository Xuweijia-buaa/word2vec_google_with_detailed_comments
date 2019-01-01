make
if [ ! -e text8 ]; then
  wget http://mattmahoney.net/dc/text8.zip -O text8.gz
  gzip -d text8.gz -f
fi
time ./word2vec -train text8 -output vectors.bin -cbow 1 -size 200 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 20 -binary 1 -iter 15
./compute-accuracy vectors.bin 30000 < questions-words.txt
# to compute accuracy with the full vocabulary, use: ./compute-accuracy vectors.bin < questions-words.txt
# < 表示的是输入重定向的意思，就是把<后面跟的文件取代键盘作为新的输入设备。不作为命令行参数
