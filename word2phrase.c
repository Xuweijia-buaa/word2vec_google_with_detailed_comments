//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define MAX_STRING 60

const int vocab_hash_size = 500000000; // Maximum 500M entries in the vocabulary

typedef float real;                    // Precision of float numbers

struct vocab_word {
  long long cn;
  char *word;
};

char train_file[MAX_STRING], output_file[MAX_STRING];
struct vocab_word *vocab;
int debug_mode = 2, min_count = 5, *vocab_hash, min_reduce = 1;
long long vocab_max_size = 10000, vocab_size = 0;
long long train_words = 0;
real threshold = 100;

unsigned long long next_random = 1;

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
// 从f里读取一个词 假设 space + tab + EOL 是词的分隔符
void ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}

// Returns hash value of a word [0-hase_size-1]
int GetWordHash(char *word) {
  unsigned long long a, hash = 1;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
// 没有这个词，returns -1  /  有，根据hash_table,返回word　idx
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

// Reads a word and returns its index in the vocabulary
// 从当前位置读一个词,如果读到文件结尾，或者之前字典里没有这个词，返回-1/否则返回word idx
int ReadWordIndex(FILE *fin) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(word);
}

// Adds a word to the vocabulary
// 给该值找到对应的没有被用过的hash值。更新hash table (hash:word_idx).  vocab[idx]:(word,count)
//　有必要的话，给实现分配的内存vocabV扩容
int AddWordToVocab(char *word) {
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
  strcpy(vocab[vocab_size].word, word);
  vocab[vocab_size].cn = 0;
  vocab_size++;
  // Reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 10000;
    vocab=(struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  vocab_hash[hash]=vocab_size - 1;
  return vocab_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
  int a;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  for (a = 0; a < vocab_size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab
    if (vocab[a].cn < min_count) {
      vocab_size--;
      free(vocab[vocab_size].word);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash = GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
    }
  }
  vocab = (struct vocab_word *)realloc(vocab, vocab_size * sizeof(struct vocab_word));
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < vocab_size; a++) if (vocab[a].cn > min_reduce) {
    vocab[b].cn = vocab[a].cn;
    vocab[b].word = vocab[a].word;
    b++;
  } else free(vocab[a].word);
  vocab_size = b;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  for (a = 0; a < vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}

//用train_file 初始化字典   .word,.cont
void LearnVocabFromTrainFile() {
  char word[MAX_STRING], last_word[MAX_STRING], bigram_word[MAX_STRING * 2];
  FILE *fin;
  long long a, i, start = 1;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  vocab_size = 0;
  AddWordToVocab((char *)"</s>");
  while (1) {
    ReadWord(word, fin);             // 读取train_file中每个word到word   (按照‘\n’分句。'\t',' '分词)
    if (feof(fin)) break;
    if (!strcmp(word, "</s>")) {　　　// new sentence , satrt==1
      start = 1;
      continue;
    } else start = 0;
    train_words++;
    if ((debug_mode > 1) && (train_words % 100000 == 0)) {
      printf("Words processed: %lldK     Vocab size: %lldK  %c", train_words / 1000, vocab_size / 1000, 13);
      fflush(stdout);
    }
    i = SearchVocab(word);           
    if (i == -1) {                   //　没有这个词，returns -1 ,
      a = AddWordToVocab(word);      //  新词 加到字典中，更新V,vocab.word,cont 更新hash_table(给该值找到对应的没有被用过的hash值)     有必要的话扩容vocab
      vocab[a].cn = 1;
    } else vocab[i].cn++;           //　有这个词,根据hash_table,返回word　idx
    if (start) continue;
    //两两相邻拼起来，截断后的bigram字符，加入字典(当做一个普通word)
    sprintf(bigram_word, "%s_%s", last_word, word); 　// bigram:赋值为 "lastword_word"   初始，lastword未赋值，strlen(last_word)==0,但size=MAX  (strlen不包括结束字符) 
    bigram_word[MAX_STRING - 1] = 0;　　　　　　　　　　// 截断bigram，使得bigram字符串最长为max_leng  最后一个字符为空字符0　　null==0＝＝'\0'
    strcpy(last_word, word);        　　　　　　　　　　//　word　　-->　last word
    i = SearchVocab(bigram_word);
    if (i == -1) {
      a = AddWordToVocab(bigram_word);　　　　　　　　
      vocab[a].cn = 1;
    } else vocab[i].cn++;
    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();  // 如果vocab_size过大了，不容易hash了，根据当前词频，减小vocab,vocab_size,重算hash.
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("\nVocab size (unigrams + bigrams): %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  fclose(fin);
}

void TrainModel() {
  // pa:前一个词的出现次数。如果上一个词已经被合并了，是pa==0, oov=1，该词不和上一个词合并　
  // pb:该词本身的出现次数  
  // pab:该词和前一个词组成的bigram出现的次数 a,b的共现次数
  // 该词不在词典里/该词前一个词不在词典里/该词和前一个词组成的bigram不在词典里/前一个词/该词 词频太低/前一个词已经合并  oov=1　不与前一个词合并
  // 只有score=P(AB)/P(A)P(B)  ,>  threshold 1:0.01,不独立，才合并，否则单独加入output_text中
  // 合并的，按a_b截断，加入output_text中
  long long pa = 0, pb = 0, pab = 0, oov, i, li = -1, cn = 0;
  char word[MAX_STRING], last_word[MAX_STRING], bigram_word[MAX_STRING * 2];
  real score;
  FILE *fo, *fin;
  printf("Starting training using file %s\n", train_file);
  LearnVocabFromTrainFile();        //　用train_file 初始化字典 . .word,.cont　　train_file同word_vec.c,　　只不过在初始化过程中考虑所有bigram(不考虑过长的)
  fin = fopen(train_file, "rb");
  fo = fopen(output_file, "wb");
  word[0] = 0;
  while (1) {
    strcpy(last_word, word);
    ReadWord(word, fin);            // 从train_file里读取一个词到　char * word
    if (feof(fin)) break;
    if (!strcmp(word, "</s>")) {　　//　从train_file读到一句话结尾，（\n）,ReadWord会返回</s>.  写入新文件output_file
      fprintf(fo, "\n");
      continue;
    }
    cn++;
    if ((debug_mode > 1) && (cn % 100000 == 0)) {
      printf("Words written: %lldK%c", cn / 1000, 13);
      fflush(stdout);
    }
    oov = 0;    // 该词不在词典里/该词前一个词不在词典里/该词和前一个词组成的bigram不在词典里/前一个词/该词 词频太低/前一个词已经合并  oov=1　不与前一个词合并
    i = SearchVocab(word);
    if (i == -1) oov = 1; else pb = vocab[i].cn;  // 该词不在词典里，oov=1
    if (li == -1) oov = 1;                        // 该词前一个词不在词典里，oov=１
    li = i;
    sprintf(bigram_word, "%s_%s", last_word, word);　// 得到新的bigram
    bigram_word[MAX_STRING - 1] = 0;　　　　　　　　　　//　同样截断
    i = SearchVocab(bigram_word);                   // 看是否在词典里
    if (i == -1) oov = 1; else pab = vocab[i].cn;   // 该bigram不在词典里，oov=1
    if (pa < min_count) oov = 1;
    if (pb < min_count) oov = 1;
    //
    if (oov) score = 0; else score = (pab - min_count) / (real)pa / (real)pb * (real)train_words; // P(AB)/P(A)P(B)    c/n(c1/n *  c2/n)  c/(c1c2)*n  　
    if (score > threshold) { //100
      fprintf(fo, "_%s", word);       //这个bigram出现次数首先需要大于min_count,然后P(AB)/P(A)P(B)要大于阈值。P(AB)概率远远大于P(A)P(B),1:0.001   　即有很小的独立性
      pb = 0;                         //已经合并了，这个词不和下一个词合并
    } else fprintf(fo, " %s", word); // score是0的词，直接将该词写进新的output_text
    pa = pb;
  }
  fclose(fo);
  fclose(fin);
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}


//执行两次，可以把原始文本转化为　所有可能的，概率高的２-gram, 3-gram，并写成新的文本文件　
// 之前其他符号全去掉，只留下字母，空格，换行，单引号，下划线
//之后全变为小写
int main(int argc, char **argv) {
  int i;
  if (argc == 1) {
    printf("WORD2PHRASE tool v0.1a\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors / word clusters / phrases\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-threshold <float>\n");
    printf("\t\t The <float> value represents threshold for forming the phrases (higher means less phrases); default 100\n");//去掉低频phrase
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
    printf("\nExamples:\n");
    printf("./word2phrase -train text.txt -output phrases.txt -threshold 100 -debug 2\n\n");
    return 0;
  }
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threshold", argc, argv)) > 0) threshold = atof(argv[i + 1]);
  vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word)); // just contain word/count
  vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));                       // vocab_hash数组，每个元素是一个整数。初始为500M个元素，用来hash不同word.初始全为0  vocab_hash[w_hash]=w_idx
  TrainModel();
  return 0;
}
