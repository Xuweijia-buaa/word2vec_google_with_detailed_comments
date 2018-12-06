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

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;                    // Precision of float numbers

struct vocab_word {
  long long cn;
  int *point;
  char *word, *code, codelen;//  word:词本身 word[24]
};

char train_file[MAX_STRING], output_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
struct vocab_word *vocab;
int binary = 0, cbow = 1, debug_mode = 2, window = 5, min_count = 5, num_threads = 12, min_reduce = 1;
int *vocab_hash;
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;
long long train_words = 0, word_count_actual = 0, iter = 5, file_size = 0, classes = 0;
real alpha = 0.025, starting_alpha, sample = 1e-3;
real *syn0, *syn1, *syn1neg, *expTable;
clock_t start;

int hs = 0, negative = 5;
const int table_size = 1e8;
int *table;
// 根据单词出现频率的(3/4)，生成一张均匀分布的表，每个单词占据表中位置的长度正比于单词频率。从table中随机抽样相当于按概率抽样，返回word_idx
void InitUnigramTable() {
  int a, i;
  double train_words_pow = 0;
  double d1, power = 0.75;
  table = (int *)malloc(table_size * sizeof(int));                              // table:1e8个int数组,初始随机
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);  //所有词总的（3/4）power
  i = 0;
  d1 = pow(vocab[i].cn, power) / train_words_pow;                               //每个词的power后的出现概率
  //          词０(0.25)　　　  词１(0.15-0.25)　 词２(0.3-0.59)　  　词Ｖ-1(0.01-0.2)
  //d1                  0.25          0.4/0.5              0.8/0.99            1　　　　　　　　//到该词时的累计概率　　该词在table中占据的长度与出现概率成正比
  //a          0  1  2       3   4   5         6  7  8   9
  //table[a]   0  0  0       1   1   1         2  2  2   2
  //d1                  0.25          0.4/0.5              0.8/0.99            1　　　　　　　　//到该词时的累计概率　　该词在table中占据的长度与出现概率成正比
  //a          0  1  2       3   4   5         6  7  8   9
  //table[a]   0  0  0       1   1   1         2  2  2   2 
  //d1                  0.25           0.4/0.5             0.8/0.99             1　　　　　　　　//到该词时的累计概率　　该词在table中占据的长度与出现概率成正比
  //a          0  1  2       3   4   5         6  7  8   9
  //table[a]   0  0  0       1   1   1         2  2  2   2 
  //d1                  0.25        0.4(>=)                0.9                  1　　　　　　　　//到该词时的累计概率　　该词在table中占据的长度与出现概率成正比
  //a          0  1  2       3   4         5  6  7  8   9
  //table[a]   0  0  0       1   1         2  2  2  2   2  
  //d1                  0.25        0.4(>=)                0.8(>=)              1　　　　　　　　//到该词时的累计概率　　该词在table中占据的长度与出现概率成正比
  //a          0  1  2       3   4         5  6  7  8                  9
  //table[a]   0  0  0       1   1         2  2  2  2                  3 
  //给每个词分配 table_size×p个空间   p是power加权后的p  
     
  for (a = 0; a < table_size; a++) {
    table[a] = i;
    if (a / (double)table_size > d1) {  //change into >=
      i++;
      d1 += pow(vocab[i].cn, power) / train_words_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1;// 如i=V-1时，d1<1,如0.999 a/(double)table_size才有可能>d1，i才有可能==V　d1加一个乱七八糟的数。但剩下table都给最后一个词V-1
  }
}

// 从f里读取一个词 假设 space + tab + EOL 是词的分隔符
// input: word[100],file 
// word重新赋值　file状态保留　　原始文本每句话用'\n'分隔，词用'\t',' '分隔。　加入'<s>'，作为一个token,表示不同句子之间的分隔
void ReadWord(char *word, FILE *fin) {
  int a = 0, ch;         // a: index of 'word'.
  // Read until the end of the word or the end of the file.
  while (!feof(fin)) {   // !feof(fin)：文件读取未结束　　　　
    ch = fgetc(fin);　　　// 从指定的文件f中读取下一个字符，并把位置标识符往前移动. 如果已经读到文件末尾或出错，返回EOF,且把状态feof(fin)置为非0值(true)/ferror置为非0值(true)
                         //  while (c != EOF) ch = fgetc(fin);
    if (ch == 13) continue;　// '\r' ASCII码为13 是回车   '\n'ASCII码为10 是换行
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {   　　　　// 如果读到分隔符
      if (a > 0) {　　　　　　　　　　　　　　　　　　　　　　 　　　　 // 而且word里至少有一个字符了
        if (ch == '\n') ungetc(ch, fin);　　　　　　　　　　　　　　// （如果下一个字符是换行符，退回去，让之后的word一看到就知道是一句的结尾）
        break;　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 // 结束读取，（并且把下一个字符是换行符退回去，让之后的word一看到就知道是一句的结尾）
        　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　// word[100]的前i个内容就是要读取的单词了。停止继续读.
      }
      if (ch == '\n') {　　　　　　　　　　　　　　　　　　　　　　　　// 一开始就遇到了'\n',说明是句子开头。直接把句间分隔标志"</s>"给word,作为本次读到的单词。（常量字符串连同'\0'一起copy给word）
        strcpy(word, (char *)"</s>");
        return;
      } else continue;　　　　　　　　　　　　　　　　　　　　　　　　　//一开始碰到了其他分隔符，比如空格，tab，不用管，直接跳过
    }
    word[a] = ch;  //　word i' 位置: ch
    a++;           
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words  如果一个词的长度已经到100了(a=99),继续读直到这个词结束。但是a会一直保持99.
  }
  word[a] = 0;                      // 在单词碰到分隔符的位置i停止读，结尾加上ascll码是0的空字符‘\0’  /  (int)null　　最长的词a==99,a[99]=0 。包括空字符长100
}

// hash word to a unique value [0-hase_size-1]
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}


//　没有这个词，returns -1
// 有，根据hash_table,返回word　idx
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;   // 没有直接hash, 一直往下走，如果本来没有这个词，就碰上的是其他词的hash,然后就是某个没有被占用的hash. 　指定word为这个hash. 之前无此词，返回-1
    //否则以前就有这个词的hash(从直接hash往下找，总能找到该word对应的hash，从而返回word_idx
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];//!strcmp:相同
    hash = (hash + 1) % vocab_hash_size;                                    
  }
  return -1;
}

// Reads a word and returns its index in the vocabulary
// 从当前位置读一个词,如果读到文件结尾，或者之前字典里没有这个词，返回-1/否则返回word idx
int ReadWordIndex(FILE *fin) {
  char word[MAX_STRING];
  ReadWord(word, fin);　　　　　　　　　　// 从当前位置读一个词到word中
  if (feof(fin)) return -1;
  return SearchVocab(word);　　　　　　　// 如果读到文件结尾，或者之前字典里没有这个词，返回-1/否则返回word idx
}

// Adds a new word to the vocabulary (one that hasn't been seen yet).
// 给该值找到对应的没有被用过的hash值。更新hash table (hash:word_idx).  vocab[idx]:(word,count)
//　有必要的话，给实现分配的内存vocabV扩容
int AddWordToVocab(char *word) {
  unsigned int hash, length = strlen(word) + 1;　　//length: 词word实际长度（包括空字符）。　　strlen 函数返回的是字符串的实际长度. +1包括了空字符
  if (length > MAX_STRING) length = MAX_STRING;   // 词包括空字符，最多允许长100
  // 给vocab[vocab_size].word 赋值,并统计出现次数
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char));　//初始vocab_size==0 （global）　　vocab[vocab_size]：　a struct
  strcpy(vocab[vocab_size].word, word);                          
  vocab[vocab_size].cn = 0;
  vocab_size++;
  // Reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size) {　　　　//vocab_size +1 :当前实际的|V|   再增加vocab_size,会超出目前的vocab大小。扩容：
    vocab_max_size += 1000;
    vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));　//尝试重新调整之前调用 malloc 或 calloc 所分配的 ptr 所指向的内存块的大小 (旧地址首地址，新内存大小)
  }
  hash = GetWordHash(word);                                                                  //该词的hash值　[0-30e6]
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;                        // vocab_hash[hash] ＝= -1: 该hash值还没有被任何词占用。vocab_hash[hash]＝该词的idx (添加顺序，from 0)
  vocab_hash[hash] = vocab_size - 1;                                                         // 如果这个hash值被占用，这个word需要换一个还没有被用过的hash值　　hash:word_idx
  return vocab_size - 1;
}

// 比较函数　使返回值<0的a在前
int VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;　// 频率高的元素在前
}

// 按词频对字典重新排序，词频越高越靠前.方便之后直接舍去词频较低的词占用的string内存 (vocab中原始第一个word是</s>,位置不变). 
// (重排＋扔掉min_count后)更新vocab,V,hash_table（词的hash值到词位置），train_words(所有单词出现次数的总和)，让每个词的code,point指向一片char[40],int[40]的内存
void SortVocab() {
  int a, size;
  unsigned int hash;
  // base指向要排序的数组的第一个元素的指针.　nitems是数组数目　size:每个元素字节数　compar比较函数
  // void qsort(void *base, size_t nitems, size_t size, int (*compar)(const void *, const void*))
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);//只排后边的V-1个词　　V就是实际的词数。最后一个词pos v-1
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;　　　　　　　　　　// 重新计算hash值
  size = vocab_size;
  train_words = 0;
  for (a = 0; a < size; a++) {
    // 除第一个词</s>以外的任意词，频率太小的词抛弃
    if ((vocab[a].cn < min_count) && (a != 0)) {
      vocab_size--;　　　　　　　　　　　　　　　　　// V=V-1
      free(vocab[a].word);                     // 释放这个词所占用的string内存空间. (.cont这些没有释放))
    } else {
      //重算hasｈ值。hash:word直接算的或者是后边几个，直到没人占用的值
      hash=GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;                   // hash:word idx
      train_words += vocab[a].cn;             // 所有留下来的单词的,出现的总次数
    }
  }
  // 重新分配下vocab占用的内存大小。按照所有剩下来的词重分内存Ｖ+1。缩小(或扩大)后，原先内存中的内容不变，返回新数组首址vocab
  vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
  // Allocate memory for the binary tree construction
  for (a = 0; a < vocab_size; a++) {
    vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));　　//char *   指向40个char元素的数组　code[40]
    vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));    // int *            int         point[40]
  }
}

// vocab_size过大，不好hash了 removing infrequent tokens
// vocab[i]的word,cont都变成新的。（Ｖ）
//　某次减小了Ｖ之后。随着Ｖ增大，又太大了。下次再执行需要增加min_reduce，否则V没有变化
void ReduceVocab() {
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < vocab_size; a++) if (vocab[a].cn > min_reduce) {// 遍历当前词典（这时候还没有添加完），把当前词频>min_reduce==1的a的word,cout留下，来覆盖词b。
    vocab[b].cn = vocab[a].cn;
    vocab[b].word = vocab[a].word;
    b++;
  } else free(vocab[a].word);　　　　　　　　　　　　　　　　　　　　　　// 词频过小的，word空间直接被释放.该内存会成为b,被之后留下的ａ的值替换掉
  vocab_size = b;　　　　　　　　　　　　　　　　　　　　　　　　　　　　　//　剩下的词的数目
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;　　　　//　当前vocab的每个word重算hash值.有hash 值的，都是存在过的
  for (a = 0; a < vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}

// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
void CreateBinaryTree() {
  long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];//point[40]
  char code[MAX_CODE_LENGTH]; //code[40]
  long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));    // ２V+1 记录每个词的权重，以及huffuman树每个node的权重　初始均为0
  long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));　　//　初始均为０。保存节点路径中每个node的编码码号　左0右1分类器
  long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;                          // count前V个都是单个词的权重,已经排好序
  for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;　　　　　　　　　　　　// cont之后Ｖ个，初始化为1e15　　　最后一个依然是0
  pos1 = vocab_size - 1;
  pos2 = vocab_size;
  // count 前V个之后的后V-1个，每一个node count[V+a]放node权重　　实际只需要2V-1,但pos2有可能＋＋两次，＋２个位置
  // n-4 n-3  n-2  n-1 | n  n+1   
  // 第一个判断中　pos1:V中前一个还没有合并的节点　　　pos2:上一棵树的权重构成的还没有被合并的inner节点
  // 第二个判断中 找到次小的。更新pos1,pos2
  // 第三个判断：如果全部叶子节点都被合并了，只能合并剩下的node节点。pos2:最小的还没有被合并的node节点。直到塞满V+V-2
  // pos1=n-1, pos2=n(1e5)　　　wn= wn-1 + wn-2     pos1:未合并的最小的叶子节点位置
  // pos1=n-3  pos2=n(wn)  　　 wn+1= wn-3 + w　　　 pos2:未合并的最小的node节点位置  
  // count[2V-2]=1e15  count[2V-1]=1e15  count[2V]=0   之后count[2V-2]被赋值，作为root结点
  
  // 2V-2个节点赋了值　count[V]-count[2V-2],共V-1个node节点
  for (a = 0; a < vocab_size - 1; a++) {
    // First, find two smallest nodes
    // 最小的，不是叶子中还没有结合的前一个节点(pos1)，就是已经合并的综合节点（pos2）
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min1i = pos1;　　　　　　　　　　　　　　　// １如果最小的是前一个未合并节点n-3，第二小的不是已经合并的综合节点n（pos2），就是下下一个未合并的节点n-4（pos=pos-1）
        pos1--;
      } else {
        min1i = pos2;                        // ２如果最小的是综合节点n，第二小的就一定是下一个未合并的节点n-3（pos1）.（下一个未合并的综合节点是pos2+1==1e15，下一个判断没有悬念）
        pos2++;
      }
    }                              
    else {
      min1i = pos2;
      pos2++;
    }

    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {     //　pos1:作为第二小的叶子节点位置，要么是上一轮的下下一个未合并节点n-4（１）/　要么是下一个未合并节点n-3（２）
        min2i = pos1;                      //　如果是情况２，只要pos2还没有到2V, 未合并的综合节点值总是1e5.pos2不变。min2i = pos1。n合并n-3,pos1==pos-1=n-4 pos2==n+1
        pos1--;　　　　　　　　　　　　　　　　 //  如果是情况１，最小的是下一个未合并节点n-3
        　　　　　　　　　　　　　　　　　　　　 //             那么n-3要么和pos1(下下一个未合并节点)n-4合并. 那么之后的pos1就是n-5，未来最小值的候选者之一，会和之后的n+1比较
      } else {                            //             要么和pos2 n合并. 那么pos1保持原来的n-4,作为未合并节点。而pos2变为n+1,因为n也被合并走了。
        min2i = pos2;                     
        pos2++;
      }
    } 　　　　　　　　　　　　　　　　　　　　　　
    else {
      min2i = pos2;
      pos2++;
    }

    count[vocab_size + a] = count[min1i] + count[min2i]; // 每个node额权重
    parent_node[min1i] = vocab_size + a;　　　　　　　　　　//　每次的合并信息。合并的两个节点位置的父节点位置idx　n-1-->ｎ
    parent_node[min2i] = vocab_size + a;                //                                          n-2-->n
    binary[min2i] = 1;                                  // 　左0右1
  }
  //            n(0　待定，看他是左孩子还是右孩子)
  //       n-1(0)  n-2(1)

  // 记录每个单词，从叶子节点到根节点的路径信息，更新vocab中每个单词的信息（编码信息，以及对应路径的node idx）：
  //          root                point[0]: pos in node vector
  //        0      1              point[1]
  //       0  1                   point[2]
  //     0  1                     point[3]
  //       0 1                    point[4]: word_idx
  //    (leaf)
  // vocab.codelen=4  (不包括root的路径长度/码长)
  //　vocab.code： 从(root节点　到叶子节点]的编码　  code[0]-code[3] 0010 (实际10010,编码不包括root因为都一样.规定左0右1）code[0]第一个分类器的solid truth
  //                                                                                                           code[3]最后一个分类器，叶子节点处的solid truth
  //  vocab.point：从[root节点　到叶子节点]的node位置　,point[0]-point[i-1]：[root节点　到叶子节点）的node在所有node节点中的位置。如point[0]为root,位置为node[V-2].　
  //                                               point[i]:叶子节点对应的词idx-V　　
  for (a = 0; a < vocab_size; a++) {
    b = a;  //从每个词的idx(叶子节点位置)到root结点的每个node，在count中的位置　
    i = 0;　//从叶子节点出发，到达的第i个父节点
    while (1) {
      code[i] = binary[b];         //　从下往上路过的第i个node的编码0/1,包括叶子节点，不包括root          0/1/1/0/
      point[i] = b;　　　　　　　　　 //  从下往上路过的第i个node结点在count中的位置 包括叶子节点,不包括root  5/10/12/|2V-2|　　　point[0]-point[i-1]: 从leaf到root的下一个节点，在count中位置
      i++;
      b = parent_node[b];
      if (b == vocab_size * 2 - 2) break; // count[2V-2] 直到到达root结点　
    }
    vocab[a].codelen = i;                 // s实际码长，code[0]-code[i-1],不包括root
    vocab[a].point[0] = vocab_size - 2;   // p[0]=V-2　root结点在node节点中的位置　0-V-2,共V-1个node结点
    for (b = 0; b < i; b++) {
      vocab[a].code[i - b - 1] = code[b];　　         // 编码倒过来　from code[i-1]=code[0]leaf to code[0]=code[i-1] 从root节点　到叶子节点的编码　左0右1
      vocab[a].point[i - b] = point[b] - vocab_size; // point[0]=V-2   point[1]=point[i-1]-V  point[i]=point[0]-V 从root到叶子节点经历的node在所有node节点中的位置
    }
  }
  free(count);
  free(binary);
  free(parent_node);
}
//用train_file 初始化字典   .word,.cont
void LearnVocabFromTrainFile() {
  char word[MAX_STRING];
  FILE *fin;
  long long a, i;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  vocab_size = 0;
  AddWordToVocab((char *)"</s>");　　　　　　　　// </s>放在字典首位　V=1　cn=0 分配hash值
  while (1) {
    ReadWord(word, fin);                      // 读取train_file中每个word到word  同样按照‘\n’分居。'\t',' '分词
    if (feof(fin)) break;
    train_words++;
    if ((debug_mode > 1) && (train_words % 100000 == 0)) {
      printf("%lldK%c", train_words / 1000, 13); // 13,回车
      fflush(stdout);　　　　　　　　　　　　　　　　　　// 清空stdout的缓冲区,输出缓冲区里的东西打印到标准输出设备上
    }
    
    i = SearchVocab(word);　　　　　//　没有这个词，returns -1  /  有，根据hash_table,返回word　idx
    if (i == -1) {
      a = AddWordToVocab(word);   //　新词 加到字典中，更新V,vocab.word,cont 更新hash_table(给该值找到对应的没有被用过的hash值0 有必要的话扩容vocab
      vocab[a].cn = 1;
    } else vocab[i].cn++;         //否则词频累加
    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();　// 如果vocab_size过大了，不容易hash了，根据当前词频，减小vocab,vocab_size,重算hash.
    　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　// Ｖ多次增大到需要执行该语句，每次需增大词频min_reduce
  }
  　
  SortVocab();　　// (重排＋扔掉min_count后)更新vocab,V,hash_table（词的hash值到词位置），train_words(所有单词出现次数的总和)，让每个词的code,point指向一片char[40],int[40]的内存
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  file_size = ftell(fin);　//文件包含字节数目
  fclose(fin);
}
// 保存字典本身（word cont\n) 去掉min_count,去掉min_reduce for hash,开头<\s>）
void SaveVocab() {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn); 
  fclose(fo);
}

// 从read_vocab_file中读出所有单词＋词频，　扔掉min_count,构建字典(按词频排序的，最长100char)，hash_table（字典到idx）.　
// file_size：train_file文件字节数　 train_words:所有单词出现的总次数  让每个词的code,point指向一片char[40],int[40]的内存
void ReadVocab() {
  long long a, i = 0;
  char c;
  char word[MAX_STRING];　　　　　　　　　　　　　　　// word[100]
  FILE *fin = fopen(read_vocab_file, "rb");      // 每个词只出现了一次
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;　　//vocab_hash_size:3000000. 值全部初始化为-1  int数组
  vocab_size = 0;
  while (1) {　　　　　　　//一直读到文件结束
    ReadWord(word, fin); // 从fin里读取一个词到　char * word
    if (feof(fin)) break;// 如果文件读完了，word里只有word[0]=空字符　feof(fin)＝＝非０值
    a = AddWordToVocab(word); // 把该词添加到词表中(word,count=0); 并返回该词的位置;　给该词设定该hash值，如果按照hash规则得到的hasn值被占用，就选hasn表中下一个hash值.最终　该hash值:word_idx. (当前V-1)
    fscanf(fin, "%lld%c", &vocab[a].cn, &c);　// 按file设定词频（vocab　file中每个word：'w cn\n'　　' '在AddWordToVocab中读到了不塞回去。接下来直接读cn
    i++;
  }
  //读完了全部vocab中单词，也按照规定词频更新了每个词的信息
  // (按count重排＋扔掉min_count后)更新vocab(V+1),V,hash_table（词的hash值到词位置），train_words(所有单词出现的总次数)，让每个词的code,point指向一片char[40],int[40]的内存
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);// int fseek(FILE *stream, long int offset, int whence) 最後一個位置
  file_size = ftell(fin);　//返回文件字节数
  fclose(fin);
}

void InitNet() {
  long long a, b;
  unsigned long long next_random = 1;
  //第一层变量　int posix_memalign (void   **memptr,size_t alignment,size_t size);好处：使得较大字节也能自然对齐，加快存取速度
  //memptr分配好的内存空间的首地址/  alignment 对齐边界,2^,分配的内存空间地址是alignment的倍数/ size字节:返回的动态内存字节数动态内存　
  //返回的内存块的地址放在了memptr里面，成功，函数返回值是0。否则syn0 == NULL
  a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real)); // 生成input vector syn0: V*(100)
  if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}

  //第二层变量
  if (hs) {                                                                                      // need |V-1| node vector for huffman tree
    a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(real));// syn1: V*(100),初始化为０
    if (syn1 == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
     syn1[a * layer1_size + b] = 0;
  }
  if (negative>0) {                                                                                //neg sample number
    a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real));// syn1neg: V*(100),初始化为０　output vector
    if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
     syn1neg[a * layer1_size + b] = 0;
  }
  for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++) {
    next_random = next_random * (unsigned long long)25214903917 + 11;
    syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;    // 随机初始化input vector　 初始化为[-0.5-0.5]/100
    　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　// (next_random & 0xFFFF) / (real)65536) 0-1
  }
  CreateBinaryTree();
}

//每个线程处理1/8文件，处理５epoch。　每轮每个窗口，skip-gram 每对（vc,vm），vm累积一个pos和所有neg的梯度（包括正例uc，负例u1-uk）(所有node的梯度)，之后更新
//                                                                  neg:每个output vector算到时实时更新(包括pos uc) 
//                                                                  hs: 每个node vector 算到时实时更新　（只算uc一条路径）
//                                                                  更新后算下一对(vc,vm+1)  主要依赖辅助向量的快速更新，vc在该过程中没有更新.主要是vm，uk(node)更新
//                                    cbow vc=(v1+...+vm)/m    　　　vc累积一个pos和所有neg的梯度（包括正例uc，负例u1-uk），(所有node的梯度)之后更新
//                                                                  neg:每个output vector算到时实时更新(包括pos uc) 
//                                                                  hs: 每个node vector 算到时实时更新（只算uc一条路径）
//                                                                  更新后算下一窗(vc+1,*)  主要依赖辅助向量的快速更新， vc在该过程中没有更新.主要是vm，uk(node)更新　　　　　　　　　　
void *TrainModelThread(void *id) {
  long long a, b, d, cw, word, last_word, sentence_length = 0, sentence_position = 0;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1]; //　sen[0] -sen[1000]句子最长可以包括1000个词的idx
  long long l1, l2, c, target, label, local_iter = iter;                       // iter=5
  unsigned long long next_random = (long long)id;
  real f, g;
  clock_t now;
  real *neu1 = (real *)calloc(layer1_size, sizeof(real));    // neu1[100] for CBOW architecture
  real *neu1e = (real *)calloc(layer1_size, sizeof(real));   // neu1e[100]for both architectures.
  FILE *fi = fopen(train_file, "rb");
  fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);　// 该thread从文件train_file的i/8部分开始读取内容　0/8 1/8 ... 7/8
  while (1) {
    // 1 每读10000个词，输出一下，更新学习率
    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count - last_word_count;　　//　到此iter的此刻，所有线程真实读了的词数目
      last_word_count = word_count;　　　　　　　　　　　　　　　//　之前读了的词，作为下次再进入时的base.
      if ((debug_mode > 1)) {
        now=clock();
        printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
         word_count_actual / (real)(iter * train_words + 1) * 100,                       // 所有线程目前读的总词数/(需要读的词数：epoch*所有线程需要读的所有词)
         word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));　　//CLOCKS_PER_SEC:系统时间除以这个值，就可以得到秒数
        fflush(stdout);
      }
      alpha = starting_alpha * (1 - word_count_actual / (real)(iter * train_words + 1));　//　a=a(1-jindu) 进度越大，a越小。但最小不小于0.0001a
      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
    }
    // 2 retrieves the next sentence from the training text and stores it in 'sen':sen[0] -sen[999] 　从fi此时开始的地方读 sen[i]:word_idx) 直到文件结束/句子结束（readword会把遇到的\n变成<\s>)/1001个词  sen[0] -sen[1000]）
    if (sentence_length == 0) {
      while (1) {
        word = ReadWordIndex(fi);  // 从文件当前位置读一个词,如果读到文件结尾/字典里没有这个词，返回-1/否则返回word idx
        if (feof(fi)) break;       // 读完文件了，结束
        if (word == -1) continue;　// If the word doesn't exist in the vocabulary, skip it.
        word_count++;　　　　　　　　// Track the total number of training words processed.
        if (word == 0) break;　　　//读到某句的结尾了，break
        // 频率太高的，丢掉
        if (sample > 0) {
          real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn; //　keep_p=sqrt(n/(N*rate))+1  * (rate*N/n)
          next_random = next_random * (unsigned long long)25214903917 + 11;
          if (ran < (next_random & 0xFFFF) / (real)65536) continue;          //(next_random & 0xFFFF) / (real)65536) 0-1   drop_prob:1-ran
        }
        sen[sentence_length] = word;  //word_idx
        sentence_length++;
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      sentence_position = 0;
    }
  // 3 该1/8文件遍历完了一次，进入下一轮/结束
    if (feof(fi) || (word_count > train_words / num_threads)) {
      word_count_actual += word_count - last_word_count; //global 　该时刻所有线程读到的所有词.包括之前iter读的
      local_iter--;
      if (local_iter == 0) break;                        //　每个线程循环5轮。每一轮从文档总字节的id/8处开始读，直到读的内容达到总word数目的1/8或者到文件结尾结束
      　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　// 最后一轮结束，才结束这个线程。
      　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　// 每个线程把[i-1/8-i/8]循环５遍　　　（[0/8-1/8],..,..[7/8-1]） 
      word_count = 0;　　　　　　　　　　　　　　　　　　　　　　//每一轮初始化　
      last_word_count = 0;
      sentence_length = 0;
      fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);//每一轮从文档总字节的id/8处开始读
      continue;
    }

　　//此时sen中读了一个完整的句子
   //处理完这个sentence_position，从whlle开始处理下一个sentence_position。
   //直到sentence_position＝＝sentence_length-1．这个句子结束。重新从while开始，读一个句子sen,得到新句子新的的实际长度。并使得sentence_position重新从0开始
    
    word = sen[sentence_position];　　　　　　　　　　//sentence_position: 处理该句子时，此刻的窗中心。从句首0移到句尾l-1
    if (word == -1) continue;                     // 实际上不会读到-1.遇到-1在２中直接skip了
    for (c = 0; c < layer1_size; c++) neu1[c] = 0;　// neu1[100]＝０ ,只有cbow用.用来存平均后的中心向量
    for (c = 0; c < layer1_size; c++) neu1e[c] = 0;// neu1e[100]＝０　用来存这个窗内的中心词（vc/平均vc）对应的梯度。等一个窗都过去，一起更新
    next_random = next_random * (unsigned long long)25214903917 + 11;
    b = next_random % window;                      //设置随机窗长　　window:单边窗长  b：随机缩小单边窗长到b ,[0-w-1]. 原始单边是５，随机后最长单边窗长是４

    // cbow　　　该窗内中心词固定时,最后同时更新input vector v1,v2,...vm　（根据算这个窗内的vc平均的累积梯度）
        // neu1e:窗内所有context word对应的平均input vector　vc的g
        // neu1e:neg 累加了以该word为中心，所有n_negtive对应的g. 包括和自己uc.　output vector是实时更新的，每抽一个正/负样本更新一次
        //       hs  累加了root-wc这个路径上每一个分类器的分类误差造成的g.       node   vector是实时更新的，每算一个分类器更新一次 
        // vc=(v1+v2+...vm)/m
        // dvm=dvc*m  (但是这里是按相同算的，因为neu1e梯度在量级上也累加了许多二分类器，负例)　该过程中node vector一般更新一次　　
        // 为了量级相同，每个input vector更新同vec
        // hs: P(uc｜vc平均) 每一层 
        // neg:P(uc,vc平均).P(uk,vc平均) 每一个正/负例　正例只有一个：uc
    if (cbow) {  //train the cbow architecture
      // in -> hidden 把窗内context　word向量累加，得到本次的input vector Vc平均neu1
      cw = 0;
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        c = sentence_position - window + a;　　// 第i个context word在句子中的实际位置　sentence_position:中心词位置　a-W 第i个context词相对中心词的偏移量 i:0-2w  offset:-w-w　
                        // win=2 l(sentence_length)=7
                        // b=0:    sen_pos=0 c=[vc,1,2] sen_pos=1 c=[0,vc,2,3]  sen_pos=2 c=[0,1,vc,3,4]  ,... c=[2,3,vc,5,6]... sen_pos=5 c=[3,4,vc,6], sen_pos=6 c=[4,5,vc]
                        // b=w-1=1　　　　　　　c=[vc,1]          　 c=[0,vc,2] 　　　　　　　c=[1,vc,3]
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c]; //该context word的word-idx
        if (last_word == -1) continue;
        for (c = 0; c < layer1_size; c++) neu1[c] += syn0[c + last_word * layer1_size];  // V,h　　h*idx+j 该context word第j(c)个元素 
        cw++;                                                                            //该窗内context word数目
      }
      if (cw) {
        for (c = 0; c < layer1_size; c++) neu1[c] /= cw;　　// cbow输入向量　vc平均。不论hs/ng

        if (hs) //每个word对应的每个分类器的node vector，实时更新, input　word的ｇ,存入neu1e　　1,100
          // 预测中心词，P(vc|context)  : node[i]*vc_context leaf是wc
          //          root                point[0]: pos in node vector
  　　　　　//        0      1              point[1]     code[0]=0
  　　　　　//       0  1                   point[2]     code[1]=0
  　　　　　//     0  1                     point[3]     code[2]=1
  　　　　　//       0 1                    point[4]: word_idx  code[3]=0  
  　　　　　//    (leaf)                    point[0]-code[0], ... point[3]-code[3]  父节点的node vector对子节点的分类情况
          // vocab.codelen=4  (不包括root的路径长度/码长)
          //　vocab.code： 从(root节点　到叶子节点]的编码　  code[0]-code[3] 0010 (实际10010,编码不包括root因为都一样.规定左0右1）code[0]第一个分类器的solid truth
          //                                                                                                           code[3]最后一个分类器，叶子节点处的solid truth
          //  vocab.point：从[root节点　到叶子节点]的node位置　,point[0]-point[i-1]：[root节点　到叶子节点）的node在所有node节点中的位置。如point[0]为root,位置为node[V-2].　
          //                                               point[i]:叶子节点对应的词idx-V　
          // p(context w|center)=sigmoid(-code[0]*  point[0],vc)  *  sigmoid(｜code[i]｜point[i],vc)  * sigmoid(point[3],vc)
        　for (d = 0; d < vocab[word].codelen; d++) {      //中心词的路径　from root to leaf 左0右1
          f = 0;
          l2 = vocab[word].point[d] * layer1_size;        // 该node的起始位置(在node vector中)
          // Propagate hidden -> output
          for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1[c + l2];//  f=point[i]*vc    (syn1[l2]),(neu1)
          if (f <= -MAX_EXP) continue;
          else if (f >= MAX_EXP) continue;
          // expTable[(f+6)*(1000/12)]==1/(1+exp(-f))
          else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]; // f= exp[i]   i=f+6 * (1000/12) --> f=exp[(f+6)*(1000/12)
          
          // 每个分类器，单独计算概率，单独计算损失。不一次性算总路径，更新完上边node，再算下边
          // P=1-sigmoid(point[i],vc) code[i]==1/  sigmoid(point[i],vc)  code[i]==0    将0(左边)作为正确
          // P= (1-f)^(code[i])  *  f^(1-code[i])
          //  logP= code[i] log(1-f)  + (1-code[i])*log(f)
          //  logP/f=code[i] * 1/(1-f)  + (1-code[i])* (1/f)
          //  f/x= f(1-f)
          // logpP/x= code[i]*f + (1-code[i])*(1-f)= 1-code[i]-f    (logP/(vc*node[i]))
          // 'g' is the gradient multiplied by the learning rate
          g = (1 - vocab[word].code[d] - f) * alpha;
          //　剩下的梯度vc*node[i]
          // 求对中心向量vc的梯度.每个元素vc[i][j],梯度是  g*pos[j]
          // input vector的梯度，vc=v1+v2+...vm/m    dm=dvc*cw
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];　　// input vector梯度存起来，不实时更新
          // 对每个node p[i]的每个元素p[i][j],梯度是  g*vc[j]    
          // node vector syn1[l2] 每个元素c根据梯度更新  (point[i]==l2,。梯度上升)　node向量实时更新
          for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * neu1[c];
        }
        // 有了vc neu1,cw,NEGATIVE SAMPLING   logsigmoid(vc * uc) + sum logsigmoid(-vc *uk)
        if (negative > 0) for (d = 0; d < negative + 1; d++) {
          if (d == 0) {
            target = word;   // centeral word idx  作为uc的idx
            label = 1;
          } else {
            next_random = next_random * (unsigned long long)25214903917 + 11;
            target = table[(next_random >> 16) % table_size];// 根据单词出现频率的(3/4)，从table中随机按概率抽样，返回word_idx,作为uk的idx
            if (target == 0) target = next_random % (vocab_size - 1) + 1; // 如果抽到句子分隔符，重新用这个random拿一个词，1-V-1
            if (target == word) continue;                                 //　抽到uk==uc,放弃这个neg,直接抽下一个neg词
            label = 0;
          }
          l2 = target * layer1_size;// syn1neg:output vector [V,100] 　　　l2:该neg　sample的位置
          f = 0;
          for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1neg[c + l2];  // －vc*uk　/ vc*uc　
          // P= -vc*uk^(1-label) *  vc*uc^(label)
          // logP＝（１-label)log(f) + (label)log(f)   f=sigmoid(vc*uk)/sigmoid(vc*uc)
          // logP/f=（１-label)/f(-x) + label/f(x)
          //       = -(1-label)/(1-f) + label/f
          // logP/x=  -(1-label)*f    + label*(1-f) = label-f   
          if (f > MAX_EXP) g = (label - 1) * alpha;             // 0/-1  uk, f too big,减小，梯度下降　　　   uc,　f too big,没关系　loss=0
          else if (f < -MAX_EXP) g = (label - 0) * alpha;      //  1/0　　uk, f 特别小且 f<0 ,没关系　loss=0  uc, f特别小还是-的，尽可能沿着梯度上升
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha; // g=label-f
          // vc*uk
          // input vector vc 's g,store in neu1e  g*uk[j]
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
          // output vector's uk's g, 实时更新　g*vc[j]
          for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * neu1[c];
        }
        // cbow　该窗内中心词固定时,更新input vector v1,v2,...vm
        // neu1e:窗内所有context word对应的平均input vector　vc的g
        // neu1e:neg 累加了以该word为中心，所有n_negtive对应的g. 包括和自己uc.　output vector是实时更新的，每抽一个正/负样本更新一次
        //       hs  累加了root-wc这个路径上每一个分类器的分类误差造成的g.       node   vector是实时更新的，每算一个分类器更新一次 
        // vc=(v1+v2+...vm)/m
        // dvm=dvc*m  (但是这里是按相同算的，因为neu1e梯度在量级上也累加了许多二分类器，负例)　该过程中node vector一般更新一次　　
        // 为了量级相同，每个input vector更新同vec
        for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
          c = sentence_position - window + a;
          if (c < 0) continue;
          if (c >= sentence_length) continue;
          last_word = sen[c];
          if (last_word == -1) continue;
          for (c = 0; c < layer1_size; c++) syn0[c + last_word * layer1_size] += neu1e[c];
        }
      }

    } else {  //train skip-gram
      // skip-gram　　　该窗内中心词vc固定时,每一步算一个input vector v1,v2,...vm. 这个词vm结束时，立即更新
        // 每一对（vc,vm）
        // neu1e:input vector vm的g
        // neu1e:neg 累加了以该word为中心，所有n_negtive对应的g. 包括uc.　     output vector是实时更新的，每抽一个正/负样本更新一次 (更新uk,uc)
        //       hs  累加了root-wc这个路径上每一个分类器的分类误差造成的g.       node   vector是实时更新的，每算一个分类器更新一次   (更新node[i]
        // 每个input vector vm在该窗内该对vector结束时，才更新。算该窗内的下一对(vc,vm+1)
        // hs: 本来是P(um|vc)　,现在对称的反过来:P(uc｜vm) 　每一层更新node
        // neg:本来是P(uc|vc).P(uk|vc) 现在对称的反过来:P(uc|vm).P(uk|vm)  每一个正/负例更新uk/uc　正例只有一个：uc    wm自身也是反例
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        // 该随机窗内的每个context词last_word和中心词一起。实时更新
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];             // last_word:此刻的context word_idx vm
        if (last_word == -1) continue;
        l1 = last_word * layer1_size;   // 该词在input vector中的位置 m
        for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
        // HIERARCHICAL SOFTMAX
        if (hs) 
         // 预测context词，P(context|vc)  : node[i]*vc  leaf是wc  这里也变成了P(vc|context)
        for (d = 0; d < vocab[word].codelen; d++) {
          f = 0;
          l2 = vocab[word].point[d] * layer1_size;　　　//从root开始，第d个分类器的node节点位置
          // Propagate hidden -> output
          for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1[c + l2];　　// input vector vm * node[d]  (syn0,syn1)
          if (f <= -MAX_EXP) continue;
          else if (f >= MAX_EXP) continue;
          else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];// sigmoid(f)
          // 本来应该是P(vm|vc), node[vm路径]*vc 但考虑对称性，训练vc的node向量，改为P(vc|vm)= sigmoid(vm*node[0])  sigmoid(vm*node[i])  sigmoid(vm*node[3])
          // P=1-sigmoid(point[i],vm) code[i]==1/  sigmoid(point[i],vm)  code[i]==0    将0(左边)作为正确
          // P= (1-f)^(code[i])  *  f^(1-code[i])
          //  logP= code[i] log(1-f)  + (1-code[i])*log(f)
          //  logP/f=code[i] * 1/(1-f)  + (1-code[i])* (1/f)
          //  f/x= f(1-f)
          // logpP/x= code[i]*f + (1-code[i])*(1-f)= 1-code[i]-f    (logP/(vm*node[i])
          // 'g' is the gradient multiplied by the learning rate
          g = (1 - vocab[word].code[d] - f) * alpha;
          // 每个context input vector vm==l1的梯度 g*node[l2][j] vm的梯度先存起来。在处理vc和下一个context　word之前，更新vm的梯度
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
          // 对每个node p[i]的每个元素p[i][j]实时更新,梯度是  g*vm[j]    
          // node vector syn1[l2] 每个元素c根据梯度更新  (point[i]==l2,。梯度上升)　node向量实时更新
          for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * syn0[c + l1];
        }
        // NEGATIVE SAMPLING
        // 对当前vc和vm,每一个neg sample
        // 原来是Ｐ(vm|vc),现在也改成P(vc|vm)  : sigmoid(uc｜vm)   sigmoid(uk|vm)
        if (negative > 0) for (d = 0; d < negative + 1; d++) {
          if (d == 0) {
            target = word;
            label = 1;
          } else {
            next_random = next_random * (unsigned long long)25214903917 + 11;
            target = table[(next_random >> 16) % table_size];
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == word) continue;
            label = 0;
          }
          l2 = target * layer1_size; // l2: uk/uc
          f = 0;
          for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l2]; //  －vm*uk　/ vm*uc　
          // P= -vm*uk^(1-label) *  vm*uc^(label)
          // logP＝（１-label)log(f) + (label)log(f)   f=sigmoid(vm*uk)/sigmoid(vm*uc)
          // logP/f=（１-label)/f(-x) + label/f(x)
          //       = -(1-label)/(1-f) + label/f
          // logP/x=  -(1-label)*f    + label*(1-f) = label-f   
          if (f > MAX_EXP) g = (label - 1) * alpha;
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;

          // input vector vm的梯度，存起来.每个元素梯度为g*uk[j]
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
          // output vector uk/uc, 每个元素梯度为g*vm[j]，实时更新output vector
          for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * syn0[c + l1];
        }
        // Learn weights input -> hidden
        // 在处理vc和下一个context　wordvm+1之前，更新vm的梯度
        // 此时对于vm.vc  
        //  hs:  vm的梯度累积了所有从root到leaf的分类器对vm的梯度
        //  neg:　vm的梯度累积了所有对vm来说是neg_sample的梯度　（包括正例uc贡献的梯度）对vm,除了uc,都是负例
        for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];
      }
    }
    sentence_position++;//下一个中心词在句子中位置
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }
  }
  //所有轮次结束。关掉train文件，结束该线程
  fclose(fi);
  free(neu1);
  free(neu1e);
  pthread_exit(NULL);
}

void TrainModel() {
  long a, b, c, d;
  FILE *fo;
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));// pthread_t类型的数组，用来标识thread id
  printf("Starting training using file %s\n", train_file);
  starting_alpha = alpha;                                              // 0.025 skip-gram/  0.05 cbow
  // 按词频排序的vocab(扔掉min_count,最长100char)/ word_dict(hasn_table)/file_size：train_file文件字节数/ train_words:所有现在留在字典中的word出现的总次数
  // .word,.cont .code/.point 指向一片char[40]/int[40]的内存 
  if (read_vocab_file[0] != 0) ReadVocab();          //事先提供了字典,从read_vocab_file中读出所有单词＋词频                  
  else LearnVocabFromTrainFile();                    //用train_file 初始化字典      sort vocab重新realloc了Ｖ+1的空间。　vocab[V].cn初始不确定，或者是某个dfrop word的cn          

  if (save_vocab_file[0] != 0) SaveVocab();          // 保存字典　w cn\n
  if (output_file[0] == 0) return;
  InitNet();　　　　　　　　　　　　　　　　　　　　　　　　　//　建立input vector [V,h], 并用next_random随机初始化
                                                      // hs的binary tree以及通过这个得到的路径信息code(0/1),point(node idx)，root to leaf)/node vector[V-1,h],但实际用了V个 
                                                      // neg sample需要的output vector [V,h]
  if (negative > 0) InitUnigramTable();               // 根据出现频率建立随机抽样table，可按概率的(3/4)返回word_idx,用来抽取负样本
  start = clock();                                   // 计时，相当于time.time()
  for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a); //创建子线程。TrainModelThread:子线程函数名(void*)a：该线程id
  for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);　　　　　　　　　　　　　　　　　　　　//pthread_join后，主线程TrainModel会一直到等待的线程结束自己才结束，使创建的线程有机会执行
  //输出结果
  fo = fopen(output_file, "wb");
  if (classes == 0) {
    // Save the word vectors　syn0
    fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);//首行"V h\n"
    for (a = 0; a < vocab_size; a++) {
      fprintf(fo, "%s ", vocab[a].word);　//剩下的每行，首先是单词string。然后写入向量的每个元素double　ah+b 最后\n
      //size_t fwrite ( const void * ptr, size_t size, size_t count, FILE * stream );
			// ptr:读入的起始地址
			//　size:要写入的每个元素字节数，Size in bytes
			// count:读入的元素数目
			// return:The total number of elements successfully written 
      if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
      else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn0[a * layer1_size + b]);　　　　　　　　　　　　//第a个词向量的第b个元素　ah+b \n
      fprintf(fo, "\n");
    }
  } else {
    // Run K-means on the word vectors
    int clcn = classes, iter = 10, closeid;
    int *centcn = (int *)malloc(classes * sizeof(int));
    int *cl = (int *)calloc(vocab_size, sizeof(int));
    real closev, x;
    real *cent = (real *)calloc(classes * layer1_size, sizeof(real));
    for (a = 0; a < vocab_size; a++) cl[a] = a % clcn;
    for (a = 0; a < iter; a++) {
      for (b = 0; b < clcn * layer1_size; b++) cent[b] = 0;
      for (b = 0; b < clcn; b++) centcn[b] = 1;
      for (c = 0; c < vocab_size; c++) {
        for (d = 0; d < layer1_size; d++) cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];
        centcn[cl[c]]++;
      }
      for (b = 0; b < clcn; b++) {
        closev = 0;
        for (c = 0; c < layer1_size; c++) {
          cent[layer1_size * b + c] /= centcn[b];
          closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];
        }
        closev = sqrt(closev);
        for (c = 0; c < layer1_size; c++) cent[layer1_size * b + c] /= closev;
      }
      for (c = 0; c < vocab_size; c++) {
        closev = -10;
        closeid = 0;
        for (d = 0; d < clcn; d++) {
          x = 0;
          for (b = 0; b < layer1_size; b++) x += cent[layer1_size * d + b] * syn0[c * layer1_size + b];
          if (x > closev) {
            closev = x;
            closeid = d;
          }
        }
        cl[c] = closeid;
      }
    }
    // Save the K-means classes
    for (a = 0; a < vocab_size; a++) fprintf(fo, "%s %d\n", vocab[a].word, cl[a]);
    free(centcn);
    free(cent);
    free(cl);
  }
  fclose(fo);
}
//字符串常量类型　：char *
int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++)     //每一个参数/参数值
  if (!strcmp(str, argv[a])) {  // 如果和str匹配　　　　（equal，返回0.　/ first bigger, return >0）
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);                  // 如果匹配了但已经是最后一个参数了，　说明某个参数给了命令但没有赋值，进程退出
    }
    return a;                   // argv[i]: 第i个命令行参数, 和str (char *)"-output"之类匹配, 返回该命令行参数位置
  }
  return -1;                    //没有该命令行参数
}

int main(int argc, char **argv) {
  int i;
  if (argc == 1) {
    printf("WORD VECTOR estimation toolkit v 0.1c\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");　　　　　　　　　　　　　　　　　　　　　//输入文件：已分词的语料
    printf("\t\tUse text data from <file> to train the model\n"); 
    printf("\t-output <file>\n");　　　　　　　　　　　　　　　　　　　　//输出的vector文件
    printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
    printf("\t-size <int>\n");　　　　　　　　　　　　　　　　　　　　　　//词向量的维度，默认值是100
    printf("\t\tSet size of word vectors; default is 100\n");
    printf("\t-window <int>\n");　　　　　　　　　　　　　　　　　　　　　//窗口大小，默认是5
    printf("\t\tSet max skip length between words; default is 5\n");
    printf("\t-sample <float>\n");　　　　　　　　　　　　　　　　　　　　// random sample，对高频词降采样　（同MS_baseline）.1e-3
    printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
    printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
    printf("\t-hs <int>\n");                                      // -hs 是否会分层softmax 1 (default=0,neg sample)
    printf("\t\tUse Hierarchical Softmax; default is 0 (not used)\n");
    printf("\t-negative <int>\n");                               // number of negtive samples:5 (3-10)  0不用
    printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
    printf("\t-threads <int>\n");                                // 线程数目
    printf("\t\tUse <int> threads (default 12)\n");
    printf("\t-iter <int>\n");　　　　　　　　　　　　　　　　　　　　　// epoch　　default=5
    printf("\t\tRun more training iterations (default 5)\n");
    printf("\t-min-count <int>\n");                             // discard min_count in Vocab  default=5
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-alpha <float>\n");                               // alpha:0.025/0.05
    printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n");
    printf("\t-classes <int>\n");                                //输出词类别，而不是词向量. 0:输出向量
    printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
    printf("\t-binary <int>\n");　　　　　　　　　　　　　　　　　　　// 0:非二进制输出
    printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
    printf("\t-save-vocab <file>\n");                          // vocab保存位置
    printf("\t\tThe vocabulary will be saved to <file>\n");
    printf("\t-read-vocab <file>\n");                          // 提供vocab的文件
    printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
    printf("\t-cbow <int>\n");　　　　　　　　　　　　　　　　　　　　// skip-gram(0) default=cbow=1
    printf("\t\tUse the continuous bag of words model; default is 1 (use 0 for skip-gram model)\n");
    printf("\nExamples:\n");
    printf("./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3\n\n");
    return 0;
  }
  //字符数组　char[100]
  output_file[0] = 0;
  save_vocab_file[0] = 0;
  read_vocab_file[0] = 0;
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);             //文件名： char *, const char *
  if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
  if (cbow) alpha = 0.05;
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);                // 1e-3
  if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);            // 3 - 10
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);             // 0
  vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));                // vocab数组，每个元素是一个结构体对象vocab_word。初始为1000个元素，全为0
  vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));　　　　　　　　　　　　　　　　　　　　　　　// vocab_hash数组，每个元素是一个整数。初始为30000000个元素，用来hash不同word.初始全为0
  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));　　　　　　　　　　　　　　　　　　　//  expTable,每个元素是一个实数，初始为1000+1个元素，全随机
  // expTable[(f+6)*(1000/12)]==1/(1+exp(-f))
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table    MAX_EXP:6   expTable[i] = exp((i -500)/ 500 * 6)    e^[(499/500)*6]
    // 粒度：　e^6 * e^(1/500)  [e^-6 - e^6]
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)     e-6/(e-6+1)~0     e6/(e6+1)~1   x: e^x  
                                                                     // e[i]=ez/(1+ez)=1/(1+e-z)    -z[i]:(12i/1000-6)
                                                                     //  i的范围（0,1000）, z=12i/1000-6的范围为（-6,6）
                                                                     // i=(f+6)*(1000/12)
                                                                     // -z=((f+6)/6-1)*6=f
                                                                     // ez= exp(f)
                                                                     // output: exp(f)/(1+exp(f))=1/(1+exp(-f))
  }
  TrainModel();
  return 0;
}
