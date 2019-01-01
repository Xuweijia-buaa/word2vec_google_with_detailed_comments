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
#include <malloc.h>
#include <ctype.h>

const long long max_size = 2000;         // max length of strings
const long long N = 1;                   // number of closest words
const long long max_w = 50;              // max length of vocabulary entries
//   ./compute-accuracy vectors.bin 30000  < questions-words.txt
//   ./compute-accuracy vectors.bin < questions-words.txt                  (全部词　threshold==0))
//   ./compute-accuracy vectors-phrase.bin < questions-phrases.txt

//argc:输入的命令行参数个数　int  可执行程序程序本身的文件名，也算一个命令行参数，因此，argc 的值至少是1.   transe -size 100 -thread 8  : argc=5
//argv 是一个数组(指针数组)，其中的每个元素都是一个char* 类型的指针，argv[i]指针指向一个字符串，这个字符串里就存放着命令行参数  
//argv[0]:char *,指向该transe.exe 本身的文件名transe
int main(int argc, char **argv)
{
  FILE *f;
  char st1[max_size], st2[max_size], st3[max_size], st4[max_size], bestw[N][max_size], file_name[max_size], ch;
  float dist, len, bestd[N], vec[max_size];
  long long words, size, a, b, c, d, b1, b2, b3, threshold = 0;
  float *M;
  char *vocab;
  int TCN, CCN = 0, TACN = 0, CACN = 0, SECN = 0, SYCN = 0, SEAC = 0, SYAC = 0, QID = 0, TQ = 0, TQS = 0;
  // 至少２个参数，脚本和文件名
  if (argc < 2) {
    printf("Usage: ./compute-accuracy <FILE> <threshold>\nwhere FILE contains word projections, and threshold is used to reduce vocabulary of the model for fast approximate evaluation (0 = off, otherwise typical value is 30000)\n");
    return 0;
  }
  strcpy(file_name, argv[1]);             　　　// vectors.bin / vectors-phrase.bin, 存放向量的文件
  　　　　　　　　　　　　　　　　　　　　　　　　　　　// 首行"V size\n"  剩下每行:"单词string　0.01 0.03 0.05 \n".词顺序都按出现频率从高到低排好了
  if (argc > 2) threshold = atoi(argv[2]);     // 如果有第三个命令行参数，　即给定的throshold,　　 reduce vocabulary
  f = fopen(file_name, "rb");
  if (f == NULL) {
    printf("Input file not found\n");
    return -1;
  }
  fscanf(f, "%lld", &words);                  // words:V    
  if (threshold) if (words > threshold) words = threshold;  // 限制单词数为throshold
  fscanf(f, "%lld", &size);　　　　　　　　　　  // h

  vocab = (char *)malloc(words * max_w * sizeof(char));　　// V*max_w　char      vocab[i * max_w + j]:第i个单词的第j个元素char
  M = (float *)malloc(words * size * sizeof(float));      // V*h      float　　　M[i * size+ j]      :第i个单词vector的第j个元素
  if (M == NULL) {
    printf("Cannot allocate memory: %lld MB\n", words * size * sizeof(float) / 1048576);
    return -1;
  }
  for (b = 0; b < words; b++) {　　　//第b个词
    a = 0;
    while (1) {
      vocab[b * max_w + a] = fgetc(f);　　　　　　　　　　　　　//　第b个词的第a个字符
      if (feof(f) || (vocab[b * max_w + a] == ' ')) break;  // 读取一个字符，直到碰到空格，如果还没到max_w,结束。读完了完整word
      if ((a < max_w) && (vocab[b * max_w + a] != '\n')) a++;　//　一直读到a==max_w-1还没读完  即vocab[b]中超过了max_w个字符。　
      　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　//  a不变：a==max_w-1，一直把剩下的word从f里读完,且都读到vocab[b]最后一个位置
    }
    vocab[b * max_w + a] = 0;　　　　　　　　　　　　　　　　　　　　// 直到空格前的都读完了，把vocab[b]截断，截为max_w或者真实长度（如果小于max_w）　.包括最后一个空字符  
                                                              // word　b字符数组：vocab[bw]-vocab[bw+max_w]
    for (a = 0; a < max_w; a++) vocab[b * max_w + a] = toupper(vocab[b * max_w + a]);  // vocab[b]全部大写
    // size:要读取的每个元素的大小，以字节为单位。nmemb:元素的个数
    for (a = 0; a < size; a++) fread(&M[a + b * size], sizeof(float), 1, f);           // 单词b对应的vector的每个float,每次读一个，读h次
    len = 0;
    for (a = 0; a < size; a++) len += M[a + b * size] * M[a + b * size];              　// 该向量平方和
    len = sqrt(len);　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　// 模长
    for (a = 0; a < size; a++) M[a + b * size] /= len;　　　　　　　　　　　　　　　　　　　　 // 向量每个元素，全部用二范数归一化
  }
  fclose(f);
  TCN = 0;　// 每一part,该part命中的词组数目
  while (1) {
    // N=1
    // bestd[1]   bestw[1][h]
    // # < 表示的是输入重定向的意思，就是把<后面跟的文件取代键盘作为新的输入设备。不作为命令行参数.因此之后的输入是question-word.txt 文件
    // phrase
    // : newspapers
    // Albuquerque Albuquerque_Journal Baltimore Baltimore_Sun

    // word
    //: capital-common-countries
    // Athens Greece Baghdad Iraq
    for (a = 0; a < N; a++) bestd[a] = 0;
    for (a = 0; a < N; a++) bestw[a][0] = 0;
    scanf("%s", st1);
    for (a = 0; a < strlen(st1); a++) st1[a] = toupper(st1[a]);　　　　　　//读到的该word全大写

    if ((!strcmp(st1, ":")) || (!strcmp(st1, "EXIT")) || feof(stdin)) {  //  结束/读到冒号　　　QID+1, TCN,CCN 归0,到下一part
      if (TCN == 0) TCN = 1;                                             //　 如果上一part TCN=＝0. 给TCN赋１
      if (QID != 0) {
        printf("ACCURACY TOP1: %.2f %%  (%d / %d)\n", CCN / (float)TCN * 100, CCN, TCN);　　// 每一个小部分,命中的acc
        //　存在的词组中　，预测正确的/总的　　不分sem,sys  /   　 SEAC / SECN　　　/   SYAC / SYCN
        printf("Total accuracy: %.2f %%   Semantic accuracy: %.2f %%   Syntactic accuracy: %.2f %% \n", CACN / (float)TACN * 100, SEAC / (float)SECN * 100, SYAC / (float)SYCN * 100);
      }
      QID++;
      scanf("%s", st1);　　　　　　　　　　　　　　　　　　　　　　　　　　　　　// 只要没读完，将下一个读到的字符给st1
      if (feof(stdin)) break;
      printf("%s:\n", st1);
      TCN = 0;　　　　　　　　　　　　　　　　　　　　　　　　　　
      CCN = 0;
      continue;
    }
    if (!strcmp(st1, "EXIT")) break;　　　　　　　　　　　　　　　　　　　　　　// 如果都没有读到
    scanf("%s", st2);　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　// st2,st3,st4都读入，且变大写
    for (a = 0; a < strlen(st2); a++) st2[a] = toupper(st2[a]);
    scanf("%s", st3);
    for (a = 0; a<strlen(st3); a++) st3[a] = toupper(st3[a]);
    scanf("%s", st4);
    for (a = 0; a < strlen(st4); a++) st4[a] = toupper(st4[a]);

    for (b = 0; b < words; b++) if (!strcmp(&vocab[b * max_w], st1)) break;　　// 看原始单词集合中，哪个可以match str1（都大写了）。把该单词index给b1
    b1 = b;
    for (b = 0; b < words; b++) if (!strcmp(&vocab[b * max_w], st2)) break;   // 看原始单词集合中，哪个可以match str2（都大写了）。把该单词index给b2
    b2 = b;
    for (b = 0; b < words; b++) if (!strcmp(&vocab[b * max_w], st3)) break;   // 看原始单词集合中，哪个可以match str3（都大写了）。把该单词index给b3
    b3 = b;

     //float  bestd[1]   char bestw[1][2000]
    for (a = 0; a < N; a++) bestd[a] = 0;
    for (a = 0; a < N; a++) bestw[a][0] = 0;
    TQ++;　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 // 总共有多少组单词
    if (b1 == words) continue;                                              // 如果任一单词不在word_vec词表中，　看下一组单词
    if (b2 == words) continue;
    if (b3 == words) continue;
    // 如果该组前３个单词都在词表中
    for (b = 0; b < words; b++) if (!strcmp(&vocab[b * max_w], st4)) break;　// 第四个单词不在
    if (b == words) continue;


    // 该组所有词都在词表中  float vec[max_size]
    for (a = 0; a < size; a++) vec[a] = (M[a + b2 * size] - M[a + b1 * size]) + M[a + b3 * size];　// a-b=c-d   vec:d' ==b+c-a     M[a + b * size]:第b个单词vector的第a个元素
    TQS++;                                                                  // 有多少组单词在词表中
    for (c = 0; c < words; c++) {
      // 除了a,b,c以外的第i个单词c
      if (c == b1) continue;
      if (c == b2) continue;
      if (c == b3) continue;
      dist = 0;
      for (a = 0; a < size; a++) dist += vec[a] * M[a + c * size];　// 　dist：　d' vec[a]  和第c个单词M[c]  的cos
  
      // 与最大的值（当前最近的单词）依次比较，当cos值大于某一当前单词，该单词被替换为单词c,之后单词依次往后排
      for (a = 0; a < N; a++) {
        if (dist > bestd[a]) {　　　　　　　　　　　　　　　　　　　　　　　// 值大于当前与d'第ｉ近的单词，替换
          for (d = N - 1; d > a; d--) {　 　　　　　　　　　　　　　　　　// 单词N被单词N-1替换（包括距离）直到替换掉　best[i+1]
            bestd[d] = bestd[d - 1];　　　　　　　　　　　　　　　　　　　//　 　距离
            strcpy(bestw[d], bestw[d - 1]);                         // 　　对应字符
          }
          bestd[a] = dist;                                         //  更新best[i]的距离　　　为当前最相近的top[i] word　
          strcpy(bestw[a], &vocab[c * max_w]);　　　　　　　　　　　　　//             字符
          break;
        }
      }
    }
    // top1 和 d 是否相同
    if (!strcmp(st4, bestw[0])) {
      CCN++;　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　// 每一个小部分　总的预测正确的数目
      CACN++;　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　// 所有部分总的预测正确的数目
      if (QID <= 5) SEAC++; else SYAC++;　　　　　　　　　　　　　　　　　// 前６部分是语义，后边是语法　 SEAC:语义预测正确数目　　　　SYAC：语法预测正确数目
    }

    if (QID <= 5) SECN++; else SYCN++;　　　　　　　　　　　　　　　　　　//　在所有命中的组中，总的语义/语法数目

    TCN++;　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　//　每一个小部分，总的命中数目
    TACN++;　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　//　所有部分，总的命中数目
  }

  
  printf("Questions seen / total: %d %d   %.2f %% \n", TQS, TQ, TQS/(float)TQ*100); // TQ 总的词组数目　　　TQS：命中的词组数目　应该等于TACN
  return 0;
}
