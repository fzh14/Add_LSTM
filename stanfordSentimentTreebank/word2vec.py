import gensim
import string

model = gensim.models.KeyedVectors.load_word2vec_format('/Users/ivanfzh/Downloads/glove.6B/glove.6B.50d.txt', binary=False)
a = model['feel']

fp = open('output_dataset.txt','r')
fo = open('output_50d.txt', 'w')
count = 0
for l in fp.readlines():
    line = l.strip('\n').split('|')
    word_list = line[0].split(' ')
    s = ''
    judge = True
    for item in word_list:
        item = string.lower(item)
        try:
            '''
            li = item.split('-')
            for i in range(len(li)):
                word = li[i]
                if string.count(word,'\*') == 0:
                    if string.find(word,'\/') == -1:
                        s.append(model[word].tolist())
                    else:
                        index = string.find(word,'\/')
                        s.append(model[word[0:index]].tolist())
                        s.append(model[word[index+1]].tolist())
                        s.append(model[word[index+2:]].tolist())
                        '''
            '''li = item.split('-')
            for i in range(len(li)):
                word = li[i]
                s += str(model[word].tolist()) + ' '
                '''
            s += str(model[item].tolist())
        except:
            judge = False
         #   print item
    if judge == True:
        s = str(line[0]) + '|' + str(line[1]) + '\n'
        fo.write(s)
        count += 1

print count
print a
