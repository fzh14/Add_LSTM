import gensim
import numpy as np
def extract():
	count=0
	sentence=[]
	with open('output_dataset.txt') as f:
		for i in f: 
			count+=1
			line = i.split('|')
			list = line[0].split(' ')
			sentence.append(list)
			if count == 100:
				print list
		
	model = gensim.models.Word2Vec(sentence, min_count=1)			
	model.save("output")
	new_model = gensim.models.Word2Vec.load('output')

	a = new_model['feel']
	print a

if __name__ == '__main__':
	extract()
