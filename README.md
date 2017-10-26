# Similar-words-searcher-by-Word2Vec
## Introduction

 - It's a similar words searcher by Word2Vec model made with tensorflow

用TensorFlow做的一个基于Word2Vec模型的一个词语训练、储存、读取和近似特征词搜寻的python code。

 - It's a python code that allow trainning text, saving/read model and use it for searching approximate characteristic words, which based on Word2Vec model, and made by TensorFlow.

 - It's hyper parameter is:

 ![pic](https://github.com/AdamAlive/MarkdownRef/blob/master/189.jpg?raw=true )
 
 - When it runs, it shows some words randomly, and show them all along the trainning.

 ![pic](https://github.com/AdamAlive/MarkdownRef/blob/master/190.jpg?raw=true )

 ![pic](https://github.com/AdamAlive/MarkdownRef/blob/master/192.jpg?raw=true )

 ![pic](https://github.com/AdamAlive/MarkdownRef/blob/master/213.jpg?raw=true )

 - Then, after it trained, we can turn on the TO_JUDGE switch and search.

 ![pic](https://github.com/AdamAlive/MarkdownRef/blob/master/212.jpg?raw=true )
 
 - By the way, we can also get a picture about the words' distribution projected to 2 dim world.

 ![pic](https://github.com/AdamAlive/MarkdownRef/blob/master/202.png?raw=true )

*********************

## How to use

 - Just run the python code `` python Simarliy_Searcher_byWord2Vec.py `` and if you haven't download the dataset, it will download automatically. Train it and switch on `` TO_SAVE ``, Then run again with `` TO_JUDGE `` on, then you can input words to find its similar words.

- I also put the pretrained file in the folder. Unzip it and put it in the path that the hyperparameter `` OUTPUT_PATH `` is, and then you can just start to JUDGE.
