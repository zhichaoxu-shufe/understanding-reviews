# understanding-reviews

this is the official implementation of "Understanding the Effectiveness of Reviews in E-commerce Top-N Recommendation" in proceedings of ICTIR 2021. If you use our code, please cite our paper:
##### Zhichao Xu, Hansi Zeng, and Qingyao Ai. 2021. Understanding the Effectiveness of Reviews in E-commerce Top-N Recommendation. In Proceedings of the 2021 ACM SIGIR International Conference on Theory of Information Retrieval (ICTIR '21). Association for Computing Machinery, New York, NY, USA, 149–155. DOI:https://doi.org/10.1145/3471158.3472258

if you need code for rating prediction, please refer to AHN official implementation: https://github.com/Moonet/AHN, ZARM official implementation https://github.com/HansiZeng/ZARM

Requirements:</br>
python 3.6+
PyTorch 1.4.0</br>
Scikit-learn 0.23.2</br>

##### Dataset:</br>
download subcategory files from http://jmcauley.ucsd.edu/data/amazon/links.html</br>
##### Pretrained Word2vec:</br>
download from https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g

##### Run matrix factorization:
run preprocess.py</br>
run train.py</br>
run rerank.py to create ranklist_with_gt.json</br>
run train.py to rerank</br>
run evaluate.py to calculate hit rate and ndcg</br>

##### Run deepconn:
run divide_and_create_example_doc.py</br>
run train.py</br>
run evaluate.py</br>

more models incoming</br>

