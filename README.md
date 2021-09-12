# understanding-reviews

this is the official implementation of "Understanding the Effectiveness of Reviews in E-commerce Top-N Recommendation" in proceedings of ICTIR 2021. If you use our code, please cite our paper:
##### Zhichao Xu, Hansi Zeng, and Qingyao Ai. 2021. Understanding the Effectiveness of Reviews in E-commerce Top-N Recommendation. In Proceedings of the 2021 ACM SIGIR International Conference on Theory of Information Retrieval (ICTIR '21). Association for Computing Machinery, New York, NY, USA, 149–155. DOI:https://doi.org/10.1145/3471158.3472258

Requirements:</br>
python 3.6+
PyTorch 1.4.0</br>
Scikit-learn 0.23.2</br>

##### Run matrix factorization:
run preprocess.py</br>
run train.py</br>
run rerank.py to create ranklist_with_gt.json</br>
run train.py to rerank</br>

##### Run deepconn:
