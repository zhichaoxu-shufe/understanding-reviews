# understanding-reviews

this is the official implementation of "Understanding the Effectiveness of Reviews in E-commerce Top-N Recommendation" in proceedings of ICTIR 2021. If you use our code, please cite our paper:
```bibtex
@inproceedings{xu2021understanding,
  title={Understanding the Effectiveness of Reviews in E-commerce Top-N Recommendation},
  author={Xu, Zhichao and Zeng, Hansi and Ai, Qingyao},
  booktitle={Proceedings of the 2021 ACM SIGIR International Conference on Theory of Information Retrieval},
  pages={149--155},
  year={2021}
}
```

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
```bash
python preprocess.py
python train.py
python rerank.py to create ranklist_with_gt.json
python train.py to rerank
python evaluate.py to calculate hit rate and ndcg
```

##### Run deepconn:
```bash
python divide_and_create_example_doc.py
python train.py
python evaluate.py
```
