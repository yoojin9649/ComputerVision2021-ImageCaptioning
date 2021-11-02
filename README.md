### 개요

본 챌린지는 Vision-Language Task 중에서 이미지 캡셔닝을 수행하는 챌린지입니다. 이미지 캡셔닝은 아래와 같이 이미지의 내용을 설명하는 문장을 생성하는 작업입니다. 

![Learning CNN-LSTM Architectures for Image Caption Generation](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSJXd5zruE1QWsirEpjPMqB9CI7SmpA94LrDnYm2QLJujxrGuMRQunT4Gg7Pr4kEuBr2Q&usqp=CAU)

이미지 캡셔닝 기술을 통해 시력이 낮거나 없는 사람들에게 사진에 대한 설명을 제공할 수 있고 사진에 설명을 추가하여 검색 효율을 높일 수 있습니다.



### 데이터셋

이미지 캡셔닝에서 사용하는 데이터셋은 이미지와 텍스트로 이루어져있고 보통 한 이미지에 대해 5개의 문장으로 구성되어 있습니다. 아래는 MSCOCO 데이터의 예시이며 여기서는 이미지 당 한 문장씩 보여줍니다.

![COCO dataset](https://blog.kakaocdn.net/dn/cew1mu/btqwWy7p8Jh/wmSGLJ9GngjKxCVydUgEKK/img.png)



**이미지**

이미지 데이터는 [cocodataset](https://cocodataset.org/#download) 에서 받을 수 있습니다. 본 챌린지에서는  2014 Train images 와 2014 Val images 를 사용합니다.



**텍스트**

텍스트 데이터는 "Andrej Karpathy's training, validation, and test splits 를 사용하고 [standford](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip) 에서 다운 받으실 수 있습니다. 다운 받으신 후 압축을 풀면 3개의 json 파일이 있을텐데 본 챌린지에서는 `dataset_coco.json`을 사용합니다.

텍스트 데이터는 json 구조이며 이미지에 대한 경로, 캡션 등으로 이루어져있습니다.

```
[{'filepath': 'val2014',
  'sentids': [681330, 686718, 688839, 693159, 693204],
  'filename': 'COCO_val2014_000000522418.jpg',
  'imgid': 1,
  'split': 'restval',
  'sentences': [{'tokens': ['a',
     'woman',
     'wearing',
     'a',
     'net',
     'on',
     'her',
     'head',
     'cutting',
     'a',
     'cake'],
    'raw': 'A woman wearing a net on her head cutting a cake. ',
    'imgid': 1,
    'sentid': 681330},
    ...]
```

splits 는 'train', 'restval', 'val', 'test'로 이루어져 있으며 'train', 'restval' splits를 사용해 모델 훈련을 하고 'val' splits를 사용해 검증을 합니다. 마지막으로 'test' splits를 사용해 성능을 평가합니다.

- train(train+restval): 113287
- val: 5000
- test: 5000

데이터에 대한 정보가 궁금하신 분들은 아래 사이트에서 확인하면 도움이 될 것 같습니다.

- https://www.tensorflow.org/datasets/catalog/coco_captions

- https://cs.stanford.edu/people/karpathy/deepimagesent/



### 베이스라인

본 챌린지에서 사용한 베이스 라인 방법론은 Convolutional Image Captioning 논문입니다.

- 깃허브 주소: https://github.com/aditya12agd5/convcap



### 동영상

챌린지 설명, 논문의 방법론, 베이스라인 코드 설명하는 영상의 주소는 다음과 같습니다.

- 유튜브: https://www.youtube.com/watch?v=gfzjD70osvE



### 추천하는 논문

- Convolutional sequence to sequence learning.

  베이스라인 모델은 위 논문에서 사용된 convolutional 기계 번역 모델을 기반으로 했습니다.

- Deep visual-semantic alignments for generating image descriptions.

  베이스라인과 다르게 기존의 이미지 캡셔닝 모델은 LSTM을 사용했습니다. 위 논문이 대표적으로 LSTM 기반 이미지 캡셔닝 모델을 사용했습니다.
  
  
  
### 제출 파일 형식

제출할 파일의 확장자는 `.json` 입니다. 제출할 파일의 형식은 다음과 같고 형식을 맞춰줘야 평가가 가능합니다.

```
[{"image_id": 391895, "caption": "a man is standing on a bike in the woods"}, {"image_id": 60623, "caption": "a woman is eating a piece of cake"},
...]
```


### 평가지표
이미지 캡셔닝에서 자주 사용하는 평가지표로 **Bleu_1, Bleu_2, Bleu_3, Bleu_4, METEOR, ROUGE_L, CIDEr, SPICE** 를 사용합니다. 점수가 높을수록 성능이 좋다는 것을 의미합니다.
