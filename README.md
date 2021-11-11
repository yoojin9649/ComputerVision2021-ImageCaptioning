### 환경 설정 업로드 중입니다.

## 환경 설정

1. 해당 [github](https://github.com/aditya12agd5/convcap) 에서 **Download ZIP**을 클릭하여 다운 받은 후 압축을 풉니다.
   ![image-20211108172604868](https://github.com/yoojin9649/ComputerVision2021-ImageCaptioning/raw/main/imgs/image-20211108172604868.png)

![image-20211111102018406](C:\Users\dbwls\AppData\Roaming\Typora\typora-user-images\image-20211111102018406.png)

2. 코드 실행을 위한 가상환경을 만들어 줍니다.

   ```
   conda create -n convcap_test python=3.8
   conda activate convcap_test
   ```

3. 필요한 라이브러리를 설치해 줍니다.

   - torch 설치: https://pytorch.kr/get-started/previous-versions/

     ```
     conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
     ```

   - 기타 라이브러리

     ```
     pip install cython matplotlib scikit-image tqdm numpy Pillow
     pip install jupyter
     ```

4. 본 코드는 Python2.7에서 돌아가므로 Python3 이상에서 사용하려면 고쳐야 하는 부분이 있습니다.

   - evaluate.py `print 'Using %d/%d predictions' % (len(preds_filt), len(preds))` 코드를 `print('Using %d/%d predictions' % (len(preds_filt), len(preds)))` 로 변경합니다.

5. 아래 명령어를 Terminal 창에서 실행해 데이터를 train/val/test 로 나눠줍니다.

   ```
   bash scripts/fetch_splits.sh
   ```

   `scripts/fetch_splits.sh` 를 확인하면 아래와 같이 `caption_datasets.zip` 를 받아와 data 폴더에 이동시켜 주는 것을 확인할 수 있습니다.

   ```
   wget http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
   unzip caption_datasets.zip
   mv dataset_* ./data/
   rm caption_datasets.zip
   ```

   Pycharm의 Terminal에서 명령어를 실행한 모습은 다음과 같습니다.

   ![image-20211111102845133](C:\Users\dbwls\AppData\Roaming\Typora\typora-user-images\image-20211111102845133.png)

   `data` 폴더 내부에 다음과 같은 파일들이 생긴 것을 알 수 있습니다.

   ![image-20211111102955043](C:\Users\dbwls\AppData\Roaming\Typora\typora-user-images\image-20211111102955043.png)

6. train2014, val2014 이미지 데이터를 [MSCOCO](http://cocodataset.org/#download) 웹페이지에서 다운받아 `./data/coco` 에 저장합니다.

   - 이미지 데이터를 `./data/coco` 에 저장하기 전 폴더의 모습
     ![image-20211111102955043](C:\Users\dbwls\AppData\Roaming\Typora\typora-user-images\image-20211111102955043.png)

   - 이미지 데이터를 `./data/coco` 에 저장하여 압축을 푼 후 폴더의 모습
     ![image-20211111110716013](C:\Users\dbwls\AppData\Roaming\Typora\typora-user-images\image-20211111110716013.png)

7. 결과를 저장하기 위한 `output` 폴더를 생성합니다.
       ![image-20211111103653229](C:\Users\dbwls\AppData\Roaming\Typora\typora-user-images\image-20211111103653229.png)

8. third_party 안에 coco-caption이라는 폴더가 있습니다.
   ![image-20211111111709154](C:\Users\dbwls\AppData\Roaming\Typora\typora-user-images\image-20211111111709154.png)

   `coco-caption` 폴더를 확인해보시면 빈 폴더일 것입니다. 그렇기 때문에 삭제해주시고 link 에 제가 `coco-caption-master.zip` 을 올려놨으니 다운 받으셔서 압축을 풀어주시면 됩니다.

   ![image-20211111112146913](C:\Users\dbwls\AppData\Roaming\Typora\typora-user-images\image-20211111112146913.png)

   `coco-caption-master` 폴더의 이름을 `coco_caption` 으로 변경합니다.
   ![image-20211111112231699](C:\Users\dbwls\AppData\Roaming\Typora\typora-user-images\image-20211111112231699.png)

   그리고 `coco_caption` 폴더 안을 보면 annotations 폴더가 있는데 삭제합니다.

   ![image-20211111112325609](C:\Users\dbwls\AppData\Roaming\Typora\typora-user-images\image-20211111112325609.png)

   **주의할 점!!**

   여기서 주의할 점은 다음 coco-caption 모듈은 다음 링크 https://github.com/daqingliu/coco-caption 의 조건을 만족해야 합니다. 제가 사용한 결과 python3.6 이상하면 작동하고 java도 깔려있어야 합니다. 위의 링크에서 Requirements를 확인해주세요.

   

   [MSCOCO](http://images.cocodataset.org/annotations/annotations_trainval2014.zip) 웹페이지에서 *2014 Train/Val annotations [241MB]* 를 다운 받아 `third_party/coco_caption` 경로 안에 넣고 압축을 풀어줍니다.

   - *2014 Train/Val annotations [241MB]* 클릭
     ![image](https://user-images.githubusercontent.com/59722489/140843939-23ea5a62-ac48-419a-b371-d37a36fd3452.png)
   - 압축 푼 모습

   ![image-20211111112445272](C:\Users\dbwls\AppData\Roaming\Typora\typora-user-images\image-20211111112445272.png)

9. evaluate에서 아래 코드를

   ```
   from pycocotools.coco import COCO
   from pycocoevalcap.eval import COCOEvalCap
   ```

   다음 코드로 변경합니다.

   ```
   from third_party.coco_caption.pycocotools.coco import COCO
   from third_party.coco_caption.pycocoevalcap.eval import COCOEvalCap
   ```

10. 코드를 또 수정합니다.

    - coco_loader.py의 코드를 바꿔줍니다. `words = str(caption).lower().translate(None, string.punctuation).strip().split()` 코드를 `words = str(caption).lower().translate(str.maketrans('', '', string.punctuation)).strip().split()` 로 바꿔줍니다.

    - evaluate.py에서 `preds_filt = [p for p in preds if p['image_id'] in valids]` 코드 이후에 아래의 코드를 추가합니다.

      ```
        len_p = len(preds_filt)
        for i in range(len_p):
          preds_filt[i]['image_id'] = int(preds_filt[i]['image_id'])
      ```

      아래 사진은 코드를 추가한 모습입니다.
      ![image-20211111112935204](C:\Users\dbwls\AppData\Roaming\Typora\typora-user-images\image-20211111112935204.png)

    - `evaluate.py` 에서 `annFile = 'third_party/coco-caption/annotations/captions_val2014.json'` 코드를 `annFile = 'third_party/coco_caption/annotations/captions_val2014.json'` 로 변경합니다.

11. `third_party/coco_caption` 폴더 안의 `get_standford_models.sh`를 더블클릭합니다.
    ![image-20211111113039608](C:\Users\dbwls\AppData\Roaming\Typora\typora-user-images\image-20211111113039608.png)

    ![image-20211111113119521](C:\Users\dbwls\AppData\Roaming\Typora\typora-user-images\image-20211111113119521.png)

    이런식으로 쭉 실행되다가 완료되면 꺼집니다.

12. 제가 올린 2개의 파일을 살펴봅니다.

- `convcap train&test (ComputerVision).ipynb` : coco 데이터 셋에 대해 훈련하고 테스트하는 파일입니다. 환경만 잘 설정하셨다면 실행할 때 문제는 없을 것입니다.
- `convcap new image test (ComputerVision).ipynb` : 훈련한 모델을 새로운 이미지를 사용해 테스트하는 파일입니다. 저는 `my_image` 폴더를 생성하여 10개의 이미지에 대해 테스트를 진행했습니다.



13. 테스트 결과를 살펴봅니다.
![image1](https://user-images.githubusercontent.com/59722489/140758437-0520130b-7a32-44b2-84db-b14bf9b4a375.jpg)

a group of people standing on top of a mountain

![image2](https://user-images.githubusercontent.com/59722489/140758440-27c8558d-1579-427b-8829-e7a9165763c0.jpg)

a cat sitting on a ground next to a dead tree

![image3](https://user-images.githubusercontent.com/59722489/140758444-fd4d78cc-cfb3-4d99-ba74-047270ebaa47.jpg)

a group of people sitting around a table with a laptop

![image4](https://user-images.githubusercontent.com/59722489/140758446-aff3c8f4-9d6f-45a6-be20-edb3eeb5ac8a.jpg)

a large building with a large building in the background

![image5](https://user-images.githubusercontent.com/59722489/140758452-e0054f52-2e05-447a-872d-f2947ec55539.jpg)
a plate of food with a fork and a fork

![image6](https://user-images.githubusercontent.com/59722489/140758454-9fbde970-7f74-424b-9b89-b1f1265ae42c.jpg)

a table with a bunch of different types of items

![image7](https://user-images.githubusercontent.com/59722489/140758456-2e746a79-f56f-400d-9930-8a446710d065.jpg)

a busy city street with cars and cars

![image8](https://user-images.githubusercontent.com/59722489/140758459-0dc9159b-34d2-4335-aabf-003c2e900355.jpg)

two brown bears are sitting on a tree

![image9](https://user-images.githubusercontent.com/59722489/140758460-e7aa8aab-620e-40f4-a217-2be36c1fb22f.jpg)

a man is standing next to a large elephant

![image10](https://user-images.githubusercontent.com/59722489/140758462-1e09364f-a518-48d9-8966-65e7a9a7a856.jpg)

a man is standing next to a large elephant



==============================================================================================



## 개요

본 챌린지는 Vision-Language Task 중에서 이미지 캡셔닝을 수행하는 챌린지입니다. 이미지 캡셔닝은 아래와 같이 이미지의 내용을 설명하는 문장을 생성하는 작업입니다. 

![Learning CNN-LSTM Architectures for Image Caption Generation](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSJXd5zruE1QWsirEpjPMqB9CI7SmpA94LrDnYm2QLJujxrGuMRQunT4Gg7Pr4kEuBr2Q&usqp=CAU)

이미지 캡셔닝 기술을 통해 시력이 낮거나 없는 사람들에게 사진에 대한 설명을 제공할 수 있고 사진에 설명을 추가하여 검색 효율을 높일 수 있습니다.



## 데이터셋

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



## 베이스라인

본 챌린지에서 사용한 베이스 라인 방법론은 Convolutional Image Captioning 논문입니다.

- 깃허브 주소: https://github.com/aditya12agd5/convcap



## 동영상

챌린지 설명, 논문의 방법론, 베이스라인 코드 설명하는 영상의 주소는 다음과 같습니다.

- 유튜브: https://www.youtube.com/watch?v=gfzjD70osvE



## 추천하는 논문

- Convolutional sequence to sequence learning.

  베이스라인 모델은 위 논문에서 사용된 convolutional 기계 번역 모델을 기반으로 했습니다.

- Deep visual-semantic alignments for generating image descriptions.

  베이스라인과 다르게 기존의 이미지 캡셔닝 모델은 LSTM을 사용했습니다. 위 논문이 대표적으로 LSTM 기반 이미지 캡셔닝 모델을 사용했습니다.
  
  
  
  
## 제출 파일 형식

제출할 파일의 확장자는 `.json` 입니다. 제출할 파일의 형식은 다음과 같고 형식을 맞춰줘야 평가가 가능합니다.

```
[{"image_id": 391895, "caption": "a man is standing on a bike in the woods"}, {"image_id": 60623, "caption": "a woman is eating a piece of cake"},
...]
```



## 평가지표
이미지 캡셔닝에서 자주 사용하는 평가지표로 **Bleu_1, Bleu_2, Bleu_3, Bleu_4, METEOR, ROUGE_L, CIDEr, SPICE** 를 사용합니다. 점수가 높을수록 성능이 좋다는 것을 의미합니다.
