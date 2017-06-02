## simple implementation of pix2pix by pytorch

Original paper: [Isola, P., Zhu, J. Y., Zhou, T., & Efros, A. A. (2016). Image-to-image translation with conditional adversarial networks](https://arxiv.org/pdf/1611.07004v1.pdf). arXiv preprint arXiv:1611.07004.   [[Project Website]](https://phillipi.github.io/pix2pix/)


## Results  
Testdata（200epochs)
![代替テキスト](./fig/facadetestresult.jpg)



## Prerequisites
* PyTorch
* python 3
* OpenCV 3
* Nvidia GPU


## Usage

Download the CMP Facade Database.  
 http://cmp.felk.cvut.cz/~tylecr1/facade/CMP_facade_DB_base.zip  
(unzip ./datasets/CMP_facade_DB_base.zip)


#### Train
```
python train.py
```

#### Test
 ```
 python test.py
 ```




### 別のデータセットを使用する (例：線画着色)

   ##### データセット  
   データセットフォルダ（デフォルト ./datasets/cnvframesEx）に、線画画像ファイルとカラー画像ファイルをペアで置いてください。


  線画画像ファイル名: シーケンス番号8桁+A.jpg  (例：00000000A.jpg)  
  カラー画像ファイル名: シーケンス番号8桁+B.jpg  (例：00000000B.jpg)  
  画像サイズ:  縦横256×256ピクセル
  ```
  線画画像:  　00000000A.jpg 　00000001A.jpg 　00000002A.jpg ...  
カラー画像: 　00000000B.jpg 　00000001B.jpg 　00000002B.jpg ...
  ```  

   ##### Train
   訓練に使用するデータの範囲を指定してくだい。任意のデータセットフォルダを使用するときは--datasetオプションを追加してくだい。　( --dataset ./datasets/mydataset )
   ```
   python train_line.py --data_start 0 --data_end 999
   ```
   ##### Test
   テストに使用するデータの範囲を指定してくだい。任意のデータセットフォルダを使用するときは--datasetオプションを追加してくだい。　( --dataset ./datasets/mydataset )
   ```
   python test_line.py --data_start 1000 --data_end 1500
   ```



* テスト例  
  テストデータ　（訓練データ画像1,000枚、200エポック)
   ![代替テキスト](./fig/colorize.jpg)
   このテストの訓練データはいくつかのアニメ作品の画像です。しかしながらここでのテストはアニメでなく、以下のサイトの線画作品を使用させていただきました。  
   [着色練習用線画](http://iradukai.com/nurie2.htm)  
   （上段） [No.019] 線画提供 - ドクター博士様,	（下段左）[No.024] 線画提供 - トシトキコ様 ,（下段右）	[No.030] 線画提供 - ぬこ野郎様  
   画像は処理の都合上、すべて256x256にクリッピングおよびリサイズさせていただきました。（絵師様すいません）

## Reference
* [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
* [mrzhu-cool/pix2pix-pytorch](https://github.com/mrzhu-cool/pix2pix-pytorch)  
* [pfnet-research/chainer-pix2pix](https://github.com/pfnet-research/chainer-pix2pix)

## License
#### 　MIT

<!--
## ブログ
* http://eiji-kb.hatenablog.com/entry/2017/
-->
