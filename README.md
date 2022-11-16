# ORF 絵しりとり モデル

ORF絵しりとりで使うためのモデル。このレポジトリのコードでは絵を受け取って、書かれた絵に対して推論し、分類結果をクラス名で返す。

## 環境設定

### 1.docker-composeのバージョン確認
```
docker-compose -v
```
の出力が `docker-compose version 1.29.2, build 5becea4c` じゃなかったら、docker-composeのバージョンを上げる

参照：https://qiita.com/kottyan/items/c892b525b14f293ab7b3

### 2.docker環境の立ち上げ
```
make init
```
して、docker環境を立ち上げる。

### 3.データセットのダウンロード
```
make run filename=src/dataset.py
```
で、データセットをダウンロードする。（少し時間がかかるはず）


## 動かし方
ファイルを動かしたい場合は、
```
make run filename=src/ファイル名.py
```
で動かす


## データセット

[quickdraw dataset](https://quickdraw.withgoogle.com/data) を使う。
- 手書きのスケッチ 345 クラスから成る、合計5000万枚のデータセット
- [クラス名一覧](https://github.com/googlecreativelab/quickdraw-dataset/blob/master/categories.txt)

