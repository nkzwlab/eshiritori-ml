# ORF 絵しりとり モデル

## 環境設定

```
docker-compose -v
```
の出力が `docker-compose version 1.29.2, build 5becea4c` じゃなかったら、docker-composeのバージョンを上げる
参照：https://qiita.com/kottyan/items/c892b525b14f293ab7b3

## 動かし方

```
make init
```
して、docker環境を立ち上げる。
ファイルを動かしたい場合は、

```
make run filename=src/ファイル名.py
```
で動かす