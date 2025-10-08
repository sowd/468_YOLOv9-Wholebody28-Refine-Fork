# BoT-SORT

下記のリポジトリをマージしたものです。

- <https://github.com/PINTO0309/PINTO_model_zoo/tree/main/468_YOLOv9-Wholebody28-Refine>

- <https://github.com/PINTO0309/PINTO_model_zoo/tree/main/462_Gaze-LLE>

- <https://github.com/PINTO0309/BoT-SORT-ONNX-TensorRT>

- マージして作成したファイル

  [demo_bottrack_wholebody28_gazelle.py](demo_bottrack_wholebody28_gazelle.py)

## 468_YOLOv9-Wholebody28-Refine

[README.md](README-468_YOLOv9-Wholebody28-Refine.md)

[LICENSE](LICENSE-468_YOLOv9-Wholebody28-Refine)

[url.txt](url-468_YOLOv9-Wholebody28-Refine.txt)

## 462_Gaze-LLE

[README.md](README-462_Gaze-LLE.md)

[LICENSE](LICENSE-462_Gaze-LLE)

[url.txt](url-462_Gaze-LLE.txt)

## BoT-SORT-ONNX-TensorRT

[README](README-BoT-SORT-ONNX-TensorRT.md)

[LICENSE](LICENSE-BoT-SORT-ONNX-TensorRT)

## 使い方

`BoT-SORT-ONNX-TensorRT`の手順にあるDockerコンテナを使用します。

```bash
docker run -it --gpus all -v `pwd`:/workdir -e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix:rw pinto0309/botsort_onnx_tensorrt:latest
```

`468_YOLOv9-Wholebody28-Refine`と`462_Gaze-LLE`で必要なパッケージを追加します。

```bash
sudo apt update
sudo apt install gcc build-essential libdbus-glib-1-dev libgirepository1.0-dev
sudo apt install libcairo2-dev libjpeg-dev libgif-dev
sudo apt install libgirepository1.0-dev gir1.2-girepository-2.0 libcairo2-dev pkg-config python3-dev gir1.2-gtk-3.0
sudo apt install xvfb
```

Pythonモジュールを追加します。

```bash
pip install Pillow matplotlib pycairo PyGObject gnuhealth-client
pip install -U onnx==1.19.0
pip install -U onnx_graphsurgeon==0.5.8
pip install -U sne4onnx==1.0.13
pip install -U sor4onnx==1.0.7
```

ディスプレイが無い環境の場合は、`Xvfb`を実行しておきます。

```bash
export DISPLAY=:1
nohup Xvfb -ac ${DISPLAY} -screen 0 1280x780x24 &
```

モデルをダウンロードします。

```bash
./download_e_withpost.sh
./download.sh
```

下記のコマンドで実行します。

```bash
python demo_bottrack_wholebody28_gazelle.py -ep cuda -v input.mp4
```
