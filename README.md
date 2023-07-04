# Transform ESRGAN Pytorch to TensorRT and Inference

## Transform Process
<br>

### 1. Pytorch 2 ONNX
```python
python torch2onnx.py --input esrgan.pth --output esrgan.onnx --fp16 
```
<br>

### ONNX 2 TensorRT
```python
python onnx2trt.py --input esrgan.onnx --output esrgan.trt --fp16
```
<br>

### Test inference with TensorRT ESRGAN
```python
python test_trt.py --model esrgan.trt --image input.png --output output.png
```