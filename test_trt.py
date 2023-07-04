import argparse

import cv2
import numpy as np
import torch
import pycuda.driver as cuda
import tensorrt as trt


def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='TensorRT model path')
    parser.add_argument('--image', type=str, required=True, help='Input image path')
    parser.add_argument('--output', type=str, required=True, help='Output image path')
    parser.add_argument('--upscale', type=int, default=4, help='Upscale Rate')
    parser.add_argument('--image_size', type=int, default=256, help='Image size used for TensorRT conversion.')
    
    args = parser.parse_args()
    return args


# https://github.com/Devyanshu/image-split-with-overlap/blob/master/split_image_with_overlap.py
def start_points(size, stride=256):
    points = [0]
    counter = 1
    while True:
        pt = stride * counter
        if pt + stride >= size:
            if stride == size:
                break
            points.append(size - stride)
            break
        else:
            points.append(pt)
        counter += 1
    return points



def build_engine_context(trt_model_path='experiments/pretrained_models/finetune_realesrgan_x4plus_pairdata_ARCHI4K_ver2.trt'):
    
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)
    trt.init_libnvinfer_plugins(None, '')
    
    engine = runtime.deserialize_cuda_engine(open(trt_model_path, 'rb').read())
    assert engine
    
    context = engine.create_execution_context()
    assert context
    context.set_binding_shape(0, (1, 3, 256, 256))

    return engine, context


def alloc_buffers(engine, context):
    
    inputs = []
    outputs = []
    allocations = []
    # alloc buf
    for i in range(engine.num_bindings):
        is_input = False
        if engine.binding_is_input(i):
            is_input = True
        
        name = engine.get_binding_name(i)
        dtype = np.dtype(trt.nptype(engine.get_binding_dtype(i)))
        shape = context.get_binding_shape(i)
        
        if is_input and shape[0] < 0:
            assert engine.num_optimization_profiles > 0
            profile_shape = engine.get_profile_shape(0, name)
            assert len(profile_shape) == 3  # min, opt, max
            context.set_binding_shape(i, profile_shape[2])
            shape = context.get_binding_shape(i)
        if is_input:
            batch_size = shape[0]
        
        size = dtype.itemsize 
        for s in shape:
            size *= s
        
        allocation = cuda.mem_alloc(size)
        host_allocation = None if  is_input else np.zeros(shape, dtype)
        binding = {
            'index': i,
            'name': name,
            'dtype': dtype,
            'shape': list(shape),
            'allocation': allocation,
            'host_allocation': host_allocation
        }
        
        allocations.append(allocation)
        if engine.binding_is_input(i):
            inputs.append(binding)
        else:
            outputs.append(binding)

    return inputs, outputs, allocations



if __name__ == '__main__':
    
    args = parse_args()

    cuda.init()
    device = cuda.Device(0)
    ctx = device.make_context()
    
    engine, context = build_engine_context(args.model)    
    inputs, outputs, allocations = alloc_buffers(engine, context)

    # inference
    im = cv2.imread(args.image)
    H, W, C = im.shape
    X_points = start_points(W)
    Y_points = start_points(H)

    im = im / 255
    im = torch.from_numpy(np.transpose(im, (2, 0, 1)))
    im = im.unsqueeze(0).numpy()

    b, c, h, w = im.shape
    output_height = h * args.upscale
    output_width = w * args.upscale
    output_shape = (b, c, output_height, output_width)

    output_image = np.zeros(output_shape)
    mask_image = np.zeros(output_shape)

    # upscale using image tile 
    for i in Y_points:
        for j in X_points:
            input_tile = im[:, :, i:i+args.image_size, j:j+args.image_size]
            input_tile = np.ascontiguousarray(input_tile, dtype=np.float16)
            
            cuda.memcpy_htod(inputs[0]['allocation'], input_tile)
            context.execute_v2(allocations)
            cuda.memcpy_dtoh(outputs[0]['host_allocation'], outputs[0]['allocation'])
            
            output_tile = outputs[0]['host_allocation']

            output_start_x = j * args.upscale
            output_end_x = (j + args.image_size) * args.upscale
            output_start_y = i * args.upscale
            output_end_y = (i + args.image_size) * args.upscale

            output_image[:, :, output_start_y:output_end_y, output_start_x:output_end_x] += output_tile[:, :, :, :]
            mask_image[:, :, output_start_y:output_end_y, output_start_x:output_end_x] += 1

    output_image = output_image / mask_image
    output_image = np.clip(np.squeeze(output_image, 0), 0, 1)
    output_image = np.transpose(output_image[[2, 1, 0], :, :], (1, 2, 0))
    output_image = (output_image * 255.0).round().astype(np.uint8)
    cv2.imwrite(args.output, cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))

    ctx.pop()