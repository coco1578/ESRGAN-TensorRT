import argparse
import os

import tensorrt as trt



def build_engine(onnx_model_path,
                 tensorrt_engine_path,
                 fp16=True,
                 dynamic_axes=False,
                 image_size=(3, 256, 256),
                 batch_size=1,
                 min_engine_batch_size=(3, 64, 64),
                 opt_engine_batch_size=(3, 256, 256),
                 max_engine_batch_size=(3, 512, 512),
                 ):

    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    
    # Set FP16
    if fp16 is True:
        config.set_flag(trt.BuilderFlag.FP16)
    
    # Parse ONNX
    parser = trt.OnnxParser(network, logger)
    
    with open(onnx_model_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    input_tensor = network.get_input(0)
    if dynamic_axes:
        profile.set_shape(input_tensor.name, 
                          (min_engine_batch_size, *image_size),
                          (opt_engine_batch_size, *image_size), 
                          (max_engine_batch_size, *image_size)
                          )
    else:
        profile.set_shape(input_tensor.name, 
                          (batch_size, *image_size), 
                          (batch_size, *image_size), 
                          (batch_size, *image_size)
                          )
    config.add_optimization_profile(profile)
    
    engine_string = builder.build_serialized_network(network, config)
    if engine_string == None:
        return None
    
    with open(tensorrt_engine_path, "wb") as f:
        f.write(engine_string)


def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', 
                        type=str, 
                        help='input onnx model path'
                        )
    parser.add_argument('--output', 
                        type=str,
                        help='output tensorrt engine path'
                        )
    parser.add_argument('--fp16', 
                        default=True, 
                        action='store_true',
                        help='use float16 precision' 
                        )
    
    args = parser.parse_args()
    return args


def main():
    
    args = parse_args()
    
    build_engine(args.input, args.output, args.fp16)