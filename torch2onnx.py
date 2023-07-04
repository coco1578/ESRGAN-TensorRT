import argparse

import torch
import torch.onnx
from basicsr.archs.rrdbnet_arch import RRDBNet


def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',
                        type=str,
                        required=True,
                        help='input pytorch model path'
                        )
    parser.add_argument('--output', 
                        type=str, 
                        required=True, 
                        help='output onnx model path'
                        )
    parser.add_argument('--params',
                        action='store_false',
                        help='use params instead of params_ema'
                        )
    parser.add_argument('--fp16',
                        action='store_true',
                        help='use float16 precision'
                        )
    parser.add_argument('--opset',
                        type=int,
                        default=17,
                        help='onnx opset version'
                        )
    
    args = parser.parse_args()
    return args


def main(args):
    
    model = RRDBNet(num_in_ch=3,
                    num_out_ch=3,
                    num_feats=64,
                    num_block=23,
                    num_grow_ch=32,
                    scale=4
                    )

    if args.params:
        keyname = 'params'
    else:
        keyname = 'params_ema'
    
    model.load_state_dict(torch.load(args.input)[keyname])
    model.train(False)
    if args.fp16:
        model.half()
    model.cuda().eval()
    
    x = torch.rand(1, 3, 256, 256)
    if args.fp16:
        x = x.half().cuda()
    else:
        x = x.cuda()
        
    torch.onnx.export(model,
                      x,
                      args.output,
                      verbose=True,
                      input_names=['input'],
                      output_names=['output'],
                      opset_version=args.opset,
                      export_params=True
                      )


if __name__ == '__main__':
    
    args = parse_args()
    main()