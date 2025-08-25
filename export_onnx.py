from io import BytesIO
from vision.ssd.vgg_ssd import create_vgg_ssd
import onnx
import torch

VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

if __name__ == "__main__":
    trained_model = "/mnt/quedoulin/github_code/pytorch-ssd/vgg16-ssd-mp-0_7726.pth"
    num_classes = len(VOC_CLASSES) + 1 # +1 background

    net = create_vgg_ssd(num_classes, True)
    net.load_state_dict(torch.load(trained_model))
    net.eval()
    model = net.to("cpu")
    x = torch.randn((1, 3, 300, 300), device="cpu")
    y = net(x)
    print(y[0].shape)

    with BytesIO() as f:
        torch.onnx.export(net, x, f, verbose=False, opset_version=11,
                            training=torch.onnx.TrainingMode.EVAL,
                            do_constant_folding=True,
                            input_names=["images"],
                            output_names=["output"]
                         )
        f.seek(0)
        # Checks
        onnx_model = onnx.load(f)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model

        import onnxsim
        print('\nStarting to simplify ONNX...')
        try:
            onnx_model, check = onnxsim.simplify(onnx_model)
            assert check, 'assert check failed'
        except Exception as e:
            print(f'Simplifier failure: {e}')
        onnx.save(onnx_model, "vgg16-ssd-mp-0_7726.onnx")
        print(f'ONNX export success, save into vgg16-ssd-mp-0_7726.onnx')