import torch
import torch.nn as nn
from torch.autograd import Variable
from gpu_mem_track import  MemTracker
import inspect
import warnings
import gc

def detach_variable(inputs):
    if isinstance(inputs, tuple):
        out = []
        for inp in inputs:
            x = inp.detach()
            x.requires_grad = True
            out.append(x)
        return tuple(out)
    else:
        raise RuntimeError(
            "Only tuple of tensors is supported. Got Unsupported input type: ", type(inputs).__name__)


def check_backward_validity(inputs):
    if not any(inp.requires_grad for inp in inputs):
        warnings.warn("None of the inputs have requires_grad=True. Gradients will be None")


class CheckpointFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, run_function, stash, *args):
        check_backward_validity(args)
        ctx.run_function = run_function
        ctx.stash = stash
        if ctx.stash == True:
            ctx.save_for_backward(*args)
        with torch.no_grad():
            outputs = run_function[-1](*args)
        return outputs

    @staticmethod
    def backward(ctx, *args):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError("Checkpointing is not compatible with .grad(), please use .backward() if possible")

        if ctx.stash:
            inputs = ctx.saved_tensors
        else:
            inputs = inp.cuda()
            with torch.no_grad():
                for i in range(len(ctx.run_function) - 1):
                    # print(inputs.size())
                    inputs = ctx.run_function[i](inputs)
                    gpu_tracker.track()
            inputs = (inputs, )


        detached_inputs = detach_variable(inputs)
        del inputs
        with torch.enable_grad():
            print(ctx.run_function[-1].__module__)
            outputs = ctx.run_function[-1](*detached_inputs)
        del ctx.run_function
        gpu_tracker.track()


        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        # print(args)
        torch.autograd.backward(outputs, args)
        del outputs
        gpu_tracker.track()


        return_value = (None, None) + tuple(inp.grad for inp in detached_inputs)
        # del inputs
        del detached_inputs
        # del outputs
        # torch.cuda.empty_cache()
        gc.collect()
        # print(return_value)
        return return_value


def checkpoint(function, *args):
    r"""Checkpoint a model or part of the model

    Checkpointing works by trading compute for memory. Rather than storing all
    intermediate activations of the entire computation graph for computing
    backward, the checkpointed part does **not** save intermediate activations,
    and instead recomputes them in backward pass. It can be applied on any part
    of a model.

    Specifically, in the forward pass, :attr:`function` will run in
    :func:`torch.no_grad` manner, i.e., not storing the intermediate
    activations. Instead, the forward pass saves the inputs tuple and the
    :attr:`function` parameter. In the backwards pass, the saved inputs and
    :attr:`function` is retreived, and the forward pass is computed on
    :attr:`function` again, now tracking the intermediate activations, and then
    the gradients are calculated using these activation values.

    .. warning::
        Checkpointing doesn't work with :func:`torch.autograd.grad`, but only
        with :func:`torch.autograd.backward`.

    .. warning::
        If :attr:`function` invocation during backward does anything different
        than the one during forward, e.g., due to some global variable, the
        checkpointed version won't be equivalent, and unfortunately it can't be
        detected.

    .. warning:
        At least one of the inputs needs to have :code:`requires_grad=True` if
        grads are needed for model inputs, otherwise the checkpointed part of the
        model won't have gradients.

    Args:
        function: describes what to run in the forward pass of the model or
            part of the model. It should also know how to handle the inputs
            passed as the tuple. For example, in LSTM, if user passes
            ``(activation, hidden)``, :attr:`function` should correctly use the
            first input as ``activation`` and the second input as ``hidden``
        args: tuple containing inputs to the :attr:`function`

    Returns:
        Output of running :attr`function` on *:attr:`args`
    """
    return CheckpointFunction.apply(function, *args)

def view(input):
    return input.view(32,-1)


inp = Variable(torch.rand(32,3,224,224), requires_grad = True)

VGG16_layers=[]

VGG16_layers.append(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1))
VGG16_layers.append(nn.ReLU())
VGG16_layers.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))
VGG16_layers.append(nn.ReLU())
VGG16_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

VGG16_layers.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1))
VGG16_layers.append(nn.ReLU())
VGG16_layers.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1))
VGG16_layers.append(nn.ReLU())
VGG16_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

VGG16_layers.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1))
VGG16_layers.append(nn.ReLU())
VGG16_layers.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))
VGG16_layers.append(nn.ReLU())
VGG16_layers.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))
VGG16_layers.append(nn.ReLU())
VGG16_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

VGG16_layers.append(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1))
VGG16_layers.append(nn.ReLU())
VGG16_layers.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
VGG16_layers.append(nn.ReLU())
VGG16_layers.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
VGG16_layers.append(nn.ReLU())
VGG16_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

VGG16_layers.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
VGG16_layers.append(nn.ReLU())
VGG16_layers.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
VGG16_layers.append(nn.ReLU())
VGG16_layers.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
VGG16_layers.append(nn.ReLU())
VGG16_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

VGG16_layers.append(view)

VGG16_layers.append(nn.Linear(512 * 7 * 7, 4096))
VGG16_layers.append(nn.ReLU())

VGG16_layers.append(nn.Linear(4096, 4096))
VGG16_layers.append(nn.ReLU())

VGG16_layers.append(nn.Linear(4096, 1000))

for i in VGG16_layers:
    if i != view:
        i = i.cuda()
x = inp.cuda()

frame = inspect.currentframe()  # define a frame to track
gpu_tracker = MemTracker(frame)  # define a GPU tracker
gpu_tracker.track()

# forward()
for i in range(len(VGG16_layers)):
    # y = x
    # if i == 3:
    #     stash = True
    # else:
    stash = False
    x = checkpoint(tuple(VGG16_layers[:i + 1]), stash, x)
    # if i > 0:
    #     del y
    #     gc.collect()
    gpu_tracker.track()

# backward()
x.sum().backward()

gpu_tracker.track()


