use strict;
use warnings;
use Test::More;
use PDL;
use PDL::Constants qw(PI);
use Torch qw(tensor);

# Mimic PyTorch test structure: subtests for tensors, ops, autograd, NN
subtest 'Tensor basics' => sub {
    my $t = tensor([1,2,3]);
    isa_ok($t, 'Torch::Tensor', 'Tensor creation');
    is_deeply($t->data->list, (1,2,3), 'Data access');

    my $qt = $t->quantize(8);
    is($qt->data->type, 'byte', 'Quantization to 8-bit');
};

subtest 'Operations' => sub {
    my $a = tensor([1,2]);
    my $b = tensor([3,4]);

    my $sum = $a + $b;
    is_deeply($sum->data->list, (4,6), 'Add op');

    my $prod = $a * $b;
    is_deeply($prod->data->list, (3,8), 'Mul op');

    my $mat_a = tensor([[1,2],[3,4]]);
    my $mat_b = tensor([[5,6],[7,8]]);
    my $mm = $mat_a->matmul($mat_b);
    is_deeply([ $mm->data->list ], [19,22,43,50], 'Matmul op');

    my $relu_t = tensor([-1,0,1])->relu;
    is_deeply($relu_t->data->list, (0,0,1), 'ReLU op');

    my $sig_t = tensor([0])->sigmoid;
    is($sig_t->data->sclr, 0.5, 'Sigmoid op');
};

subtest 'Autograd simple (mimic PyTorch autograd_tutorial)' => sub {
    my $a = tensor([2,3], requires_grad => 1);
    my $b = tensor([6,4], requires_grad => 1);
    my $Q = ($a * 3)->mul($a ** 2) - ($b ** 2);  # 3*a**3 - b**2

    $Q->backward;

    my $expected_a_grad = 9 * ($a ** 2);
    my $expected_b_grad = -2 * $b;
    is_deeply([$a->grad->list], [$expected_a_grad->list], 'a grad check');
    is_deeply([$b->grad->list], [$expected_b_grad->list], 'b grad check');
};

subtest 'NN Modules' => sub {
    my $linear = Torch::NN::Linear->new(2, 3);
    my $input = tensor([[1,2]]);
    my $out = $linear->forward($input);
    is_deeply($out->data->dims, (1,3), 'Linear forward shape');

    my $conv = Torch::NN::Conv2d->new(1, 1, 3);
    my $conv_in = tensor([[[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]]]);  # Adjust to 1 batch, 4x4, 1 channel for kernel 3
    my $conv_out = $conv->forward($conv_in);
    is_deeply($conv_out->data->dims, (1,2,2,1), 'Conv2d forward shape');  # (batch, h-2, w-2, out_ch)

    my $pool = Torch::NN::MaxPool2d->new(2);
    my $pool_in = tensor([[[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]]]);  # 1b,4h,4w,1ch
    my $pool_out = $pool->forward($pool_in);
    is_deeply([$pool_out->data->list], [6,8,14,16], 'MaxPool2d forward');  # Adjust expect

    my $bn = Torch::NN::BatchNorm1d->new(2);
    my $bn_in = tensor([[1,2],[3,4]]);
    my $bn_out = $bn->forward($bn_in, 1);  # training
    ok(defined $bn_out, 'BatchNorm1d forward');

    my $seq = Torch::NN::Sequential->new(
        Torch::NN::Linear->new(2, 3),
    );  # Remove ReLU if not needed for test; or add above
    my $seq_out = $seq->forward($input);
    is_deeply($seq_out->data->dims, (1,3), 'Sequential forward');

    my @params = $seq->parameters;
    is(scalar @params, 2, 'Parameters count');  # weight + bias
};

subtest 'Autograd with NN (mimic PyTorch nn tutorial)' => sub {
    my $model = Torch::NN::Sequential->new(
        Torch::NN::Linear->new(3, 1),
    );
    my $num_points = 2000;
    my $x = zeroes($num_points)->xlinvals(-PI, PI);
    my $powers = pdl([1,2,3]);
    my $xx = ($x->dummy(0,3) ** $powers->dummy(1))->transpose;  # (2000,3) via broadcast
    my $y = sin($x);

    my $y_pred = $model->forward($xx);
    my $loss = ($y_pred - $y->dummy(1))->pow(2)->sum;
    $loss->backward;

    ok($model->{layers}[0]{weight}->grad->nelem > 0, 'Grad propagation to params');

    $model->zero_grad;
    is($model->{layers}[0]{weight}->grad->sum->sclr, 0, 'Zero grad');
};

done_testing;
