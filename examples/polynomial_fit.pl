use strict;
use warnings;
use PDL;
use Torch qw(tensor);

# Mimic PyTorch polynomial fit: y = a + b x + c x^2 + d x^3 to exp(x)
my $num_points = 2000;
my $x = zeroes($num_points)->xlinvals(-1, 1);
my $y = exp($x);

my $a = tensor(grandom->sclr, requires_grad => 1);
my $b = tensor(grandom->sclr, requires_grad => 1);
my $c = tensor(grandom->sclr, requires_grad => 1);
my $d = tensor(grandom->sclr, requires_grad => 1);

my $learning_rate = 1e-5;
my $initial_loss;

for my $t (0 .. 4999) {
    my $y_pred = $a + $b * $x + $c * ($x ** 2) + $d * ($x ** 3);
    my $loss = ($y_pred - $y)->pow(2)->sum;

    $initial_loss //= $loss->data->sclr;

    if ($t % 100 == 99) {
        printf "Iteration %4d loss(t)/loss(0) = %10.6f a = %10.6f b = %10.6f c = %10.6f d = %10.6f\n",
            $t, $loss->data->sclr / $initial_loss, $a->data->sclr, $b->data->sclr, $c->data->sclr, $d->data->sclr;
    }

    $loss->backward;

    $a->{data} -= $learning_rate * $a->grad;
    $b->{data} -= $learning_rate * $b->grad;
    $c->{data} -= $learning_rate * $c->grad;
    $d->{data} -= $learning_rate * $d->grad;

    $a->zero_grad;
    $b->zero_grad;
    $c->zero_grad;
    $d->zero_grad;
}

printf "Result: y = %.6f + %.6f x + %.6f x^2 + %.6f x^3\n",
    $a->data->sclr, $b->data->sclr, $c->data->sclr, $d->data->sclr;
