package Torch;

use strict;
use warnings;
use PDL;  # Core for dense, efficient tensor storage and ops
use PDL::NiceSlice;  # For intuitive slicing
use PDL::MatrixOps;  # For matrix operations
use PDL::Image2D;    # Added for Conv2D support
use Inline 'C';      # XS integration for speed-critical ops
use Exporter 'import';
our @EXPORT_OK = qw(tensor);
our $VERSION = '0.01';  # Initial version; increment as needed for releases

Inline::C->bind(<<'END_C');  # Use heredoc for Inline C to avoid DATA/__END__ parsing issues
// Fast element-wise add: Takes PDL SVs, outputs to result for inplace efficiency
void fast_add(SV* a_sv, SV* b_sv, SV* out_sv) {
    PDL_Long *dims;
    int ndims, size;
    PDL_Double *a_data, *b_data, *out_data;
    PDL *a_pdl = PDL->SvPDLV(a_sv);
    PDL *b_pdl = PDL->SvPDLV(b_sv);
    PDL *out_pdl = PDL->SvPDLV(out_sv);
    // Assume matching dims/broadcast; simplified for matching
    ndims = a_pdl->ndims;
    dims = a_pdl->dims;
    size = PDL->nelem(a_pdl);
    a_data = (PDL_Double *) a_pdl->data;
    b_data = (PDL_Double *) b_pdl->data;
    out_data = (PDL_Double *) out_pdl->data;
    for(int i = 0; i < size; i++) {
        out_data[i] = a_data[i] + b_data[i];
    }
}

// Fast matmul: Basic GEMM-like, extend with BLAS if available
void fast_matmul(SV* a_sv, SV* b_sv, SV* out_sv) {
    PDL *a_pdl = PDL->SvPDLV(a_sv);
    PDL *b_pdl = PDL->SvPDLV(b_sv);
    PDL *out_pdl = PDL->SvPDLV(out_sv);
    // Assume 2D, row-major; m x k * k x n -> m x n
    int m = a_pdl->dims[0], k = a_pdl->dims[1], n = b_pdl->dims[1];
    PDL_Double *a = (PDL_Double *) a_pdl->data;
    PDL_Double *b = (PDL_Double *) b_pdl->data;
    PDL_Double *out = (PDL_Double *) out_pdl->data;
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            out[i*n + j] = 0;
            for(int p = 0; p < k; p++) {
                out[i*n + j] += a[i*k + p] * b[p*n + j];
            }
        }
    }
}
END_C

# Basic Tensor class, wrapping PDL with autograd
{
    package Torch::Tensor;
    use overload
        '+' => \&add,
        '*' => \&mul,
        '""' => sub { $_[0]->{data} };

    sub new {
        my ($class, %args) = @_;
        bless {
            data => $args{data} // pdl([]),
            grad => $args{grad} // zeroes($args{data}->dims),
            _prev => $args{_prev} // [],
            _op => $args{_op} // '',
            _backward => $args{_backward} // sub {},
            requires_grad => $args{requires_grad} // 0,
        }, $class;
    }

    sub Torch::tensor {
        my ($data, %opts) = @_;
        my $pdl_data = ref $data eq 'ARRAY' ? pdl($data) : $data;
        $pdl_data = $pdl_data->convert($opts{type} // 'float');
        Torch::Tensor->new(
            data => $pdl_data,
            requires_grad => $opts{requires_grad} // 0,
        );
    }

    sub data { shift->{data} }
    sub grad { shift->{grad} }
    sub requires_grad { shift->{requires_grad} }

    sub backward {
        my $self = shift;
        return unless $self->requires_grad;
        my @topo = ();
        my %visited = ();
        _build_topo($self, \@topo, \%visited);
        $self->{grad} .= ones($self->{data}->dims);
        foreach my $node (reverse @topo) {
            $node->{_backward}->($node) if $node->{_backward};
        }
    }

    sub _build_topo {
        my ($node, $topo, $visited) = @_;
        return if $visited->{$node}++;
        _build_topo($_, $topo, $visited) for @{$node->{_prev}};
        push @$topo, $node;
    }

    # Add: Use XS for speed if dims match, else fallback to PDL
    sub add {
        my ($self, $other) = @_;
        $other = Torch::tensor($other) unless ref $other eq 'Torch::Tensor';
        my $out_data;
        if ($self->{data}->ndims == $other->{data}->ndims && $self->{data}->nelem == $other->{data}->nelem) {  # Simple case for XS
            $out_data = zeroes_like($self->{data});
            fast_add($self->{data}, $other->{data}, $out_data);  # XS call
        } else {
            $out_data = $self->{data} + $other->{data};  # PDL broadcast
        }
        my $out = Torch::Tensor->new(
            data => $out_data,
            _prev => [$self, $other],
            _op => '+',
            requires_grad => $self->requires_grad || $other->requires_grad,
        );
        $out->{_backward} = sub {
            my $node = shift;
            $self->{grad} += $node->{grad} if $self->requires_grad;
            $other->{grad} += $node->{grad} if $other->requires_grad;
        };
        $out;
    }

    # Mul: Similar, can add XS fast_mul if needed
    sub mul {
        my ($self, $other) = @_;
        $other = Torch::tensor($other) unless ref $other eq 'Torch::Tensor';
        my $out = Torch::Tensor->new(
            data => $self->{data} * $other->{data},
            _prev => [$self, $other],
            _op => '*',
            requires_grad => $self->requires_grad || $other->requires_grad,
        );
        $out->{_backward} = sub {
            my $node = shift;
            $self->{grad} += $other->{data} * $node->{grad} if $self->requires_grad;
            $other->{grad} += $self->{data} * $node->{grad} if $other->requires_grad;
        };
        $out;
    }

    # Matmul: Use XS for 2D cases
    sub matmul {
        my ($self, $other) = @_;
        $other = Torch::tensor($other) unless ref $other eq 'Torch::Tensor';
        my $out_data;
        if ($self->{data}->ndims == 2 && $other->{data}->ndims == 2) {
            $out_data = zeroes($self->{data}->dim(0), $other->{data}->dim(1));
            fast_matmul($self->{data}, $other->{data}, $out_data);  # XS speedup
        } else {
            $out_data = $self->{data}->inner($other->{data}->transpose);
        }
        my $out = Torch::Tensor->new(
            data => $out_data,
            _prev => [$self, $other],
            _op => 'matmul',
            requires_grad => $self->requires_grad || $other->requires_grad,
        );
        $out->{_backward} = sub {
            my $node = shift;
            if ($self->requires_grad) {
                $self->{grad} += $node->{grad}->inner($other->{data});
            }
            if ($other->requires_grad) {
                $other->{grad} += $self->{data}->transpose->inner($node->{grad});
            }
        };
        $out;
    }

    sub relu {
        my $self = shift;
        my $out = Torch::Tensor->new(
            data => maximum($self->{data}, 0),  # Dense PDL max for efficiency
            _prev => [$self],
            _op => 'relu',
            requires_grad => $self->requires_grad,
        );
        $out->{_backward} = sub {
            my $node = shift;
            $self->{grad} += ($self->{data} > 0) * $node->{grad} if $self->requires_grad;
        };
        $out;
    }

    # Sigmoid: Added layer activation
    sub sigmoid {
        my $self = shift;
        my $out_data = 1 / (1 + exp(-$self->{data}));
        my $out = Torch::Tensor->new(
            data => $out_data,
            _prev => [$self],
            _op => 'sigmoid',
            requires_grad => $self->requires_grad,
        );
        $out->{_backward} = sub {
            my $node = shift;
            $self->{grad} += $out_data * (1 - $out_data) * $node->{grad} if $self->requires_grad;
        };
        $out;
    }

    sub quantize {
        my ($self, $bits) = @_;
        my $type = $bits == 8 ? 'byte' : $bits == 16 ? 'short' : 'float';
        Torch::Tensor->new(
            data => $self->{data}->convert($type),
            requires_grad => $self->requires_grad,
        );
    }

    sub zero_grad {
        my $self = shift;
        $self->{grad} .= zeroes($self->{data}->dims);
    }
}

# Expanded NN modules
package Torch::NN::Module {
    sub parameters {
        my $self = shift;
        my @params;
        foreach my $key (keys %$self) {
            next unless ref $self->{$key} eq 'Torch::Tensor' && $self->{$key}->requires_grad;
            push @params, $self->{$key};
        }
        @params;
    }

    sub zero_grad {
        my $self = shift;
        $_->zero_grad for $self->parameters;
    }
}

package Torch::NN::Linear {
    our @ISA = qw(Torch::NN::Module);
    sub new {
        my ($class, $in_features, $out_features) = @_;
        my $weight = Torch::tensor(grandom($out_features, $in_features), requires_grad => 1)->quantize(16);
        my $bias = Torch::tensor(zeros($out_features), requires_grad => 1);
        bless { weight => $weight, bias => $bias }, $class;
    }

    sub forward {
        my ($self, $input) = @_;
        $input->matmul($self->{weight}) + $self->{bias};
    }
}

# Added: ReLU module for Sequential
package Torch::NN::ReLU {
    our @ISA = qw(Torch::NN::Module);
    sub new {
        bless {}, shift;
    }

    sub forward {
        my ($self, $input) = @_;
        $input->relu;
    }
}

# Conv2D, MaxPool2d, BatchNorm1d, Sequential as before...
package Torch::NN::Conv2d {
    our @ISA = qw(Torch::NN::Module);
    sub new {
        my ($class, $in_channels, $out_channels, $kernel_size) = @_;
        my $weight = Torch::tensor(grandom($out_channels, $in_channels, $kernel_size, $kernel_size), requires_grad => 1)->quantize(16);
        my $bias = Torch::tensor(zeros($out_channels), requires_grad => 1);
        bless { weight => $weight, bias => $bias }, $class;
    }

    sub forward {
        my ($self, $input) = @_;
        # Assume input: (batch, height, width, channels) adjust if needed; simplified loop for test
        my $out = zeroes($input->dim(0), $input->dim(1) - $self->{weight}->dim(2) + 1, $input->dim(2) - $self->{weight}->dim(3) + 1, $self->{weight}->dim(0));
        for my $b (0..$input->dim(0)-1) {
            for my $o (0..$self->{weight}->dim(0)-1) {
                my $conv_sum = zeroes($out->dim(1), $out->dim(2));
                for my $i (0..$input->dim(3)-1) {
                    $conv_sum += conv2d($input->(:,:,:,$b,$i), $self->{weight}->(:,:,$o,$i));  # Adjust slices
                }
                $out->(:,:,$o,$b) .= $conv_sum + $self->{bias}->at($o);
            }
        }
        Torch::Tensor->new(data => $out, requires_grad => $input->requires_grad);
    }
}

package Torch::NN::MaxPool2d {
    our @ISA = qw(Torch::NN::Module);
    sub new {
        my ($class, $kernel_size) = @_;
        bless { kernel_size => $kernel_size }, $class;
    }

    sub forward {
        my ($self, $input) = @_;
        my $ks = $self->{kernel_size};
        my $out_h = int($input->dim(1) / $ks);
        my $out_w = int($input->dim(2) / $ks);
        my $out = zeroes($input->dim(0), $out_h, $out_w, $input->dim(3));
        for my $i (0..$out_h-1) {
            for my $j (0..$out_w-1) {
                $out->(:, $i, $j, :) .= $input->(:, $i*$ks:($i+1)*$ks-1, $j*$ks:($j+1)*$ks-1, :)->maximum_nd( [1,2] );
            }
        }
        Torch::Tensor->new(data => $out, requires_grad => $input->requires_grad);
    }
}

package Torch::NN::BatchNorm1d {
    our @ISA = qw(Torch::NN::Module);
    sub new {
        my ($class, $num_features) = @_;
        my $gamma = Torch::tensor(ones($num_features), requires_grad => 1);
        my $beta = Torch::tensor(zeros($num_features), requires_grad => 1);
        my $running_mean = zeroes($num_features);
        my $running_var = ones($num_features);
        bless { gamma => $gamma, beta => $beta, running_mean => $running_mean, running_var => $running_var, momentum => 0.1 }, $class;
    }

    sub forward {
        my ($self, $input, $training) = @_;
        $training //= 0;
        my $mean = $training ? $input->mean(0) : $self->{running_mean};
        my $var = $training ? (($input - $mean)**2)->mean(0) : $self->{running_var};
        if ($training) {
            $self->{running_mean} .= (1 - $self->{momentum}) * $self->{running_mean} + $self->{momentum} * $mean;
            $self->{running_var} .= (1 - $self->{momentum}) * $self->{running_var} + $self->{momentum} * $var;
        }
        my $norm = ($input - $mean) / sqrt($var + 1e-5);
        $self->{gamma} * $norm + $self->{beta};
    }
}

package Torch::NN::Sequential {
    our @ISA = qw(Torch::NN::Module);
    sub new {
        my ($class, @layers) = @_;
        bless { layers => \@layers }, $class;
    }

    sub forward {
        my ($self, $input) = @_;
        my $x = $input;
        $x = $_->forward($x) for @{$self->{layers}};
        $x;
    }

    sub parameters {
        my $self = shift;
        my @params;
        push @params, $_->parameters for @{$self->{layers}};
        @params;
    }
}

1;

__END__

=head1 NAME

Torch - Enhanced Perl PyTorch emulation with more NN layers and XS opts

=head1 SYNOPSIS

  use Torch qw(tensor);
  my $model = Torch::NN::Sequential->new(
      Torch::NN::Conv2d->new(3, 16, 3),
      Torch::NN::MaxPool2d->new(2),
      Torch::NN::Linear->new(16*..., 10),  # Adjust dims
  );
  # Train with XS-accelerated ops for speed

=head1 DESCRIPTION

Updated draft adds Conv2d (via PDL conv2d), MaxPool2d, BatchNorm1d, Sigmoid, and Sequential. XS via Inline::C optimizes add/matmul for denser, faster execution than Python equivalents.

=cut
