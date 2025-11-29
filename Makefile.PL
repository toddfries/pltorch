use 5.006;
use strict;
use warnings;
use ExtUtils::MakeMaker;

WriteMakefile(
    NAME             => 'Torch',
    AUTHOR           => q{Your Name <your@email.com>},  # Replace with actual
    VERSION_FROM     => 'lib/Torch.pm',
    ABSTRACT_FROM    => 'lib/Torch.pm',
    LICENSE          => 'perl_5',
    MIN_PERL_VERSION => '5.006',
    CONFIGURE_REQUIRES => {
        'ExtUtils::MakeMaker' => '0',
    },
    BUILD_REQUIRES => {
        'Test::More' => '0',
    },
    PREREQ_PM => {
        'PDL'             => '2.000',  # Core for tensors
        'PDL::NiceSlice'  => '0',
        'PDL::MatrixOps'  => '0',
        'PDL::Image2D'    => '0',      # For Conv2D
        'Inline::C'       => '0.82',   # For XS optimizations
        'Exporter'        => '0',
    },
    dist  => { COMPRESS => 'gzip -9f', SUFFIX => 'gz', },
    clean => { FILES => 'Torch-*' },
);
