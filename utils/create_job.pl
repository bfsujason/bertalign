#!/usr/bin/perl

use strict;
use warnings;

use 5.010;
use utf8;

use Getopt::Long;
use File::Spec;

sub _main {
    GetOptions( \ my %opts,
        'i=s',
        'o=s',
		'j=s',
        's=s',
        't=s',
		'trans',
    );
	
	_create_job_file(
		File::Spec->rel2abs($opts{i}),
		File::Spec->rel2abs($opts{o}),
		File::Spec->rel2abs($opts{j}),
		$opts{s},
		$opts{t},
		$opts{trans}
	);
}

sub _create_job_file {
	my ($data_dir, $auto_dir, $job_fn, $src, $tgt, $trans) = @_;
	my ($src_fns, $tgt_fns) = _get_src_tgt_fns($data_dir, $src, $tgt);
    my @align_fns = map { my ($id) = $_ =~ /(\d+)\./; $id . '.align'; } @{$src_fns};
    my @table     = map { join "\t", File::Spec->catfile($data_dir, $src_fns->[$_]),
                                     File::Spec->catfile($data_dir, $tgt_fns->[$_]),
                                     File::Spec->catfile($auto_dir, $align_fns[$_]) } ( 0 .. scalar @{$src_fns} - 1 );
	if ( $trans ) {
		my @trans_fns = map { my ($id) = $_ =~ /(\d+)\./; $id . '.trans'; } @{$src_fns};
		@table = map { join "\t", (File::Spec->catfile($data_dir, $trans_fns[$_]), $table[$_]) } ( 0 .. scalar @table - 1 );
	}

	#my $job_fn = File::Spec->catfile($job_dir, $aligner . '.job');
	open my $OUT, '>:utf8', $job_fn;
    binmode $OUT; # output unix-like LF(\n) instead of CRLF(\r\n)
    print $OUT join "\n", @table;
}

sub _get_src_tgt_fns {
	my ($dir, $src, $tgt) = @_;
	my ($src_fns, $tgt_fns);
	opendir(my $DH, $dir);
    foreach my $fn ( sort readdir $DH ) {
		next if $fn =~ /^\./;
		push @{$src_fns}, $fn if $fn =~ /\.$src\z/;
        push @{$tgt_fns}, $fn if $fn =~ /\.$tgt\z/;
	}
	return ($src_fns, $tgt_fns);
}


unless ( caller ) {
	_main();
}

__END__