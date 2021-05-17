#!/usr/bin/perl

use strict;
use warnings;

use 5.010;
use utf8;

use Getopt::Long;
use File::Spec;

sub _main {
    GetOptions( \ my %opts,
		'in_dir=s',
        'out_dir=s',
    );
	my $src_fns = _get_src_fns($opts{in_dir});
	foreach my $src_fn ( @{$src_fns} ) {
		my ($base_fn) = $src_fn =~ /(.*)\-/;
		my $src_path = File::Spec->catfile($opts{in_dir}, $src_fn);
		my $tgt_path = File::Spec->catfile($opts{in_dir}, $base_fn . '-t');
		my $out_path = File::Spec->catfile($opts{out_dir}, $base_fn);
		#say $src_path;
		#say $tgt_path;
		#say $out_path;
		my $src_seg = _read_align($src_path);
		my $tgt_seg = _read_align($tgt_path);
		my @bi_seg  = map { 
			'[' . $src_seg->[$_] . ']' .
			':' .
			'[' . $tgt_seg->[$_] . ']'
		} ( 0 .. scalar @{$src_seg} - 1 );
		open my $OUT, '>:utf8', $out_path;
		say $OUT join "\n", @bi_seg;
	}
	
}

sub _read_align {
	my $file = shift;
	my $align = [];
	open my $IN, '<:utf8', $file;
	while ( defined(my $line = <$IN>) ) {
		chomp $line;
		push @{$align}, $line;
	}
	return $align;
}

sub _get_src_fns {
    my $dir = shift;
    my $fns;
    opendir(my $DH, $dir);
    while( my $fn = readdir($DH) ) {
        next if $fn =~ /^\./;
        push @{$fns}, $fn if $fn =~ /\-s/;
    }
    return $fns;
}

unless ( caller ) {
    _main();
}

__END__