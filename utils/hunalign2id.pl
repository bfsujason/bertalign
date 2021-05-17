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
	# convert text output to sentence IDs
    _para2id_batch($opts{in_dir}, $opts{out_dir});
}

sub _para2id_batch {
    my ($in_dir, $out_dir) = @_;
    my $fns = _get_align_fns($in_dir);
    foreach my $fn ( @{$fns} ) {
        my $in_path  = File::Spec->catfile($in_dir, $fn);
		my $out_path = File::Spec->catfile($out_dir, $fn);
        my $ids = _para2id($in_path);
        open my $OUT, '>:utf8', $out_path;
        say $OUT join "\n", @{$ids};
    } 
}

sub _para2id {
	my $text = shift;
	my @para_id;
	open my $IN, '<:utf8', $text;
	my $src_id = -1;
	my $tgt_id = -1;
	while ( defined(my $line = <$IN>) ) {
		chomp $line;
		#say $line;
		my ($src, $tgt, $score) = split /\t/, $line;
		next if not $src and not $tgt; # skip empty line
		my ($src_len, $src_seg_id) = _seg2id($src, $src_id);
		my ($tgt_len, $tgt_seg_id) = _seg2id($tgt, $tgt_id);
		$src_id += $src_len;
		$tgt_id += $tgt_len;
		push @para_id, join ':', ($src_seg_id, $tgt_seg_id);
	}
	return \@para_id;
}

sub _seg2id {
	my ($text, $id) = @_;
	my @seg = split /\s+~~~\s+/, $text;
	my $len = scalar @seg;
	if ( $len > 0 ) {
		my @seg_id = map { $id + $_ } ( 1 .. $len);
        my $_seg   = join ',', @seg_id;
        $_seg      = '[' . $_seg . ']';
		return $len, $_seg;
	} else {
        return $len, '[]';
	}
}

sub _get_align_fns {
	my $dir = shift;
	my $fns = [];
	opendir(my $DH, $dir);
	while ( my $fn = readdir $DH ) {
		next if $fn =~ /^\./;
        #say $fn;
		push @{$fns}, $fn;
	}
	return $fns;
}

unless ( caller ) {
	_main();
}

__END__