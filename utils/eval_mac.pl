#!/usr/bin/perl

# bfsujason@163.com
# 2021.02.02

# This script evaluates the performance of sentence alignment algorithms.

# Usage:
# perl eval.pl --meta ../corpus/test/meta_data.tsv --gold ../corpus/test/gold --auto ../corpus/test/auto --by book|chapter|align

use strict;
use warnings;

use 5.010;
use utf8;

use File::Spec;
use Getopt::Long;
use List::Util qw(first);

binmode(STDOUT, ":utf8");

sub main {
	GetOptions( \ my %opts,
		'meta=s',
		'gold=s',
		'auto=s',
		'by=s'
	);
	my $meta = read_meta($opts{meta});
	my $gold = [];
    my $auto = [];
	foreach my $record ( @{$meta} ) {
		my $text_id = $record->[0];
		my $book_id = $record->[1];
		my $cur_gold = read_align(File::Spec->catfile($opts{gold}, $text_id . '.align'), $text_id, $book_id);
		my $cur_auto = read_align(File::Spec->catfile($opts{auto}, $text_id . '.align'), $text_id, $book_id);
        push @{$gold}, @{$cur_gold};
        push @{$auto}, @{$cur_auto};
	}
    
    my ($p, $r, $f1) = _eval($gold, $auto);
    say "\nOveral performance:";
    say join "\t", ("P: $p", "R: $r", "F1: $f1");
    
    my $gold_by_group = _group_align($gold, $opts{by});
    my $auto_by_group = _group_align($auto, $opts{by});
    
    say "\nPerformance by $opts{by}:";
    foreach my $k ( sort {$a <=> $b} keys %{$gold_by_group} ) {
        my ($p, $r, $f1) = _eval($gold_by_group->{$k}, $auto_by_group->{$k});
        say join "\t", ($k, "P: $p", "R: $r", "F1: $f1");
    }
}

sub _group_align {
	my ($align, $by) = @_;
	my $align_by_group = {};
	my $group_id;
	if ( $by eq 'book' ) {
		$group_id = 0;
    } elsif ( $by eq 'chapter' ) {
		$group_id = 1;
	} elsif ( $by eq 'align' ) {
		$group_id = 3;
	}
	
	foreach my $item ( @{$align} ) {
		my @records = split /\|\|/, $item;
        push @{$align_by_group->{$records[$group_id]}}, $item;
    }
    return $align_by_group;
}

sub _group_align_old {
    my $align = shift;
    my $align_by_book = {};
    my $align_by_type = {};
    foreach my $item ( @{$align} ) {
        my ($book_id, $text_id, $bead, $type) = split /\|\|/, $item;
        push @{$align_by_book->{$book_id}}, $item;
        push @{$align_by_type->{$type}}, $item;
    }
    return ($align_by_book, $align_by_type);
}

sub _eval {
    my ($gold, $auto) = @_;
    my $intersect = find_intersect($gold, $auto);
    my $gold_num = scalar @{$gold};
    my $auto_num = scalar @{$auto};
    my ($p, $r, $f1) = (0, 0, 0);
    $p = sprintf("%.3f", $intersect / $auto_num);
	$r = sprintf("%.3f", $intersect / $gold_num);
    if ( $p + $r > 0 ) {
        $f1 = sprintf("%.3f", (2 * $p * $r) / ($p + $r));
    }
	return ($p, $r, $f1);
}

sub find_intersect {
	my ($gold_align, $auto_align) = @_;
	my $intersect = 0;
	foreach my $bead ( @{$gold_align} ) {
		my $match = first {$_ eq $bead} @{$auto_align};
		$intersect++ if $match;
	}
	return $intersect;
}

# parse align file
sub read_align {
	my ($auto_align_fn, $text_id, $book_id) = @_;
	my $auto_align = [];
	open my $IN, '<:utf8', $auto_align_fn;
	while ( defined(my $bead = <$IN>) ) {
		chomp $bead;
		$bead =~ s/\s+//g;
        my ($src, $tgt) = split /:/, $bead;
        my $src_type = get_seg_type($src);
		my $tgt_type = get_seg_type($tgt);
		#my $seg_type = join "<=>", ($src_type, $tgt_type);
        my $seg_type = $src_type + $tgt_type;
        $bead = join "||", ($book_id, $text_id, $bead, $seg_type);
		push @{$auto_align}, $bead;
	}
	return $auto_align;
}

sub get_seg_type {
	my $seg = shift;
	my $type = 0;
	if ( $seg ne '[]' ) {
        my @idx = split /\,/,$seg;
		$type = scalar @idx;
	}
	return $type;
}

# parse metadata file
sub read_meta {
	my $meta_fn = shift;
	my $meta = [];
	open my $IN, '<:utf8', $meta_fn;
	while ( defined(my $line = <$IN>) ) {
		next if $. == 1;
     next if $line =~ /^#/;
		chomp $line;
		my @records = split /\t/, $line;
		push @{$meta}, [$records[0], $records[1]];
	}
	return $meta;
}

unless ( caller ) {
	main();
}

__END__