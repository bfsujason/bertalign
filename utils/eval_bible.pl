#!/usr/bin/perl

# bfsujason@163.com
# 2021.02.15

use strict;
use warnings;

use 5.010;
use utf8;

use File::Spec;
use Getopt::Long;
use List::Util qw(first uniq);

binmode(STDOUT, ":utf8");

sub main {
	GetOptions( \ my %opts,
        'meta=s',
		'gold=s',
		'auto=s',
        'src_verse=s',
        'tgt_verse=s',
	);
    my $meta = read_meta($opts{meta});
    my $src_sent_verse = read_verse($opts{src_verse});
    my $tgt_sent_verse = read_verse($opts{tgt_verse});
	#foreach my $k ( sort {$a <=> $b} keys %{$src_sent_verse} ) {
	#	say $k, '=>', $src_sent_verse->{$k};
	#}
    foreach my $id ( @{$meta} ) {
        my $auto_align = read_align(
            File::Spec->catfile($opts{auto}, $id . '.align')
        );
        my $gold_align = read_align(
            File::Spec->catfile($opts{gold}, $id . '.align')
        );
        my $merged_auto_align = merge_align(
            $auto_align, $src_sent_verse, $tgt_sent_verse
        );
        #open my $OUT, '>:utf8', 'merged_align';
        #write_align($OUT, $merged_auto_align);
        my ($p, $r, $f1) = _eval($gold_align, $merged_auto_align);
        say join "\t", ($id, $p, $r, $f1);
    } 
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
	my ($gold, $auto) = @_;
    my $gold_align = flatten_align($gold);
    my $auto_align = flatten_align($auto);
	my $intersect = 0;
	foreach my $bead ( @{$gold_align} ) {
		my $match = first {$_ eq $bead} @{$auto_align};
		$intersect++ if $match;
	}
	return $intersect;
}

sub flatten_align {
    my $align = shift;
    my @flattened_align = map { 
        my $src = join ',', @{$_->[0]};
        my $tgt = join ',', @{$_->[1]};
        join '<=>', ($src, $tgt)
   } @{$align};
   return \@flattened_align;
}

sub write_align {
	my ($fh, $align) = @_;
	foreach my $bead ( @{$align} ) {
		my $src = join ",", @{$bead->[0]};
		my $tgt = join ",", @{$bead->[1]};
		say $fh '['.$src .']:['.$tgt.']';
	}
}

sub merge_align {
    my ($align, $src_sent_verse, $tgt_sent_verse) = @_;
    my $merged_align = [];
    my $last_bead_type = '';
    foreach my $bead ( @{$align} ) {
       my $bead_type = find_bead_type(
            $bead, $src_sent_verse, $tgt_sent_verse
        );
        if ( not $last_bead_type ) {
            push @{$merged_align}, $bead;
        } else {
            if ( $bead_type eq $last_bead_type ) {
                push @{$merged_align->[-1]->[0]}, @{$bead->[0]};
                push @{$merged_align->[-1]->[1]}, @{$bead->[1]};
            } else {
                push @{$merged_align}, $bead;
            }
        }
        $last_bead_type = $bead_type;
    }
    return $merged_align;
}

sub find_seg_type {
	my ($seg, $sent2verse) = @_;
	my $seg_len = scalar @{$seg};
	if ( $seg_len == 0 ) {
		return ['NULL'];
	} else {
		my @uniq_seg = uniq map { $sent2verse->{$_} } @{$seg};
		return \@uniq_seg;
	}
}

sub find_bead_type {
    my ($bead, $src_verse, $tgt_verse) = @_;
    my $bead_type = '';
    my $src_seg = $bead->[0];
    my $tgt_seg = $bead->[1];
    
    my $src_seg_type = find_seg_type($src_seg, $src_verse);
	my $tgt_seg_type = find_seg_type($tgt_seg, $tgt_verse);
	
	my $src_seg_len = scalar @{$src_seg_type};
    my $tgt_seg_len = scalar @{$tgt_seg_type};
	
	if ( $src_seg_len != 1 or $tgt_seg_len != 1 ) {
		return $bead_type;
    } else {
		my $src_verse = $src_seg_type->[0];
		my $tgt_verse = $tgt_seg_type->[0];
		if ( $src_verse ne $tgt_verse ) {
			if ( $src_verse eq 'NULL' ) {
				return $tgt_verse;
			} elsif ( $tgt_verse eq 'NULL' ) {
				return $src_verse;
			} else {
				return $bead_type;
			}
        } else {
            return $src_verse;
        }
	}
}

sub _find_bead_type {
    my ($bead, $src_verse, $tgt_verse) = @_;
    my $bead_type = '';
    my $src_seg = $bead->[0];
    my $tgt_seg = $bead->[1];
    
    my $src_seg_len = scalar @{$src_seg};
    my $tgt_seg_len = scalar @{$tgt_seg};
    if ( $src_seg_len == 0 or $tgt_seg_len == 0 ) { # addition OR omission 
        return $bead_type;
    } else {
        my @src_seg_verse = uniq map { $src_verse->{$_} }  @{$src_seg};
        my @tgt_seg_verse = uniq map { $tgt_verse->{$_} } @{$tgt_seg};
        my $src_seg_verse_len = scalar @src_seg_verse;
        my $tgt_seg_verse_len = scalar @tgt_seg_verse;
        if ( $src_seg_verse_len != 1 or $tgt_seg_verse_len != 1 ) {
            return $bead_type;
        } else {
            if ( $src_seg_verse[0] ne $tgt_seg_verse[0] ) {
                return $bead_type;
            } else {
                $bead_type = $src_seg_verse[0];
                return $bead_type;
            }
        }
    }
}

sub read_align {
	my $file = shift;
	my $align = [];
	open my $IN, "<:utf8", $file;
	while ( defined(my $line = <$IN>) ) {
		chomp $line;
        $line =~ s/\s+//g;
		my ($src, $tgt) = split /:/, $line;
		$src =~ s/\[|\]//g;
		$tgt =~ s/\[|\]//g;
		my @src = split /,/, $src;
		my @tgt = split /,/, $tgt;
		push @{$align}, [\@src, \@tgt];
	}
	return $align;
}

sub read_verse {
    my $file = shift;
    my $sent2verse = {};
    open my $IN, '<:utf8', $file;
    while ( defined(my $line = <$IN>) ) {
        chomp $line;
        $sent2verse->{$. - 1} = $line;
    }
    return $sent2verse;
}

sub read_meta {
	my $file = shift;
	my $meta = [];
	open my $IN, '<:utf8', $file;
	while ( defined(my $line = <$IN>) ) {
		next if $. == 1;
        next if $line =~ /^#/;
		chomp $line;
		my @records = split /\t/, $line;
		push @{$meta}, $records[0];
	}
	return $meta;
}

unless ( caller ) {
    main();
}

__END__