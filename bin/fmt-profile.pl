#! /usr/bin/perl

#########################################################################
# fmt-profile.pl -- Format VSIPL++ profiler output			#
#									#
# author: Jules Bergmann						#
# date:   2006-11-01							#
#									#
# Usage:								#
#   fmt-profile.pl [-sec] [-sum] [-extra <event>] [-o <out.txt>]	#
#                  <profile.txt>					#
#									#
# Options:								#
#   -sec	-- convert ticks into seconds				#
#   -sum	-- sum nested operations for events with 0 ops		#
#   -extra <event>							#
#		-- create pseudo event for unaccounted-for time		#
#		   in nested events under <event>			#
#   -o <out.txt>							#
#		-- write output to <out.txt> (instead of overwriting	#
#		   <profile.txt).					#
#########################################################################

use strict;

use vars qw($indent); $indent = 2;

# --------------------------------------------------------------------- #
# get_info -- get format info
# --------------------------------------------------------------------- #
sub get_info {
    my $info = {};

    $info->{0} = {fmt => '%-${len}s ', maxlen => 0 };
    $info->{1} = {fmt => '%${len}d',  maxlen => 0 };
    $info->{2} = {fmt => '%${len}d',  maxlen => 0 };
    $info->{3} = {fmt => '%${len}d',  maxlen => 0 };
    $info->{4} = {fmt => '%${len}f',  maxlen => 0 };

    return $info;
}



# --------------------------------------------------------------------- #
# fmt_profile -- format a profile file
# --------------------------------------------------------------------- #
sub fmt_profile {
   my ($file, %opt) = @_;

   my $tmp_file = "$file.tmp";

   open(IN, $file)           || die "Can't read '$file': $!\n";

   my @key;
   my $clocks_per_sec = 1;
   my $conv_tick = 0;

   my $info = get_info();

   my @hdr;
   my $data = { sub => {} };

   # ------------------------------------------------------------------
   # Pass 1: determine column widths
   # ------------------------------------------------------------------
   while (<IN>) {
      chomp;

      if (/# clocks_per_sec: (\d+)/) {
	 $clocks_per_sec = $1;
	 push @hdr, $_;
	 next;
      }
      elsif (/#\s*(tag\s*:.*)$/) {
	 my $keys = $1;
         $keys = "tag:ticks:calls:ops:mop/s" if ($keys =~ /total ticks/);
	 @key = split(':', $keys);
	 foreach my $i (0 .. $#key) {
	     $key[$i] =~ s/^\s+//;
	     $key[$i] =~ s/\s+$//;
	     my $fmt = "%s";
	     $fmt = ($i == 0) ? "$fmt " : " $fmt ";
	     my $str = sprintf("$fmt", $key[$i]);
	     my $len = length($str);
	     $info->{$i}{maxlen} = $len if ($len > $info->{$i}{maxlen})
	 }

	 if ($key[1] eq "ticks") {
	     if ($opt{conv_tick}) {
		 $conv_tick = 1;
		 $key[1] = "secs";
		 $info->{1} = {fmt => '%${len}f' };
	     }
	     else {
		 $info->{1} = {fmt => '%${len}d' };
	     }
	 }
	 else {
	     $info->{1} = {fmt => '%${len}f' };
	 }
	 next;
      }

      push(@hdr, $_), next if (/^\s*\#/);

      s/::/NOT_A_COLON/g;
      my @val = split(':');

      foreach my $i (0 .. $#val) {
	 $val[$i] =~ s/NOT_A_COLON/::/g;
	 $val[$i] =~ s/^\s+//;
	 $val[$i] =~ s/\s+$//;
     }

      my @path = split(/\\,/, $val[0]);
      my $depth = scalar(@path) * 2;
      $val[0] = (" " x $depth) . $path[$#path];

      $val[1] = sprintf("%f", $val[1] / $clocks_per_sec) if ($conv_tick);

      foreach my $i (0 .. $#val) {
	 my $len = ""; # $info->{$i}{maxlen};
	 my $fmt = eval('"' . $info->{$i}{fmt} . '"');
	 $fmt = ($i == 0) ? "$fmt " : " $fmt ";
	 my $str = sprintf("$fmt", $val[$i]);

	 $len = length($str);
	 $info->{$i}{maxlen} = $len if ($len > $info->{$i}{maxlen})
	 }

      my $x = $data;
      foreach my $k (@path) {
	  if (!defined $x->{sub}{$k}) {
	      $x->{sub}{$k} = { name => $k, sub => {}, entry => [] };
	  }
	  $x = $x->{sub}{$k};
      }

      $x->{entry} = \@val;
   }

   close(IN);


   # ------------------------------------------------------------------
   # Pass 2: pretty-print
   # ------------------------------------------------------------------

   open(OUT, "> $tmp_file") || die "Can't write '$tmp_file': $!\n";

   # header
   foreach my $line (@hdr) {
       print OUT "$line\n";
   }

   # column keys
   my @line;
   foreach my $i (0 .. $#key) {
       my $len = $info->{$i}{maxlen};
       my $fmt = "%${len}s";
       $fmt = ($i == 0) ? "$fmt " : " $fmt ";
       my $str = sprintf("$fmt", $key[$i]);
       push @line, $str;
   }
   print OUT "#" . join(":", @line) . "\n";

   sum_tree($data) if ($opt{sum});
   find_extra_time($data, $opt{extra});

   # Entries
   dump_tree($info, $data, 0);

   close(OUT);

   my $outfile = $opt{o} || $file;
   print "OUTFILE: $outfile\n";
   system("mv $tmp_file $outfile");
}



# --------------------------------------------------------------------- #
# sum_tree -- sum accum tree
# --------------------------------------------------------------------- #
sub sum_tree {
    my ($data) = @_;

    my $sum_ops  = 0;
    foreach my $k (keys %{$data->{sub}}) {
	sum_tree($data->{sub}{$k});
	$sum_ops += $data->{sub}{$k}{entry}[2] *
	            $data->{sub}{$k}{entry}[3];
    }

    if ($data->{entry}[2] == 0) {
	$data->{entry}[2] = 1;
	}

    if ($data->{entry}[3] == 0) {
	$data->{entry}[3] = $sum_ops / $data->{entry}[2];
	if ($data->{entry}[1] != 0) {
	    $data->{entry}[4] = $sum_ops / (1e6 * $data->{entry}[1]);
	}
	else {
	    $data->{entry}[4] = 0;
	}
    }
}



# --------------------------------------------------------------------- #
# find_extra_time -- Find extra time in tree nodes
#
# Args:
#  data		- tree
#  nodes	- nodes to find extra time in
# --------------------------------------------------------------------- #
sub find_extra_time {
    my ($data, $nodes) = @_;

    my $sum_time  = 0;
    foreach my $k (keys %{$data->{sub}}) {
	find_extra_time($data->{sub}{$k}, $nodes);
	$sum_time += $data->{sub}{$k}{entry}[1];
    }

    if (defined $nodes->{$data->{name}}) {
	$data->{sub}{extra} = { name => "*extra-time*",
				sub  => {},
				entry => ["*extra-time*",
					  $data->{entry}[1] - $sum_time,
					  0, 0, 0] };
    }
}



# --------------------------------------------------------------------- #
# dump_tree -- dump accum tree
# --------------------------------------------------------------------- #
sub dump_tree {
    my ($info, $data, $depth) = @_;

    if (defined $data->{entry}) {
	my @entry = @{$data->{entry}};
	$entry[0] = (' ' x $depth) . $data->{name};
	my @result;
	foreach my $i (0 .. $#entry) {
	    my $len = $info->{$i}{maxlen};
	    my $fmt = eval('"' . $info->{$i}{fmt} . '"');
	    $fmt = ($i == 0) ? "$fmt " : " $fmt ";
	    my $str = sprintf("$fmt", $entry[$i]);
	    push @result, $str;
	}
	print OUT join(":", @result) . "\n";
    }

    foreach my $k (sort { $data->{sub}{$b}{entry}[1] <=>
			  $data->{sub}{$a}{entry}[1] }
		   keys %{$data->{sub}}) {
	dump_tree($info, $data->{sub}{$k}, $depth + $indent);
    }
}



# --------------------------------------------------------------------- #
my %opt;
my @files;
$opt{extra} = {};

while (@ARGV) {
   my $arg = shift @ARGV;

   if ($arg eq '-sec') {
      $opt{conv_tick} = 1;
      next;
      }
   elsif ($arg eq '-sum') {
      $opt{sum} = 1;
      next;
      }
   elsif ($arg eq '-extra') {
      $opt{extra}{shift @ARGV} = 1;
      next;
      }
   elsif ($arg eq '-o') {
      $opt{o} = shift @ARGV;
      next;
      }
   else {
       push @files, $arg;
   }
}

if ($opt{o}) {
    if (scalar(@files) > 1) {
	die "Too many files specified with -o option.";
    }
    fmt_profile($files[0], %opt);
}
else {
    foreach my $file (@files) {
	fmt_profile($file, %opt);
    }
}
	
