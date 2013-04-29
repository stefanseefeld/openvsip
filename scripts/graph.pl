#! /usr/bin/perl
# --------------------------------------------------------------------- #
# scripts/graph.pl -- VSIPL++ Graph Generation Script			#
# (4 Sep 06) Jules Bergmann						#
# --------------------------------------------------------------------- #

use strict;

use vars qw($do_all);
use vars qw($do_evo);

use vars qw($suffix);  $suffix = "";
use vars qw($plotdir); $plotdir = "x";

use vars qw($sys); $sys = {};


# --------------------------------------------------------------------- #
# read_db
sub read_db {
   my ($sys, $file) = @_;

   my $db = {};

   open(FILE, $file) || die "Can't read '$file': $!\n";

   while (<FILE>) {
      s/\#.*$//; next if /^\s*$/;

      if (/^graph:\s*(.+)/) {
         my $page = $1;
         $db->{$page} = {};
         $db->{$page}{lines} = [];
         while (<FILE>) {
	    next if /^\s*\#/;	# A comment doesn't end a page
            s/\#.*$//; 
            last if /^\s*$/;	# An empty line does
            if (/\s+line:\s*([\w\-\/]+)\s+(.+)$/) {
	       push @{$db->{$page}{lines}}, [$1, $2];
	       }
            elsif (/\s+(\w+)\{([\w\-\.]+)\}:\s*(.+)$/) {
	       my $var   = $1;
	       my $guard = $2;
	       my $val   = $3;

	       if ($sys->{id} eq $guard) {
		  $db->{$page}{$var} = $val;
		  if ($var eq 'xrange') {
		     ($db->{$page}{xmin}, $db->{$page}{xmax}) = split(/,/, $val);
		     }
		  }
	       }
            elsif (/\s+(\w+):\s*(.+)$/) {
	       my $var = $1;
	       my $val = $2;

	       my $mhz  = $sys->{mhz};
	       my $tmhz = $sys->{tmhz};
	       my $ghz  = $sys->{ghz};
	       my $tghz = $sys->{tghz};
	       # while ($val =~ /\{([~\}]+)\}/) {
	       while ($val =~ /\{(.+)\}/) {
		  my $expr  = $1;
		  my $value = eval $expr;
		  $expr =~ s/\$/\\\$/g;
		  $expr =~ s/\*/\\\*/g;
		  $val =~ s/\{$expr\}/$value/g;
		  }

               $db->{$page}{$var} = $val;

	       if ($var eq 'xrange') {
		  ($db->{$page}{xmin}, $db->{$page}{xmax}) = split(/,/, $val);
		  }
	       }
	    else {
	       print "UNKNOWN: $_\n";
	       }
	    }
	 }
      }
   close(FILE);
   
   return $db;
   }



# --------------------------------------------------------------------- #
# read_config -- read configure file from data directory.
sub read_config {
   my ($dir) = @_;

   my $cfg = {};
   $cfg->{dir} = $dir;

   my $file = "$dir/config";
   open(FILE, $file) || die "Can't read '$file': $!\n";

   while (<FILE>) {
      if (/\s*(\w+):\s*(.+)$/) {
	 $cfg->{$1} = $2;
	 }
      }
   close(FILE);

   return $cfg;
}



# --------------------------------------------------------------------- #
# convert_data -- convert raw -data output from benchmarks into format
#                 for plotting
sub convert_data {
   my ($inf, $outf, $xwhat, $ywhat, $data, $idx) = @_;

   open(IN, $inf)       || die "Can't read '$inf': $!\n";
   open(OUT, "> $outf") || die "Can't write '$outf': $!\n";

   $data->{$idx} = {};

   my $yscale = 'm';
   if ($ywhat =~ /^(\w+)-(\w+)$/) {
      $ywhat  = $1;
      $yscale = $2;
      }


   my $points = 0;
   while (<IN>) {
      my @line = split(',');

      # skip over header
      next if (scalar(@line) < 9 || $line[0] !~ /^\d+$/);
      
      my $xfactor = 1;
      my $yfactor = 1;
      my $recip   = 0;

      $xfactor = 1                 if $xwhat eq 'points';
      $xfactor = $line[4]          if $xwhat eq 'bytes';

      $yfactor = 1                 if $ywhat eq 'rate';
      $yfactor = $line[5]          if $ywhat eq 'ops';
      $yfactor = $line[5]/1000     if $ywhat eq 'gops';
      $yfactor = $line[6]          if $ywhat eq 'riob';
      $yfactor = $line[7]          if $ywhat eq 'wiob';
      $yfactor = $line[6]+$line[7] if $ywhat eq 'iob';
      $yfactor = (0 + $line[5]) / $sys->{mhz} if ($ywhat eq 'opc'  ||
						  $ywhat eq 'flopc');
      $yfactor = 1, $recip=1 if $ywhat eq 'uspp'; # us per point

      # Giga-X's
      if ($yscale eq 'g') {
	 $yfactor /= 1000;
	 }

      if ($recip == 1) {
	 printf(OUT "%d %f %f %f\n",
		$line[0]       * $xfactor,
		(1.0/$line[1]) * $yfactor,
		(1.0/$line[2]) * $yfactor,
		(1.0/$line[3]) * $yfactor);
	 $points++;
	 
	 $data->{$idx}{$line[0] * $xfactor} = (1.0/$line[1]) * $yfactor;
	 }
      else {
	 printf(OUT "%d %f %f %f\n",
		$line[0] * $xfactor,
		$line[1] * $yfactor,
		$line[2] * $yfactor,
		$line[3] * $yfactor);
	 $points++;
	 
	 $data->{$idx}{$line[0] * $xfactor} = $line[1] * $yfactor;
	 }
      }
   close(OUT);
   close(IN);

   if ($points == 0) {
      printf "WARN: no data in %s\n", $inf;
      }
   return $points;
   }



# --------------------------------------------------------------------- #
sub generate_csv {
   my ($outf, $data) = @_;

   open(OUT, "> $outf") || die "Can't write '$outf': $!\n";

   my @keys = 0 .. $data->{sets}-1;

   my @line = ("");
   foreach my $k (@keys) {
      push @line, $data->{header}{$k};
      }
   printf OUT "%s\n", join(",", @line);

   foreach my $x (sort {$a <=> $b} keys %{$data->{0}}) {
      my @line = ($x);
      foreach my $k (@keys) {
	push @line, $data->{$k}{$x};
	}
      printf OUT "%s\n", join(",", @line);
      }

   close(OUT);
   }
   


# --------------------------------------------------------------------- #
sub generate_ratio {
   my ($outf, $baseline, $idx0, $idx1) = @_;

   open(OUT, "> $outf") || die "Can't write '$outf': $!\n";

   foreach my $x (sort {$a <=> $b} keys %{$baseline->{$idx0}}) {
      my $ratio = $baseline->{$idx1}{$x} / $baseline->{$idx0}{$x};
      printf(OUT "%d %f\n", $x, $ratio);
      }

   close(OUT);
   }


sub use_ymax {
  my ($db, $sys, $page, $ywhat) = @_;

  return $db->{$page}{ymax} ||
         ($ywhat eq 'opc' || $ywhat eq 'flopc') && $sys->{ymax_opc}   ||
         ($ywhat eq 'ops')                      && $sys->{ymax_mflop} ||
         ($ywhat eq 'ops-g')                    && $sys->{ymax_gflop} ||
         $sys->{ymax};
  }

   

# --------------------------------------------------------------------- #
# gen_graph -- generate a chart
sub gen_graph {
  my ($info, $db, $page) = @_;

  if (!defined $db->{$page}) {
     die "gen_graph: no chart for '$page'\n";
     }

  my $pfile = "tmp.plot";
  my $outfile = "$plotdir/$page$suffix.png";

  open(TMP, "> $pfile") || die "Can't write '$pfile': $!";

  my $title = $db->{$page}{title};
  my $what  = $db->{$page}{what} || "ops";

  my ($xwhat, $ywhat);

  if ($what =~ /,/) {
     ($xwhat, $ywhat) = split(/,/, $what);
     }
  else {
     $xwhat = "points";
     $ywhat = $what;
     }

  my $xaxis = $db->{$page}{xaxis};
  my $yaxis = $db->{$page}{yaxis} ||
              ($ywhat eq "ops" && "MFLOP/s") ||
              ($ywhat eq "gops" && "GFLOP/s") ||
              ($ywhat eq "ops-g" && "GFLOP/s") ||
              ($ywhat eq "opc" && "op/cycle") ||
              ($ywhat eq "flopc" && "flop/cycle") ||
              ($ywhat eq "iob" && "MB/s") ||
              ($ywhat eq "iob-g" && "GB/s");
  my $ymax  = use_ymax($db, $sys, $page, $ywhat);
  my $xmin  = $db->{$page}{xmin} || $sys->{xmin} || 16;
  my $xmax  = $db->{$page}{xmax} || $sys->{xmax} || 1048576;

  my $ystart = 0;       # TODO

  print TMP "set logscale x  2\n";
  print TMP "set logscale x2 2\n";
  print TMP "set xrange  [$xmin:$xmax]\n";
  print TMP "set yrange  [$ystart:$ymax]\n";
  if ($info->{ratio} == 1) {
     print TMP "set y2range  [0:2]\n";
     print TMP "set y2tics  auto\n";
     print TMP "set ytics nomirror\n";
     print TMP "set y2label \"Ratio\"\n";
     }

  my $num_configs = scalar(@{$info->{configs}});
  if ($num_configs == 1) {
     $title .= " ($sys->{name} $info->{configs}[0]{name})";
     }
  else {
     $title .= " ($sys->{name})";
     }

  print TMP "set title \"$title\"\n";
  print TMP "set xlabel \"$xaxis\"\n";
  print TMP "set ylabel \"$yaxis\"\n";
  print TMP "set key $info->{key}\n";
  print TMP "set terminal png small\n";
  print TMP "set output \"$outfile\"\n";
  print TMP "set size 1.0,1.0\n";

  my $start = "plot";
  my $lt    = 1;
  my $idx   = 0;
  my $data = {};

  foreach my $ss (@{$info->{configs}}) {
     my $dir  = $ss->{dir};
     my $name = $ss->{name};
     print "config: $name ($xwhat, $ywhat)\n";

     foreach my $gg (@{$db->{$page}{lines}}) {
	my $file = $gg->[0];
	my $desc = $gg->[1];

	my $path = "$dir/$file.dat";
	my @path = split('/', $path);
	$path[$#path] = "tmp.$ywhat." . $path[$#path];
	# my $tmp_file = "$dir/tmp.$ywhat.$file.dat";
	my $tmp_file = join('/', @path);

	my $line_title = "$desc ($dir/$file)";

	if ($num_configs > 1) {
	   $line_title = "$name $desc ($file)";
	   }
	else {
	   $line_title = "$desc ($file)";
	   }

	if (-f "$path") {
	   $data->{header}{$idx} = $desc;
	   my $has_data = convert_data($path, $tmp_file, $xwhat, $ywhat, $data, $idx);
	   next if !$has_data;
	   $idx += 1;
	   print TMP "$start '$tmp_file' u 1:2:3:4 w yerrorlines t \"$line_title\" lt $lt\\\n";
	   $start = "   ,";
	   $lt += 1;
	   $lt = 3 if $lt == 2;
	   }
	}
  }
  $data->{sets} = $idx;

  generate_csv("$plotdir/$page$suffix.csv", $data);

  if ($info->{ratio} == 1) {
     my $tmp_file   = "tmp.baseline.dat";
     my $line_title = "$info->{ratio_title} (right axis)";
     generate_ratio($tmp_file, $data, 0, 1);
     print TMP "$start '$tmp_file' axis x1y2 w lines t \"$line_title\" lt $lt\\\n";
     }

  close(TMP);

  system("gnuplot $pfile");
}


sub system_parameters {
   my ($sys, $arg) = @_;

   $sys->{id} = $arg;

   print "system: $arg\n";
   if ($arg eq 'ppc7400-0.4') {
      $sys->{name}       = "PowerPC 7410 400 MHz";
      $sys->{ghz}        = 0.4;
      $sys->{ymax_gflop} = 1.6;
      $sys->{ymax_opc}   = 8;
      }
   elsif ($arg eq 'ppc7447-1.0') {
      $sys->{name}       = "PowerPC 7447A 1000 MHz";
      $sys->{ghz}        = 1.0;
      $sys->{tghz}       = 1.0;
      $sys->{ymax_gflop} = 4.0;
      $sys->{ymax_opc}   = 8;
      }
   elsif ($arg eq 'ppc970fx-2.0') {
      $sys->{name}       = "PowerPC 970FX 2.0 GHz";
      $sys->{ghz}        =  2.0;
      $sys->{tghz}       =  2.0;
      $sys->{ymax_gflop} = 10.0;
      $sys->{ymax_opc}   =  8;
      }
   elsif ($arg eq 'p4m-2.0') {
      $sys->{name}       = "Pentium-4M 2.0 GHz";
      $sys->{ghz}        = 2.0;
      $sys->{ymax_gflop} = 10.0;
      }
   elsif ($arg eq 'p4-3.0') {
      $sys->{name} = "Pentium-4 3.0 GHz";
      $sys->{ghz}        = 3.0;
      $sys->{ymax_gflop} = 8.0;
      }
   elsif ($arg eq 'p4x-3.6') {
      $sys->{name} = "Pentium-4-64 3.6 GHz";
      $sys->{ghz}        =  3.6;
      $sys->{ymax_gflop} = 10.0;
      $sys->{ymax_opc}   = 2;
      }
   elsif ($arg eq 'p4xeon-32-3.6') {
      $sys->{name} = "Pentium-4-32 3.6 GHz";
      $sys->{ghz}        =  3.6;
      $sys->{ymax_gflop} = 10.0;
      $sys->{ymax_opc}   = 2;
      }
   elsif ($arg eq 'p4x-32-2.8') {
      $sys->{name} = "Pentium-4-32 2.8 GHz";
      $sys->{ghz}        =  2.8;
      $sys->{ymax_gflop} = 10.0;
      $sys->{ymax_opc}   = 2;
      }
   elsif ($arg eq 'broadcom-1480-1.0') {
      $sys->{name} = "Broadcom 1480 950 MHz";
      $sys->{ghz}        = 0.95;
      $sys->{ymax_gflop} = 2.0;
      $sys->{ymax_opc}   = 2;
      }
   elsif ($arg eq 'cbe-3.2') {
      $sys->{name} = "Cell BE 3.2 GHz";
      $sys->{ghz}        = 3.2;
      $sys->{tghz}       = 1.6;
      $sys->{ymax_gflop} = 2.0;
      $sys->{ymax_opc}   = 64;
      }
   else {
      print "system '$arg' not recognized\n";
      }

   $sys->{mhz}  = $sys->{ghz}  * 1000;
   $sys->{tmhz} = $sys->{tghz} * 1000;
   }



# --------------------------------------------------------------------- #
# Main Processing
# --------------------------------------------------------------------- #

my $info = {};

$info->{configs}     = [];
$info->{ratio}       = 0;
$info->{ratio_title} = "Ratio";
$info->{key}         = "top right";

$sys->{id} = undef;


my @pages = ();
my @dirs  = ();
my $dbfile = "graph.db";

while (@ARGV) {
   my $arg = shift @ARGV;

   $do_all        = 1, next if ($arg eq '-all');
   $do_evo        = 1, next if ($arg eq '-evo');
   $info->{ratio} = 1, next if ($arg eq '-ratio');
   $suffix      = shift @ARGV, next if ($arg eq '-suffix');
   $plotdir     = shift @ARGV, next if ($arg eq '-plotdir');
   $dbfile      = shift @ARGV, next if ($arg eq '-db');
   $sys->{ymin} = shift @ARGV, next if ($arg eq '-ymin');
   $sys->{ymax} = shift @ARGV, next if ($arg eq '-ymax');
   $sys->{xmin} = shift @ARGV, next if ($arg eq '-xmin');
   $sys->{xmax} = shift @ARGV, next if ($arg eq '-xmax');
   $info->{key} = shift @ARGV, next if ($arg eq '-key');
   push(@dirs, shift @ARGV),   next if ($arg eq '-dir');
   if ($arg eq '-ratio_title') {
      $info->{ratio}       = 1;
      $info->{ratio_title} = shift @ARGV;
      next;
      }
   if ($arg eq '-sys' || $arg eq '-system') {
      my $arg = shift @ARGV;
      system_parameters($sys, $arg);
      next;
      }
   push @pages, $arg;
   }

foreach my $dir (@dirs) {
   my $cfg;
   if (-f "$dir/config") {
      $cfg = read_config($dir);
      }
   else {
      $cfg = { dir => $dir, name => "*no-config*" };
   }
   if (!defined $sys->{id} && defined $cfg->{system}) {
      system_parameters($sys, $cfg->{system});
      }
   push @{$info->{configs}}, $cfg;
   }

my $db = read_db($sys, $dbfile);

if ($do_all) {
   foreach my $page (keys %$db) {
      push @pages, $page if ($page !~ /^evo/)
      }
   }
if ($do_evo) {
   foreach my $page (keys %$db) {
      push @pages, $page if ($page =~ /^evo/)
      }
   }

foreach my $page (@pages) {
   gen_graph($info, $db, $page);
   }
