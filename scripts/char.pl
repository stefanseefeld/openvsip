#! /usr/bin/perl
# --------------------------------------------------------------------- #
# scripts/char.pl -- VSIPL++ Characterization Script			#
# (4 Sep 06) Jules Bergmann						#
# --------------------------------------------------------------------- #

use strict;

my $exe      = "";
my $bdir     = "benchmarks";
my $base_opt = "-samples 3";
my $base_extra = "";
my $make_cmd = "make";
my $make_opt = "";
my @db_files = ();
my $default_db_file  = "char.db";
my $abs_path = 0;		# if 1, run program using absolute path

my $mode     = 'normal';
my $force    = 0;
my $fast     = 0;
my $dry      = 0;
my $all      = 0;
my $hpec     = 0;
my $sheet    = 0;
my $runonly  = 0;               # if 1, run only, do not attempt to build



# --------------------------------------------------------------------- #
sub read_db {
   my ($db, $macro, $file) = @_;


   open(FILE, $file) || die "Can't read '$file': $!\n";

   while (<FILE>) {
      if (/^set:\s*([\w_\-]+)/) {
	 my $set = $1;
	 $db->{$set} = {};
	 while (<FILE>) {
            last if /^\s*$/;
	    if (/\s+(\w+):\s*(.+)$/) {
	       $db->{$set}{$1} = $2;
	       }
	    }
	 }
      elsif (/^macro:\s*([\w-]+)\s+(.+)$/) {
	 $macro->{$1} = [split(/\s+/, $2)];
	 }
      }
   close(FILE);

   return $db, $macro;
}



# --------------------------------------------------------------------- #
sub run_set {
   my ($info, $db, $set) = @_;

   print "run_set: $set\n";

   if (!defined $db->{$set}) {
      print "set $set: not defined\n";
      return;
      }

   my $x      = $db->{$set};
   my $pgm    = $db->{$set}{pgm};
   my $cases  = $db->{$set}{cases};
   my $fastc  = $db->{$set}{fastcases};
   my $hpecc  = $db->{$set}{hpeccases};
   my $sheetc = $db->{$set}{sheetcases};
   my $nps    = $db->{$set}{nps}  || "1";
   my $spes   = $db->{$set}{spes} || "0 1 8 16";
   my $what   = "data"; # $db->{$set}{what} || "ops";
   my $suffix = $db->{$set}{suffix} || "";
   my $extra  = $db->{$set}{extra} || "";
   my $req    = $db->{$set}{req};

   # check reqs
   foreach my $r (split(/\s+/, $req)) {
      if (!$info->{req}{$r}) {
	 print "set $set: missing req $req\n";
	 return;
	 }
      }

   if ($fast && $fastc) {
      $cases = $fastc;
   }
   if ($hpec && $hpecc) {
      $cases = $hpecc;
   }
   if ($sheet && $sheetc) {
      $cases = $sheetc;
   }

   my $full_pgm = "$bdir/$pgm$exe";
   my $pwd = `pwd`; $pwd =~ s/[\n\r]//;
   $full_pgm = "$pwd/$full_pgm" if ($abs_path);

   my $pgm_name = $pgm;
   $pgm_name =~ s/\//-/;

   my $opt = "-$what $base_opt $base_extra $extra";

   # do not attempt to build missing benchmarks when in run-only mode
   if (!-f $full_pgm) {
      if ($runonly) {   
         print "'$bdir/$pgm$exe' not built when -runonly used - SKIP\n";
         return;
         }
      }

   # save time by skipping benchmarks that would not build before
   if (-f "log.cannot_build.$pgm") {
      print "'$bdir/$pgm$exe' previously failed to build - SKIP\n";
      return;
      }

   # 1. make benchmark (if necessary)
   if (!-f $full_pgm) {
      my $target = "$bdir/$pgm$exe";
      print "MAKE $target\n";
      system("echo $make_cmd $make_opt $target > log.make.$pgm_name");
      system("$make_cmd $make_opt $target 2>&1 >> log.make.$pgm_name");
      }

   if (!-x "$bdir/$pgm$exe") {
      system("touch log.cannot_build.$pgm");
      print "'$bdir/$pgm$exe' not executable";
      return;
      }

   $nps = join(' ', @{$info->{nps}}) if ($nps eq 'all');

   if ($mode eq 'cell' || $mode eq 'cml') {
      foreach my $x (split(/\s+/, $cases)) {
	 foreach my $np (split(/\s+/, $nps)) {
	    foreach my $spe (split(/\s+/, $spes)) {
	       next if !defined $info->{np}{$np};
	       next if !defined $info->{spe}{$spe};
	       my $outfile = "$pgm_name$suffix-$x-$np-$spe.dat";
	    
	       run_benchmark(full_pgm => $full_pgm,
			     x        => $x,
			     np       => $np,
			     spe      => $spe,
			     opt      => $opt,
			     outfile  => $outfile);
	       }
	    }
	 }
      }
   else {
      foreach my $x (split(/\s+/, $cases)) {
	 foreach my $np (split(/\s+/, $nps)) {
	    next if !defined $info->{np}{$np};
	    my $outfile = "$pgm_name$suffix-$x-$np.dat";
	    
	    run_benchmark(full_pgm => $full_pgm,
			  x        => $x,
			  np       => $np,
			  opt      => $opt,
			  outfile  => $outfile);
	    }
	 }
      }
}



sub run_benchmark {
   my (%opt) = @_;

   my $full_pgm = $opt{full_pgm};
   my $x        = $opt{x};
   my $np       = $opt{np};
   my $spe      = $opt{spe};
   my $opt      = $opt{opt};
   my $outfile  = $opt{outfile};

   if (-f $outfile && !$force ) {
      print "$outfile already generated\n";
      next;
      }

   print "$outfile:\n";

   my $runcmd = "";
   my $runopt = "";
   if ($mode eq 'mpi') {
      $runcmd = "mpirun N -np $np";
      }
   elsif ($mode eq 'pas') {
      $runcmd = "run-pas.sh -np $np";
      }
   elsif ($mode eq 'mc') {
      $runcmd = "run.sh";
      }
   elsif ($mode eq 'cell') {
      $runcmd = "";
      $runopt = "--svpp-num-spes $spe";
      }
   elsif ($mode eq 'cml') {
      $runcmd = "";
      $runopt = "--cml-num-spes $spe";
      }
   
   my $cmd = "$runcmd $full_pgm -$x $opt $runopt > $outfile";
   print "CMD $cmd\n";
   if (!$dry) {
      system($cmd);

      system("echo '---' >> $outfile");
      $cmd = "$runcmd $full_pgm -$x $opt $runopt -diag >> $outfile";
      system($cmd);

      system("echo '---' >> $outfile");
      $cmd = "$runcmd $full_pgm -$x $opt $runopt -lib_config >> $outfile";
      system($cmd);
      }
   }


sub expand {
   my ($macro, @sets) = @_;

   my @rsets = ();

   foreach my $set (@sets) {
      if ($macro->{$set}) {
	 print "MACRO $set\n";
	 push @rsets, expand($macro, @{$macro->{$set}});
	 }
      else {
	 push @rsets, $set;
	 }
      }
   return @rsets;
   }
      
	   


# --------------------------------------------------------------------- #
# main

my $info = {};
$info->{req}  = {};
$info->{nps}  = [1];
$info->{spes} = [8];

my @sets = ();

while (@ARGV) {
   my $arg = shift @ARGV;

   $force    = 1, next if ($arg eq '-force');
   $sheet    = 1, next if ($arg eq '-sheet');
   $hpec     = 1, next if ($arg eq '-hpec');
   $fast     = 1, next if ($arg eq '-fast');
   $dry      = 1, next if ($arg eq '-dry');
   $all      = 1, next if ($arg eq '-all');
   $runonly  = 1, next if ($arg eq '-runonly');
   $abs_path = 1, next if ($arg eq '-abs_path');
   $bdir     = shift @ARGV, next if ($arg eq '-bdir');
   $make_opt = shift @ARGV, next if ($arg eq '-make_opt');
   $make_cmd = shift @ARGV, next if ($arg eq '-make_cmd');
   push(@db_files, shift @ARGV), next if ($arg eq '-db');
   $exe        = shift @ARGV, next if ($arg eq '-exe');
   $base_extra = shift @ARGV, next if ($arg eq '-extra');
   $base_opt .= " -ms " . shift @ARGV, next if ($arg eq '-ms');
   $base_opt .= " -pool " . shift @ARGV, next if ($arg eq '-pool');
   if ($arg eq '-mode') {
      $mode = shift @ARGV;
      die "Unknown mode: $mode" if ($mode !~ /(mpi|pas|mc|cell|cml|normal)/);
      next;
      }
   if ($arg eq '-np') {
      my $str = shift @ARGV;
      if ($str =~ /^(\d+)-(\d+)$/) {
	 $info->{nps} = [$1 .. $2];
	 }
      else {
	 $info->{nps} = [split(',', $str)];
	 }
      next;
      }
   if ($arg eq '-spe') {
      my $str = shift @ARGV;
      if ($str =~ /^(\d+)-(\d+)$/) {
	 $info->{spes} = [$1 .. $2];
	 }
      else {
	 $info->{spes} = [split(',', $str)];
	 }
      next;
      }
   if ($arg eq '-have') {
      my $reqs = shift @ARGV;
      foreach my $r (split(',', $reqs)) {
	 $info->{req}{$r} = 1;
	 }
      next;
      }
   push @sets, $arg;
   }

foreach my $np (@{$info->{nps}}) {
   $info->{np}{$np} = 1;
   }

foreach my $spe (@{$info->{spes}}) {
   $info->{spe}{$spe} = 1;
   }


my $db    = {};
my $macro = {};

push @db_files, $default_db_file if (scalar(@db_files) == 0);

foreach my $db_file (@db_files) {
   print "db: $db_file\n";
   read_db($db, $macro, $db_file);
   }

if ($sheet) {
   @sets = 'sheet';
   }
elsif ($hpec) {
   @sets = 'hpec';
   }
elsif ($all) {
   @sets = keys %$db;
   }


my @rsets = expand($macro, @sets);

foreach my $set (@rsets) {
   run_set($info, $db, $set);
   }
