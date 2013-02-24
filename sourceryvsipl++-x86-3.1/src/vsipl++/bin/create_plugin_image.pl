#! /usr/bin/perl
#########################################################################
# create_plugin_image.pl -- create binary plugin image from objdump	#
# Jules Bergmann, CodeSourcery, Inc					#
# Feb 12, 2009								#
#########################################################################

use strict;

sub create_plugin_image {

   printf("warning: 'create_plugin_image.pl' is deprecated; use 'create_plugin' instead\n");

   my ($inf, $outf) = @_;

   open(OUT, "> $outf") and binmode OUT || die "Can't write $outf: $!\n";

   # 1. Find addresses of functions
   #
   #    This is done be readying through the objdump file once, looking
   #    for all functions, which start with the form:
   #
   #    00004000 <input>:
   #        4000:       34 00 02 84     lqd     $4,0($5)

   open(IN, $inf) || die "Can't read $inf: $!\n";

   my %func;
   while (<IN>) {
      if (/([0-9a-f]+) <(.+)>:/) {
	 $func{$2} = hex($1);
	 }
      }
   close(IN);


   # 2. Write out header block.
   #
   #    The first 128 bytes of the image contain header information.
   #    Currently this contains the locations of the kernel, input, and
   #    output functions (as found during step 1).  The remainder is
   #    zero padded.

   print OUT pack 'N', $func{"kernel"};
   print OUT pack 'N', $func{"input"};
   print OUT pack 'N', $func{"output"};
   for (my $i=0; $i<32-3; $i+=1) {
     print OUT pack 'N*', 0;
     }

   # 3. Translate opcodes/data
   #
   #    In a second pass through objdump output, translate the text,
   #    data, rodata, and ctors sections into a binary image.

   open(IN, $inf) || die "Can't read $inf: $!\n";

   my $section;
   my $size = 0;
   my $last_addr = hex("4000") - 4;

   my $conv = 0;

   while (<IN>) {
      if (/Disassembly of section \.(.+):/) {
         $section = $1;

	 if ($section =~ /^text/ ||
	     $section =~ /^data/ ||
	     $section =~ /^rodata/ ||
	     $section =~ /^ctors/) {
     	    $conv = 1;
	    }
	 else {
     	   $conv = 0;
	   }

	 next;
	 }
      next if $conv == 0;

      if (/^\s*([0-9a-f]+):\s*([0-9a-f][0-9a-f]) ([0-9a-f][0-9a-f]) ([0-9a-f][0-9a-f]) ([0-9a-f][0-9a-f])/) {
	 my $addr = hex($1);
	 my $b1 = hex($2);
	 my $b2 = hex($3);
	 my $b3 = hex($4);
	 my $b4 = hex($5);

	 # Sometimes there are gaps in the objdump file.  Find them and
	 # insert zero padding as necessary.
	 while ($addr - $last_addr > 4) {
	    print OUT pack 'C*', 0, 0, 0, 0;
	    $last_addr += 4;
	    $size += 4;
	    }

	 print OUT pack 'C*', $b1, $b2, $b3, $b4;
	 $size += 4;
	 $last_addr = $addr;
	 }
      }

   close(IN);
   close(OUT);

   # Report code size.
   printf("size (hdr: %d  code: %d  total: %d)\n", 128, $size, $size+128);
   }
   



my ($infile, $outfile) = @ARGV;

create_plugin_image($infile, $outfile);
