from qm import host
import sys
from telnetlib import Telnet
import os
import re
from stat import *

class Connection(Telnet):


    def __init__(self, host, verbose):

        Telnet.__init__(self, host)
        self.verbose = verbose


    def read_until(self, str):

        what = Telnet.read_until(self, str)
        if self.verbose:
            print what
        return what


    def read_all(self):
        what = Telnet.read_all(self)
        if self.verbose:
            print what
        return what


    def read_reply(self):
        what = self.read_until("-> ");
        return_code = 0
        lines = what.splitlines()
        for line in lines:
            if not self.verbose and re.match("^runmc error:", line):
                return_code = 1
            elif not self.verbose and re.match("^runmc", line):
                continue
            elif not self.verbose and re.match("^value =", line):
                continue
            elif not self.verbose and re.match("^-> $", line):
                continue
            elif re.match("CE\s+\d+> Process 0x[a-f\d]+ on CE \d+ killed at", line):
                return_code = 1
            print line
        return return_code



class VxWorks(host.Host):

    def Run(self, path, arguments, environment = None, timeout = -1):

	verbose = True
        basename = str(os.path.basename(path))
        dirname = str(os.path.dirname(path))
        ce = '-ce 2'
        reset_ce = '-ce 2'
        
	if verbose:
            print 'pwd     : %s' % dirname
            print 'cmd     : %s %s' % (basename, ' '.join(arguments))
            print 'ce      : %s' % ce
            print 'reset_ce: %s' % reset_ce

        # check that the application exists
        mode = os.stat(path)[ST_MODE]
        if not S_ISREG(mode):
            return False

        connection = Connection('vxworks', verbose)
        connection.read_until('-> ')
        connection.write('cd "%s"\n' % dirname)
        connection.read_until('-> ')
        connection.write('sysmc "%s -bcs=0 init"\n' % ce)
        connection.read_until('-> ')
        connection.write('runmc "%s %s %s"\n' % (ce, basename, ' '.join(arguments)))
        rv = connection.read_reply()
        connection.write('sysmc "%s reset"\n' % reset_ce)
        connection.read_until('-> ')
        connection.write('exit\n')
        connection.read_all()
        return rv, ''
