from qm.fields import TextField
from qm.test.classes import ssh_host
import sys
import os

class RemoteHost(ssh_host.SSHHost):

    # The wrapper script name to be used to invoke
    # remote programs
    wrapper = TextField()


    def Run(self, path, arguments, environment = None, timeout = -1,
            relative = False):
        
        if self.wrapper:
            arguments = [path] + arguments[:]
            path = self.wrapper
        return super(RemoteHost, self).Run(path, arguments,
                                           environment, timeout, relative)
